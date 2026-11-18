import copy
import json
from unittest.mock import AsyncMock

import httpx
import pytest

from miles.backends.sglang_utils.sglang_api_client import WorkerType
from miles.router.dynamo.serving_registry import DynamoServingRegistry, DynamoServingRegistryFactory
from miles.utils.http_utils import GeneralHttpClientProvider
from miles.utils.workers.worker_spec import HostAndPort


def test_registry_factory_preserves_explicit_rollout_only_mode():
    factory = DynamoServingRegistryFactory(provider=AsyncMock(), timeout_seconds=1, debug_rollout_only=True)
    first = factory.for_worker(worker_name="engine-0", api_key=None)
    second = factory.for_worker(worker_name="engine-1", api_key=None)
    assert first.debug_rollout_only is True
    assert second.debug_rollout_only is True
    assert first is not second


@pytest.fixture
async def admission(monkeypatch):
    status = {
        "status": "ok",
        "session": "sidecar-1",
        "revision": 0,
        "applied_revision": 0,
        "published": False,
        "admitted": False,
        "engine_ready": True,
        "last_error": None,
        "observed": {"engine_instance_id": 1, "observation_epoch": 1, "weight_version": "5"},
        "expected": None,
    }
    calls = []

    def handle(request):
        action = request.url.path.rsplit("/", 1)[1]
        payload = json.loads(request.content)
        calls.append((action, payload))
        if action != "status":
            assert payload["session"] == status["session"]
            assert payload["revision"] > status["revision"]
            status.update(
                revision=payload["revision"],
                applied_revision=payload["revision"],
                admitted=action == "admit",
                published=action == "admit",
                expected=payload.get("expected"),
            )
        return httpx.Response(200, json=copy.deepcopy(status))

    client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    monkeypatch.setattr(GeneralHttpClientProvider, "client", lambda: client)
    provider = AsyncMock()
    provider.get_addrs.return_value = {"dynamo_system": HostAndPort(host="127.0.0.1", port=15000)}
    registry = DynamoServingRegistry(provider=provider, worker_name="engine-0", api_key=None, timeout_seconds=0.1)
    yield registry, status, calls
    await client.aclose()


@pytest.mark.asyncio
async def test_update_withdraws_then_admits_trainer_version(admission):
    registry, status, calls = admission
    await registry.prepare_update(worker_url="http://engine:8000")
    assert status["admitted"] is False
    await registry.finish_update(worker_url="http://engine:8000", weight_version="5")
    assert [action for action, _ in calls if action != "status"] == ["withdraw", "admit"]
    assert status["expected"]["weight_version"] == "5"
    assert status["published"] is True


@pytest.mark.asyncio
async def test_wrong_version_is_not_admitted(admission):
    registry, status, calls = admission
    with pytest.raises((RuntimeError, TimeoutError)):
        await registry.finish_update(worker_url="http://engine:8000", weight_version="6")
    assert not any(action == "admit" for action, _ in calls)


@pytest.mark.asyncio
async def test_restart_requires_new_session_and_explicit_approval(admission):
    registry, status, calls = admission
    await registry.finish_update(worker_url="http://engine:8000", weight_version="5")
    status.update(session="sidecar-2", revision=0, applied_revision=0, admitted=False, published=False)
    status["observed"]["engine_instance_id"] = 2
    with pytest.raises(RuntimeError, match="reconciled"):
        await registry.finish_update(worker_url="http://engine:8000", weight_version="5")
    replacement = DynamoServingRegistry(
        provider=registry.provider, worker_name="engine-0", api_key=None, timeout_seconds=0.1
    )
    await replacement.finish_update(worker_url="http://engine:8000", weight_version="5")
    commands = [payload for action, payload in calls if action == "admit"]
    assert commands[-1]["session"] == "sidecar-2"
    assert commands[-1]["expected"]["engine_instance_id"] == 2


@pytest.mark.asyncio
async def test_pending_to_serving_does_not_reissue_an_existing_grant(admission):
    registry, status, calls = admission
    await registry.finish_update(worker_url="http://engine:8000", weight_version="5")
    await registry.register(worker_url="http://engine:8000", worker_type=WorkerType.REGULAR, bootstrap_port=None)
    assert len([action for action, _ in calls if action == "admit"]) == 1


@pytest.mark.asyncio
async def test_lost_grant_reaches_native_health_check(admission):
    registry, status, calls = admission
    await registry.finish_update(worker_url="http://engine:8000", weight_version="5")
    await registry.check_health(worker_url="http://engine:8000", serving=True)
    status.update(admitted=False, published=False)
    with pytest.raises(RuntimeError, match="revoked"):
        await registry.check_health(worker_url="http://engine:8000", serving=True)


@pytest.mark.asyncio
async def test_started_sidecar_can_lag_engine_readiness(admission, monkeypatch):
    registry, status, calls = admission
    original = GeneralHttpClientProvider.client()
    post = original.post
    attempts = 0

    async def late_start(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise httpx.ConnectError("sidecar is still starting")
        return await post(*args, **kwargs)

    monkeypatch.setattr(original, "post", late_start)
    monkeypatch.setattr("miles.router.dynamo.serving_registry.asyncio.sleep", AsyncMock())
    await registry.finish_update(worker_url="http://engine:8000", weight_version="5")
    assert status["published"] is True


@pytest.mark.asyncio
async def test_rollout_only_update_admits_the_engine_reported_version(admission, monkeypatch):
    registry, status, calls = admission
    registry.debug_rollout_only = True
    monkeypatch.setattr(
        "miles.router.dynamo.serving_registry.SGLangApiClient.get_weight_version", AsyncMock(return_value="5")
    )

    await registry.prepare_update(worker_url="http://engine:8000")
    await registry.finish_update(worker_url="http://engine:8000", weight_version=None)

    assert [action for action, _ in calls if action != "status"] == ["withdraw", "admit"]
    assert status["published"] is True
    assert status["expected"]["weight_version"] == "5"


@pytest.mark.asyncio
async def test_normal_training_rejects_a_missing_first_version(admission):
    registry, status, calls = admission
    with pytest.raises(ValueError, match="completed weight version"):
        await registry.finish_update(worker_url="http://engine:8000", weight_version=None)
    assert not any(action == "admit" for action, _ in calls)
    assert status["published"] is False


@pytest.mark.asyncio
async def test_rollout_only_cannot_drop_an_established_trainer_version(admission):
    registry, status, calls = admission
    registry.debug_rollout_only = True
    await registry.finish_update(worker_url="http://engine:8000", weight_version="5")
    with pytest.raises(ValueError, match="completed weight version"):
        await registry.finish_update(worker_url="http://engine:8000", weight_version=None)
    assert len([action for action, _ in calls if action == "admit"]) == 1
    assert status["expected"]["weight_version"] == "5"


@pytest.mark.parametrize("engine_version", [None, ""])
@pytest.mark.asyncio
async def test_rollout_only_rejects_an_empty_engine_version(admission, monkeypatch, engine_version):
    registry, status, calls = admission
    registry.debug_rollout_only = True
    monkeypatch.setattr(
        "miles.router.dynamo.serving_registry.SGLangApiClient.get_weight_version",
        AsyncMock(return_value=engine_version),
    )
    with pytest.raises(ValueError, match="non-empty weight version"):
        await registry.finish_update(worker_url="http://engine:8000", weight_version=None)
    assert not any(action == "admit" for action, _ in calls)
    assert status["published"] is False


@pytest.fixture
def route_errors(admission, monkeypatch):
    client = GeneralHttpClientProvider.client()
    original_post = client.post

    def install(codes):
        responses = iter(codes)

        async def post(url, **kwargs):
            code = next(responses, 200)
            if code != 200:
                return httpx.Response(code, request=httpx.Request("POST", url))
            return await original_post(url, **kwargs)

        monkeypatch.setattr(client, "post", post)

    return install


@pytest.mark.asyncio
async def test_initial_status_404_waits_for_the_admission_route(admission, route_errors):
    registry, status, calls = admission
    registry.timeout_seconds = 1
    route_errors([404])
    await registry.prepare_update(worker_url="http://engine:8000")
    assert registry._session == "sidecar-1"
    assert [action for action, _ in calls if action != "status"] == ["withdraw"]


@pytest.mark.asyncio
async def test_established_session_status_404_is_not_retried(admission, route_errors):
    registry, status, calls = admission
    await registry.finish_update(worker_url="http://engine:8000", weight_version="5")
    route_errors([404])
    with pytest.raises(httpx.HTTPStatusError) as error:
        await registry.prepare_update(worker_url="http://engine:8000")
    assert error.value.response.status_code == 404
    assert [action for action, _ in calls if action != "status"] == ["admit"]


@pytest.mark.asyncio
async def test_mutation_404_is_not_retried_as_route_readiness(admission, route_errors):
    registry, status, calls = admission
    route_errors([200, 404])
    with pytest.raises(httpx.HTTPStatusError) as error:
        await registry.prepare_update(worker_url="http://engine:8000")
    assert error.value.response.status_code == 404
    assert not any(action == "withdraw" for action, _ in calls)


@pytest.mark.asyncio
async def test_initial_status_404_is_bounded_by_the_control_deadline(admission, route_errors):
    registry, status, calls = admission
    registry.timeout_seconds = 0.01
    route_errors([404] * 20)
    with pytest.raises(TimeoutError):
        await registry.prepare_update(worker_url="http://engine:8000")
    assert registry._session is None
    assert not any(action in {"admit", "withdraw"} for action, _ in calls)


@pytest.mark.asyncio
async def test_initial_authentication_failure_is_not_route_readiness(admission, route_errors):
    registry, status, calls = admission
    route_errors([401])
    with pytest.raises(httpx.HTTPStatusError) as error:
        await registry.prepare_update(worker_url="http://engine:8000")
    assert error.value.response.status_code == 401
    assert registry._session is None
