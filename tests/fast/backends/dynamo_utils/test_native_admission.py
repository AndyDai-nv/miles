import copy
import json
from unittest.mock import AsyncMock

import httpx
import pytest

from miles.backends.dynamo_utils.serving_registry import DynamoServingRegistry
from miles.backends.sglang_utils.sglang_api_client import WorkerType
from miles.utils.http_utils import GeneralHttpClientProvider
from miles.utils.workers.worker_spec import HostAndPort


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
    monkeypatch.setattr("miles.backends.dynamo_utils.serving_registry.asyncio.sleep", AsyncMock())
    await registry.finish_update(worker_url="http://engine:8000", weight_version="5")
    assert status["published"] is True
