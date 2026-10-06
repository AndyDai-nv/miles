import json
import shlex
from builtins import ExceptionGroup
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from tests.fast.backends.dynamo_utils.test_stream_sample import _chunk

from miles.backends.dynamo_utils import deployment
from miles.backends.dynamo_utils.config import load_dynamo_config
from miles.backends.dynamo_utils.engine_state import EngineObservation
from miles.backends.sglang_utils.server_args_utils import parse_server_args_argv

EXAMPLE = Path(__file__).resolve().parents[4] / "examples/dynamo_sidecar"


@pytest.mark.parametrize("name", ["engine-0", "engine-1"])
def test_example_engine_roundtrips_through_native_parser(name):
    config = load_dynamo_config(EXAMPLE / "dynamo.json")
    overrides = json.loads((EXAMPLE / f"{name}.json").read_text())
    argv = deployment.engine_launch(config, name=name, overrides=overrides)
    parsed = parse_server_args_argv(argv[3:])
    binding = next(binding for binding in config.engines if binding.name == name)
    assert parsed.port == binding.http.port and parsed.grpc_port == binding.grpc.port
    assert parsed.tp_size == 2 and parsed.incremental_streaming_output
    assert parsed.kv_events_config == overrides["kv_events_config"]
    assert parsed.sidecar is None


@pytest.mark.parametrize("dry_run", [True, False])
def test_sidecar_entrypoint_uses_native_argv_and_managed_environment(monkeypatch, capsys, dry_run):
    execute = Mock()
    monkeypatch.setattr(deployment.os, "execvpe", execute)
    deployment.main(
        [
            "--config",
            str(EXAMPLE / "dynamo.json"),
            "sidecar",
            "--name",
            "engine-1",
            *(["--dry-run"] if dry_run else []),
        ]
    )
    if dry_run:
        execute.assert_not_called()
        argv = shlex.split(capsys.readouterr().out)
    else:
        executable, argv, env = execute.call_args.args
        assert executable == argv[0] == "dynamo-sglang-sidecar"
        assert env["DYN_SYSTEM_PORT"] == "8082"
        assert env["DYN_NAMESPACE"] == "miles-grpo-example"
    assert "http://127.0.0.1:31001" in argv
    assert "dynamo.sglang" not in argv


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [None, "stale", "restart", "wrong_stream"])
async def test_smoke_validates_tokens_versions_and_engine_incarnation(monkeypatch, failure):
    config = load_dynamo_config(EXAMPLE / "dynamo.json")
    before = (EngineObservation(11, "16" if failure == "stale" else "17"), EngineObservation(22, "17"))
    after = (EngineObservation(33, "17"), before[1]) if failure == "restart" else before
    monkeypatch.setattr(deployment, "observe_fleet", AsyncMock(side_effect=[before, after]))
    calls = []

    def handler(request):
        calls.append(json.loads(request.content))
        chunk = _chunk([20], text="answer", count=1, terminal={"type": "stop"})
        chunk["meta_info"]["prompt_tokens"] = 2
        if failure == "wrong_stream":
            chunk["meta_info"]["weight_version"] = "16"
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n",
        )

    client_type = httpx.AsyncClient
    monkeypatch.setattr(deployment.httpx, "AsyncClient", lambda: client_type(transport=httpx.MockTransport(handler)))
    request = dict(
        frontend_url="http://frontend:8000", weight_version="17", input_ids=[1, 2], n_samples=4, max_new_tokens=8
    )
    if failure:
        with pytest.raises((ValueError, RuntimeError, ExceptionGroup)):
            await deployment.smoke(config, **request)
    else:
        samples = await deployment.smoke(config, **request)
        assert len(samples) == 4 and len(calls) == 4
        assert all(sample.tokens == [1, 2, 20] and len(sample.rollout_log_probs) == 1 for sample in samples)
        assert all(payload["stream"] and payload["return_logprob"] for payload in calls)
    if failure == "stale":
        assert not calls
