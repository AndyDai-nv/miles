import argparse
import json
import shlex
import sys

import pytest

from miles.backends.dynamo_utils.arguments import add_dynamo_arguments
from miles.backends.dynamo_utils.dynamo_config import resolve_dynamo_config
from miles.backends.dynamo_utils.engine_args import (
    compute_dynamo_engine_args,
    compute_dynamo_engine_argv,
    compute_dynamo_engine_env_vars,
    compute_dynamo_engine_launch_cmd,
    compute_dynamo_worker_endpoint,
    dynamo_engine_args_to_argv,
)

MODEL_ID = "Qwen/Qwen3-8B"
MODEL_NAMESPACE = "miles-testrun-qwen-qwen3-8b-a82fd547"
SGLANG_ARGV = [
    "--model-path",
    "Qwen/Qwen3-8B",
    "--host",
    "10.0.0.7",
    "--port",
    "31000",
    "--tp-size",
    "2",
    "--trust-remote-code",
]


def _config(*argv: str):
    parser = add_dynamo_arguments(argparse.ArgumentParser())
    args = parser.parse_args(["--rollout-backend", "dynamo", *argv])
    args.run_uuid = "testrun"
    return resolve_dynamo_config(args)


def _kv_events_argv(*, publisher: str = "zmq", endpoint: str = "tcp://*:32000") -> list[str]:
    value = json.dumps(
        {
            "publisher": publisher,
            "topic": "kv-events",
            "endpoint": endpoint,
            "enable_kv_cache_events": True,
        }
    )
    return [*SGLANG_ARGV, "--page-size", "64", "--kv-events-config", value]


def test_default_dynamo_args_are_exact():
    assert compute_dynamo_engine_args(_config(), model_id=MODEL_ID) == {
        "namespace": MODEL_NAMESPACE,
        "endpoint": f"dyn://{MODEL_NAMESPACE}.backend.generate",
        "discovery-backend": None,
        "request-plane": "tcp",
        "event-plane": "zmq",
        "enable-rl": True,
    }


def test_default_dynamo_argv_is_exact():
    args = compute_dynamo_engine_args(_config(), model_id=MODEL_ID)
    assert dynamo_engine_args_to_argv(args) == [
        "--namespace",
        MODEL_NAMESPACE,
        "--endpoint",
        f"dyn://{MODEL_NAMESPACE}.backend.generate",
        "--request-plane",
        "tcp",
        "--event-plane",
        "zmq",
        "--enable-rl",
    ]


def test_complete_argv_wraps_the_unchanged_sglang_argv():
    argv = compute_dynamo_engine_argv(_config(), model_id=MODEL_ID, sglang_argv=SGLANG_ARGV)
    assert argv[:3] == [sys.executable, "-m", "dynamo.sglang"]
    assert argv[-len(SGLANG_ARGV) :] == SGLANG_ARGV
    assert argv.count("--model-path") == 1
    assert argv.count("--port") == 1


def test_launch_command_is_shell_safe_and_roundtrips():
    sglang_argv = [*SGLANG_ARGV, "--chat-template", "/tmp/template with spaces.jinja"]
    command = compute_dynamo_engine_launch_cmd(_config(), model_id=MODEL_ID, sglang_argv=sglang_argv)
    assert shlex.split(command) == compute_dynamo_engine_argv(_config(), model_id=MODEL_ID, sglang_argv=sglang_argv)


def test_explicit_discovery_backend_is_rendered():
    argv = compute_dynamo_engine_argv(
        _config("--dynamo-discovery-backend", "file"), model_id=MODEL_ID, sglang_argv=SGLANG_ARGV
    )
    assert argv[argv.index("--discovery-backend") + 1] == "file"


def test_disabled_rl_is_explicit_even_if_the_parent_environment_enables_it():
    argv = compute_dynamo_engine_argv(_config("--no-dynamo-enable-rl"), model_id=MODEL_ID, sglang_argv=SGLANG_ARGV)
    assert "--no-enable-rl" in argv
    assert "--enable-rl" not in argv


def test_different_models_get_disjoint_worker_endpoints():
    first = compute_dynamo_worker_endpoint(_config(), model_id=MODEL_ID)
    second = compute_dynamo_worker_endpoint(_config(), model_id="Qwen/Qwen3-14B")
    assert first != second
    assert first == f"dyn://{MODEL_NAMESPACE}.backend.generate"


def test_engine_env_is_exact_and_keeps_control_and_sglang_ports_separate():
    assert compute_dynamo_engine_env_vars(_config(), model_id=MODEL_ID, system_port=30000, sglang_port=31000) == {
        "DYN_SYSTEM_PORT": "30000",
        "DYN_NAMESPACE": MODEL_NAMESPACE,
        "DYN_ENDPOINT": f"dyn://{MODEL_NAMESPACE}.backend.generate",
        "DYN_REQUEST_PLANE": "tcp",
        "DYN_EVENT_PLANE": "zmq",
        "PYTHONHASHSEED": "0",
    }


def test_engine_env_carries_explicit_discovery_and_file_store():
    config = _config(
        "--dynamo-discovery-backend",
        "file",
        "--dynamo-file-kv-path",
        "/shared/dynamo discovery",
    )
    env = compute_dynamo_engine_env_vars(config, model_id=MODEL_ID, system_port=30000, sglang_port=31000)
    assert env["DYN_DISCOVERY_BACKEND"] == "file"
    assert env["DYN_FILE_KV"] == "/shared/dynamo discovery"


@pytest.mark.parametrize("port", [0, -1, 65536, True, 3.5, "30000"])
def test_invalid_system_port_is_rejected(port):
    with pytest.raises(ValueError, match="system_port must be an integer between"):
        compute_dynamo_engine_env_vars(_config(), model_id=MODEL_ID, system_port=port, sglang_port=31000)


def test_invalid_sglang_port_is_rejected():
    with pytest.raises(ValueError, match="sglang_port must be an integer between"):
        compute_dynamo_engine_env_vars(_config(), model_id=MODEL_ID, system_port=30000, sglang_port=65536)


def test_system_and_sglang_ports_must_not_be_conflated():
    with pytest.raises(ValueError, match="must be different"):
        compute_dynamo_engine_env_vars(_config(), model_id=MODEL_ID, system_port=31000, sglang_port=31000)


def test_shell_command_string_is_not_accepted_as_sglang_argv():
    with pytest.raises(TypeError, match="sequence of tokens"):
        compute_dynamo_engine_argv(_config(), model_id=MODEL_ID, sglang_argv="--model-path model")


def test_non_string_sglang_token_is_rejected():
    with pytest.raises(TypeError, match="every sglang_argv token"):
        compute_dynamo_engine_argv(_config(), model_id=MODEL_ID, sglang_argv=["--port", 31000])


@pytest.mark.parametrize("flag", ["--namespace", "--endpoint=other", "--enable-rl", "--no-enable-rl"])
def test_sglang_argv_cannot_override_dynamo_owned_flags(flag):
    with pytest.raises(ValueError, match="Dynamo-owned options"):
        compute_dynamo_engine_argv(_config(), model_id=MODEL_ID, sglang_argv=[*SGLANG_ARGV, flag])


def test_round_robin_rejects_unconsumed_kv_events():
    with pytest.raises(ValueError, match="does not consume them"):
        compute_dynamo_engine_argv(_config(), model_id=MODEL_ID, sglang_argv=_kv_events_argv())


def test_kv_routing_with_events_requires_an_explicit_allocated_endpoint():
    with pytest.raises(ValueError, match="requires SGLang argv"):
        compute_dynamo_engine_argv(_config("--dynamo-router-mode", "kv"), model_id=MODEL_ID, sglang_argv=SGLANG_ARGV)


def test_kv_routing_accepts_a_matching_zmq_event_config():
    sglang_argv = _kv_events_argv()
    argv = compute_dynamo_engine_argv(
        _config("--dynamo-router-mode", "kv"), model_id=MODEL_ID, sglang_argv=sglang_argv
    )
    assert argv[-len(sglang_argv) :] == sglang_argv


def test_kv_events_config_accepts_the_equals_form():
    config_json = _kv_events_argv()[-1]
    sglang_argv = [*SGLANG_ARGV, f"--kv-events-config={config_json}"]
    argv = compute_dynamo_engine_argv(
        _config("--dynamo-router-mode", "kv"), model_id=MODEL_ID, sglang_argv=sglang_argv
    )
    assert argv[-1] == f"--kv-events-config={config_json}"


def test_duplicate_kv_events_config_is_rejected():
    sglang_argv = _kv_events_argv()
    with pytest.raises(ValueError, match="at most once"):
        compute_dynamo_engine_argv(
            _config("--dynamo-router-mode", "kv"),
            model_id=MODEL_ID,
            sglang_argv=[*sglang_argv, "--kv-events-config", sglang_argv[-1]],
        )


def test_invalid_kv_events_json_is_rejected():
    with pytest.raises(ValueError, match="valid JSON"):
        compute_dynamo_engine_argv(
            _config("--dynamo-router-mode", "kv"),
            model_id=MODEL_ID,
            sglang_argv=[*SGLANG_ARGV, "--kv-events-config", "not-json"],
        )


def test_kv_routing_rejects_an_event_publisher_that_disagrees_with_the_frontend():
    with pytest.raises(ValueError, match="publisher must be 'zmq'"):
        compute_dynamo_engine_argv(
            _config("--dynamo-router-mode", "kv"),
            model_id=MODEL_ID,
            sglang_argv=_kv_events_argv(publisher="nats"),
        )


def test_kv_routing_rejects_a_missing_event_endpoint():
    with pytest.raises(ValueError, match="non-empty endpoint"):
        compute_dynamo_engine_argv(
            _config("--dynamo-router-mode", "kv"),
            model_id=MODEL_ID,
            sglang_argv=_kv_events_argv(endpoint=""),
        )


def test_prediction_only_kv_routing_does_not_require_worker_events():
    argv = compute_dynamo_engine_argv(
        _config("--dynamo-router-mode", "kv", "--no-dynamo-router-kv-events"),
        model_id=MODEL_ID,
        sglang_argv=SGLANG_ARGV,
    )
    assert "--kv-events-config" not in argv


def test_rendered_dynamo_args_are_accepted_by_installed_dynamo(monkeypatch):
    """Exercise the real argument groups in integration images without requiring Dynamo in unit CI."""
    runtime_args = pytest.importorskip("dynamo.common.configuration.groups.runtime_args")
    backend_args = pytest.importorskip("dynamo.sglang.backend_args")
    for name in (
        "DYN_NAMESPACE",
        "DYN_ENDPOINT",
        "DYN_DISCOVERY_BACKEND",
        "DYN_REQUEST_PLANE",
        "DYN_EVENT_PLANE",
        "DYN_SGL_ENABLE_RL",
    ):
        monkeypatch.delenv(name, raising=False)

    parser = argparse.ArgumentParser()
    runtime_args.DynamoRuntimeArgGroup().add_arguments(parser)
    backend_args.DynamoSGLangArgGroup().add_arguments(parser)
    rendered = dynamo_engine_args_to_argv(compute_dynamo_engine_args(_config(), model_id=MODEL_ID))
    parsed = parser.parse_args(rendered)

    assert parsed.namespace == MODEL_NAMESPACE
    assert parsed.endpoint == f"dyn://{MODEL_NAMESPACE}.backend.generate"
    assert parsed.request_plane == "tcp"
    assert parsed.event_plane == "zmq"
    assert parsed.enable_rl is True
