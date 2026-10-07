import argparse
import os
import shlex

import pytest
from tests.fast.backends.dynamo_utils.test_config import config_dict

from miles.backends.dynamo_utils.config import Address, DynamoConfig
from miles.backends.dynamo_utils.frontend_args import frontend_env, frontend_launch


def _launch(*, host="0.0.0.0", discovery=None, frontend=None):
    data = config_dict()
    if discovery is not None:
        data["discovery"] = discovery
    if frontend is not None:
        data["frontend"] = frontend
    return frontend_launch(
        DynamoConfig.model_validate(data),
        address=Address(host=host, port=8000),
        interpreter_prefix=["/env with spaces/bin/python", "-u"],
        inherited_env={
            "PATH": "/bin",
            "DYN_NAMESPACE_PREFIX": "other",
            "DYN_SYSTEM_PORT": "8081",
            "DYN_HTTP_SVC_SGLANG_GENERATE_PATH": "/wrong",
            "DYN_ROUTER_MIN_INITIAL_WORKERS": "2",
            "DYN_SGLANG_ENABLE_GENERATE": "0",
        },
    )


def test_frontend_command_and_environment():
    argv, env = _launch()
    assert argv == [
        "/env with spaces/bin/python",
        "-u",
        "-m",
        "dynamo.frontend",
        "--http-host",
        "0.0.0.0",
        "--http-port",
        "8000",
        "--namespace",
        "run-a",
        "--router-min-initial-workers",
        "0",
        "--discovery-backend",
        "etcd",
        "--request-plane",
        "tcp",
        "--response-plane",
        "tcp",
        "--event-plane",
        "zmq",
    ]
    assert shlex.split(shlex.join(argv)) == argv
    assert env["DYN_SGLANG_ENABLE_GENERATE"] == "1"
    assert env["ETCD_ENDPOINTS"] == "http://localhost:2379"
    assert env["PATH"] == "/bin"
    assert "DYN_NAMESPACE_PREFIX" not in env
    assert env["DYN_SYSTEM_PORT"] == "8081"
    assert env["DYN_HTTP_SVC_SGLANG_GENERATE_PATH"] == "/generate"
    assert env["DYN_ROUTER_MIN_INITIAL_WORKERS"] == "0"


def test_environment_before_listener_allocation():
    data = config_dict()
    data["frontend"] = {"router_mode": "kv", "env": {"DYN_LOG": "debug"}}
    inherited = {"PATH": "/bin", "DYN_HTTP_HOST": "stale-host", "DYN_HTTP_PORT": "9999"}
    before = inherited.copy()
    env = frontend_env(DynamoConfig.model_validate(data), inherited_env=inherited)
    assert inherited == before
    assert "DYN_HTTP_HOST" not in env and "DYN_HTTP_PORT" not in env
    assert env["DYN_SGLANG_ENABLE_GENERATE"] == "1"
    assert env["DYN_ROUTER_MODE"] == "kv"
    assert env["DYN_LOG"] == "debug"
    assert env["PATH"] == "/bin"


@pytest.mark.parametrize("port", [8000, 9000])
def test_explicit_listener_environment_is_validated_after_allocation(port):
    options = {"env": {"DYN_HTTP_PORT": str(port)}}
    data = {**config_dict(), "frontend": options}
    env = frontend_env(DynamoConfig.model_validate(data), inherited_env={})
    assert "DYN_HTTP_PORT" not in env
    if port == 8000:
        _, env = _launch(frontend=options)
        assert env["DYN_HTTP_PORT"] == "8000"
    else:
        with pytest.raises(ValueError, match="DYN_HTTP_PORT conflicts"):
            _launch(frontend=options)


def test_ipv6_and_file_discovery(tmp_path):
    argv, env = _launch(host="::", discovery={"backend": "file", "root": str(tmp_path)})
    assert argv[argv.index("--http-host") + 1] == "::"
    assert argv[argv.index("--discovery-backend") + 1] == "file"
    assert env["DYN_FILE_KV"] == str(tmp_path)
    assert "ETCD_ENDPOINTS" not in env


def test_missing_interpreter():
    with pytest.raises(ValueError, match="interpreter"):
        frontend_launch(
            DynamoConfig.model_validate(config_dict()),
            address=Address(host="localhost", port=8000),
            interpreter_prefix=[],
            inherited_env={},
        )


def test_configured_listener():
    data = config_dict()
    data["frontend"] = {"address": {"host": "::", "port": 9000}}
    config = DynamoConfig.model_validate(data)
    argv, env = frontend_launch(config, interpreter_prefix=["python"], inherited_env={})
    assert argv[argv.index("--http-host") + 1] == "::"
    assert argv[argv.index("--http-port") + 1] == "9000"
    assert env["DYN_HTTP_PORT"] == "9000"
    with pytest.raises(ValueError, match="conflicts"):
        frontend_launch(
            config, address=Address(host="localhost", port=8000), interpreter_prefix=["python"], inherited_env={}
        )


def test_missing_listener():
    with pytest.raises(ValueError, match="address"):
        frontend_launch(DynamoConfig.model_validate(config_dict()), interpreter_prefix=["python"], inherited_env={})


@pytest.mark.parametrize(
    "options",
    [
        {},
        {
            "router_mode": "kv",
            "extra_args": [
                "--router-event-threads",
                "8",
                "--no-tokenizer-fallback",
                "--metrics-prefix",
                "training",
            ],
        },
    ],
)
def test_frontend_cli_contract(monkeypatch, options):
    frontend = pytest.importorskip("dynamo.frontend.frontend_args")
    argv, env = _launch(frontend=options)
    for key in tuple(os.environ):
        if key.startswith("DYN_"):
            monkeypatch.delenv(key)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    parser = argparse.ArgumentParser()
    frontend.FrontendArgGroup().add_arguments(parser)
    parsed = parser.parse_args(argv[4:])
    assert parsed.namespace == "run-a"
    assert parsed.namespace_prefix is None
    assert parsed.http_port == 8000
    assert parsed.router_mode == options.get("router_mode", "round-robin")
    assert parsed.min_initial_workers == 0
    assert (parsed.request_plane, parsed.response_plane, parsed.event_plane) == ("tcp", "tcp", "zmq")
    if options:
        assert parsed.router_event_threads == 8
        assert parsed.tokenizer_fallback is False
        assert parsed.metrics_prefix == "training"


def test_native_options_and_environment_reach_frontend():
    extra = ["--metrics-prefix", "training", "--tls-cert-path=/tls/cert.pem", "--no-tokenizer-fallback"]
    argv, env = _launch(frontend={"router_mode": "kv", "extra_args": extra, "env": {"DYN_LOG": "debug"}})
    assert argv[-len(extra) :] == extra
    assert argv[argv.index("--router-mode") + 1] == "kv"
    assert env["DYN_ROUTER_MODE"] == "kv"
    assert env["DYN_LOG"] == "debug"


def test_router_mode_can_use_native_cli_or_env_when_typed_field_is_unset():
    argv, env = _launch(frontend={"extra_args": ["--router-mode", "least-loaded"], "env": {"DYN_ROUTER_MODE": "kv"}})
    assert argv.count("--router-mode") == 1
    assert argv[-1] == "least-loaded"
    assert env["DYN_ROUTER_MODE"] == "kv"


@pytest.mark.parametrize(
    "flag",
    [
        "--namespace=wrong",
        "--namespace-prefix",
        "--http-p",
        "--router-min-initial-workers=1",
        "--request-plane",
        "--interactive",
        "-i",
    ],
)
def test_frontend_owned_flags_cannot_be_overridden(flag):
    with pytest.raises(ValueError, match="managed"):
        _launch(frontend={"extra_args": [flag]})


@pytest.mark.parametrize(
    "options",
    [
        {"router_mode": "kv", "extra_args": ["--router-mode", "random"]},
        {"router_mode": "kv", "env": {"DYN_ROUTER_MODE": "random"}},
        {"env": {"DYN_SGLANG_ENABLE_GENERATE": "0"}},
        {"env": {"DYN_HTTP_SVC_SGLANG_GENERATE_PATH": "/other"}},
    ],
)
def test_conflicting_frontend_configuration(options):
    with pytest.raises(ValueError):
        _launch(frontend=options)
