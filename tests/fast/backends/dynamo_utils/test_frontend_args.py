import argparse
import os
import shlex

import pytest
from tests.fast.backends.dynamo_utils.test_config import config_dict

from miles.backends.dynamo_utils.config import Address, DynamoConfig
from miles.backends.dynamo_utils.frontend_args import frontend_launch


def _launch(*, host="0.0.0.0", discovery=None):
    data = config_dict()
    if discovery is not None:
        data["discovery"] = discovery
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
        "--router-mode",
        "round-robin",
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
    assert (
        not {
            "DYN_NAMESPACE_PREFIX",
            "DYN_SYSTEM_PORT",
            "DYN_HTTP_SVC_SGLANG_GENERATE_PATH",
            "DYN_ROUTER_MIN_INITIAL_WORKERS",
        }
        & env.keys()
    )


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


def test_frontend_cli_contract(monkeypatch):
    frontend = pytest.importorskip("dynamo.frontend.frontend_args")
    argv, env = _launch()
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
    assert parsed.router_mode == "round-robin"
    assert parsed.min_initial_workers == 0
    assert (parsed.request_plane, parsed.response_plane, parsed.event_plane) == ("tcp", "tcp", "zmq")
