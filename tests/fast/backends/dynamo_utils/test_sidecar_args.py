import os
import shlex
import subprocess

import pytest
from tests.fast.backends.dynamo_utils.test_config import config_dict

from miles.backends.dynamo_utils.config import DynamoConfig
from miles.backends.dynamo_utils.sidecar_args import sidecar_launch


def test_standalone_upstream_command_and_environment():
    config = DynamoConfig.model_validate(config_dict())
    inherited = {
        "PATH": "/bin",
        "DYN_NAMESPACE_WORKER_SUFFIX": "wrong",
        "DYN_SGLANG_CONTROLLER_MANAGED": "true",
        "DYN_SGLANG_UNREGISTER_ON_PAUSE": "true",
        "DYN_SGLANG_POLICY_VERSION_TAINTS": "true",
        "DYN_ENABLE_RL": "true",
        "DYN_SYSTEM_PORT": "9000",
    }
    before = inherited.copy()
    argv, env = sidecar_launch(
        config,
        binding=config.engines[0],
        inherited_env=inherited,
        executable="/build with spaces/dynamo-sglang-sidecar",
    )
    assert inherited == before
    assert argv == [
        "/build with spaces/dynamo-sglang-sidecar",
        "--namespace",
        "run-a",
        "--component",
        "backend",
        "--endpoint",
        "generate",
        "--grpc-endpoint",
        "http://engine-0:30001",
    ]
    assert shlex.split(shlex.join(argv)) == argv
    assert env["DYN_SYSTEM_HOST"] == "0.0.0.0"
    assert env["DYN_SYSTEM_PORT"] == "8081"
    assert "DYN_SGLANG_POLICY_VERSION_TAINTS" not in env
    assert env["DYN_ENABLE_RL"] == "true"
    assert "DYN_SGLANG_CONTROLLER_MANAGED" not in env
    assert "DYN_SGLANG_UNREGISTER_ON_PAUSE" not in env
    assert env["DYN_REQUEST_PLANE"] == env["DYN_RESPONSE_PLANE"] == "tcp"
    assert env["DYN_EVENT_PLANE"] == "zmq"
    assert "DYN_NAMESPACE_WORKER_SUFFIX" not in env
    assert "dynamo.sglang" not in argv
    assert "--defer-serving" not in argv
    assert "--require-weight-version-fence" not in argv


def test_file_discovery_and_ipv6(tmp_path):
    data = config_dict()
    data["discovery"] = {"backend": "file", "root": str(tmp_path)}
    data["engines"] = data["engines"][:1]
    for key in ("http", "grpc", "sidecar"):
        data["engines"][0][key]["host"] = "::1"
    config = DynamoConfig.model_validate(data)
    argv, env = sidecar_launch(config, binding=config.engines[0], inherited_env={}, system_host="::")
    assert argv[argv.index("--grpc-endpoint") + 1] == "http://[::1]:30001"
    assert env["DYN_SYSTEM_HOST"] == "[::]"
    assert env["DYN_FILE_KV"] == str(tmp_path)


@pytest.mark.parametrize(
    "kwargs", [{"system_host": "another-worker"}, {"system_host": "http://localhost"}, {"executable": ""}]
)
def test_invalid_launch(kwargs):
    config = DynamoConfig.model_validate(config_dict())
    with pytest.raises(ValueError):
        sidecar_launch(config, binding=config.engines[0], inherited_env={}, **kwargs)


def test_foreign_binding():
    config = DynamoConfig.model_validate(config_dict())
    binding = config.engines[0].model_copy(update={"name": "another-run"})
    with pytest.raises(ValueError, match="not part"):
        sidecar_launch(config, binding=binding, inherited_env={})


def test_configured_listener():
    data = config_dict()
    data["sidecar"] = {"system_host": "engine-0"}
    config = DynamoConfig.model_validate(data)
    _, env = sidecar_launch(config, binding=config.engines[0], inherited_env={})
    assert env["DYN_SYSTEM_HOST"] == "engine-0"
    with pytest.raises(ValueError, match="conflicts"):
        sidecar_launch(config, binding=config.engines[0], inherited_env={}, system_host="0.0.0.0")


def test_sidecar_cli_help_contract():
    binary = os.environ.get("MILES_TEST_DYNAMO_SIDECAR_BINARY")
    if binary is None:
        pytest.skip("set MILES_TEST_DYNAMO_SIDECAR_BINARY to an upstream sidecar build")
    data = config_dict()
    data["sidecar"] = {
        "grpc_connections": 16,
        "grpc_connect_attempt_timeout_secs": 20,
        "grpc_retry_interval_secs": 2,
        "grpc_startup_deadline_secs": 900,
        "extra_args": ["--bootstrap-host", "worker-a", "--dyn-tool-call-parser", "qwen25"],
    }
    config = DynamoConfig.model_validate(data)
    argv, env = sidecar_launch(config, binding=config.engines[0], inherited_env=os.environ, executable=binary)
    result = subprocess.run([*argv, "--help"], env=env, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert "--grpc-endpoint" in result.stdout
    assert "--grpc-startup-deadline-secs" in result.stdout


def test_native_transport_and_model_options():
    data = config_dict()
    data["sidecar"] = {
        "grpc_connections": 16,
        "grpc_connect_attempt_timeout_secs": 20,
        "grpc_retry_interval_secs": 2,
        "grpc_startup_deadline_secs": 900,
        "extra_args": ["--bootstrap-host", "worker-a", "--exclude-tools-when-tool-choice-none", "false"],
        "env": {"DYN_LOG": "debug"},
    }
    config = DynamoConfig.model_validate(data)
    argv, env = sidecar_launch(config, binding=config.engines[0], inherited_env={})
    for field, value in data["sidecar"].items():
        if field.startswith("grpc_"):
            assert argv[argv.index("--" + field.replace("_", "-")) + 1] == str(value)
            assert env[f"DYN_SIDECAR_{field.upper()}"] == str(value)
    assert argv[-4:] == data["sidecar"]["extra_args"]
    assert env["DYN_LOG"] == "debug"


def test_native_cli_and_env_defaults_are_not_overridden():
    data = config_dict()
    data["sidecar"] = {
        "extra_args": ["--grpc-connections", "16"],
        "env": {"DYN_SIDECAR_GRPC_RETRY_INTERVAL_SECS": "3"},
    }
    config = DynamoConfig.model_validate(data)
    argv, env = sidecar_launch(config, binding=config.engines[0], inherited_env={"DYN_SIDECAR_GRPC_CONNECTIONS": "4"})
    assert argv[-2:] == ["--grpc-connections", "16"]
    assert env["DYN_SIDECAR_GRPC_CONNECTIONS"] == "4"
    assert env["DYN_SIDECAR_GRPC_RETRY_INTERVAL_SECS"] == "3"


@pytest.mark.parametrize(
    "flag",
    [
        "--namespace=other",
        "--grpc-endpoint",
        "--controller-managed=false",
        "--no-controller-managed",
        "--unregister-on-pause=false",
        "--policy-version-taints",
        "--defer-serving",
        "--require-weight-version-fence",
        "--component",
        "--endpoint",
    ],
)
def test_sidecar_owned_flags_cannot_be_overridden(flag):
    data = config_dict()
    data["sidecar"] = {"extra_args": [flag]}
    config = DynamoConfig.model_validate(data)
    with pytest.raises(ValueError, match="managed"):
        sidecar_launch(config, binding=config.engines[0], inherited_env={})


@pytest.mark.parametrize(
    "options",
    [
        {"grpc_connections": 16, "extra_args": ["--grpc-connections", "8"]},
        {"grpc_connections": 16, "env": {"DYN_SIDECAR_GRPC_CONNECTIONS": "8"}},
        {"env": {"DYN_SGLANG_CONTROLLER_MANAGED": "false"}},
        {"env": {"DYN_SGLANG_UNREGISTER_ON_PAUSE": "false"}},
        {"env": {"DYN_SGLANG_POLICY_VERSION_TAINTS": "true"}},
        {"env": {"DYN_SYSTEM_PORT": "9000"}},
    ],
)
def test_conflicting_sidecar_configuration(options):
    data = config_dict()
    data["sidecar"] = options
    config = DynamoConfig.model_validate(data)
    with pytest.raises(ValueError):
        sidecar_launch(config, binding=config.engines[0], inherited_env={})


@pytest.mark.parametrize("scope", ["shared", "sidecar"])
@pytest.mark.parametrize(
    "key",
    ["DYN_SGLANG_CONTROLLER_MANAGED", "DYN_SGLANG_UNREGISTER_ON_PAUSE", "DYN_SGLANG_POLICY_VERSION_TAINTS"],
)
def test_explicit_fork_environment_is_rejected(scope, key):
    data = config_dict()
    if scope == "shared":
        data["env"] = {key: "true"}
    else:
        data["sidecar"] = {"env": {key: "true"}}
    config = DynamoConfig.model_validate(data)
    with pytest.raises(ValueError, match=key):
        sidecar_launch(config, binding=config.engines[0], inherited_env={})
