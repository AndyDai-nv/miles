from collections.abc import Mapping, Sequence

from miles.backends.dynamo_utils.config import Address, DynamoConfig, runtime_env


def frontend_launch(
    config: DynamoConfig,
    *,
    address: Address,
    interpreter_prefix: Sequence[str],
    inherited_env: Mapping[str, str],
) -> tuple[list[str], dict[str, str]]:
    if not interpreter_prefix:
        raise ValueError("frontend requires a Python interpreter")
    argv = [
        *interpreter_prefix,
        "-m",
        "dynamo.frontend",
        "--http-host",
        address.host,
        "--http-port",
        str(address.port),
        "--namespace",
        config.namespace,
        "--router-mode",
        "round-robin",
        # The controller must publish initial weights before any worker can register.
        "--router-min-initial-workers",
        "0",
        "--discovery-backend",
        config.discovery.backend,
        "--request-plane",
        "tcp",
        "--response-plane",
        "tcp",
        "--event-plane",
        "zmq",
    ]
    env = runtime_env(config, inherited_env=inherited_env)
    env["DYN_SGLANG_ENABLE_GENERATE"] = "1"
    return argv, env
