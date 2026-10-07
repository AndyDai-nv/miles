from collections.abc import Mapping, Sequence

from miles.backends.dynamo_utils.config import Address, DynamoConfig, launch_args, runtime_env


def frontend_launch(
    config: DynamoConfig,
    *,
    address: Address | None = None,
    interpreter_prefix: Sequence[str],
    inherited_env: Mapping[str, str],
) -> tuple[list[str], dict[str, str]]:
    if not interpreter_prefix:
        raise ValueError("frontend requires a Python interpreter")
    if config.frontend.address is not None:
        if address is not None and address != config.frontend.address:
            raise ValueError("frontend address conflicts with its configured listener")
        address = config.frontend.address
    if address is None:
        raise ValueError("frontend requires a configured or allocated address")
    managed = {
        "--http-host": address.host,
        "--http-port": str(address.port),
        "--namespace": config.namespace,
        # Frontend startup must not depend on worker discovery or initial weight sync.
        "--router-min-initial-workers": "0",
        "--discovery-backend": config.discovery.backend,
        "--request-plane": config.request_plane,
        "--response-plane": config.response_plane,
        "--event-plane": config.event_plane,
    }
    if config.frontend.router_mode is not None:
        managed["--router-mode"] = config.frontend.router_mode
    argv = [*interpreter_prefix, "-m", "dynamo.frontend"]
    argv.extend(launch_args(config.frontend, managed=managed, reserved=("--namespace-prefix", "--interactive")))
    env = frontend_env(config, inherited_env=inherited_env, address=address)
    return argv, env


def frontend_env(
    config: DynamoConfig,
    *,
    inherited_env: Mapping[str, str],
    address: Address | None = None,
) -> dict[str, str]:
    managed = {
        "DYN_ROUTER_MIN_INITIAL_WORKERS": "0",
        "DYN_SGLANG_ENABLE_GENERATE": "1",
        "DYN_HTTP_SVC_SGLANG_GENERATE_PATH": "/generate",
        "DYN_INTERACTIVE": "false",
    }
    if config.frontend.router_mode is not None:
        managed["DYN_ROUTER_MODE"] = config.frontend.router_mode
    if address is not None:
        managed.update(DYN_HTTP_HOST=address.host, DYN_HTTP_PORT=str(address.port))
    env = runtime_env(config, inherited_env=inherited_env, options=config.frontend, managed_env=managed)
    if address is None:
        # Worker env is built before port allocation. The later launch callback
        # validates explicit listener settings and supplies the resolved CLI args.
        env.pop("DYN_HTTP_HOST", None)
        env.pop("DYN_HTTP_PORT", None)
    return env
