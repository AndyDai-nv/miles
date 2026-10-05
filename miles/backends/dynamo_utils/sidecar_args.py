from collections.abc import Mapping

from miles.backends.dynamo_utils.config import Address, DynamoConfig, EngineBinding, launch_args, runtime_env


def sidecar_launch(
    config: DynamoConfig,
    *,
    binding: EngineBinding,
    inherited_env: Mapping[str, str],
    executable: str = "dynamo-sglang-sidecar",
    system_host: str | None = None,
) -> tuple[list[str], dict[str, str]]:
    if binding not in config.engines:
        raise ValueError("engine binding is not part of this run")
    if not executable.strip():
        raise ValueError("sidecar requires an executable")
    if config.sidecar.system_host is not None:
        if system_host is not None and system_host != config.sidecar.system_host:
            raise ValueError("sidecar system host conflicts with its configured listener")
        system_host = config.sidecar.system_host
    if system_host is None:
        system_host = "::" if ":" in binding.sidecar.host else "0.0.0.0"
    listener = Address(host=system_host, port=binding.sidecar.port)
    if listener.host not in ("0.0.0.0", "::", binding.sidecar.host):
        raise ValueError("sidecar system listener does not match the binding")
    managed = {
        "--namespace": config.namespace,
        "--component": "backend",
        "--endpoint": "generate",
        "--grpc-endpoint": binding.grpc.url,
    }
    managed_env = dict(
        # Dynamo's system listener constructs host:port without adding IPv6 brackets.
        DYN_SYSTEM_HOST=f"[{listener.host}]" if ":" in listener.host else listener.host,
        DYN_SYSTEM_PORT=str(listener.port),
        DYN_COMPONENT="backend",
        DYN_ENDPOINT="generate",
        DYN_SIDECAR_GRPC_ENDPOINT=binding.grpc.url,
        # Fork-only admission settings must not leak into the upstream launch.
        DYN_SGLANG_CONTROLLER_MANAGED=None,
        DYN_SGLANG_UNREGISTER_ON_PAUSE=None,
        DYN_SGLANG_POLICY_VERSION_TAINTS=None,
    )
    for field in (
        "grpc_connections",
        "grpc_connect_attempt_timeout_secs",
        "grpc_retry_interval_secs",
        "grpc_startup_deadline_secs",
    ):
        value = getattr(config.sidecar, field)
        if value is not None:
            managed[f"--{field.replace('_', '-')}"] = str(value)
            managed_env[f"DYN_SIDECAR_{field.upper()}"] = str(value)
    reserved = (
        "--controller-managed",
        "--unregister-on-pause",
        "--policy-version-taints",
        "--defer-serving",
        "--require-weight-version-fence",
    )
    argv = [executable, *launch_args(config.sidecar, managed=managed, reserved=reserved)]
    env = runtime_env(config, inherited_env=inherited_env, options=config.sidecar, managed_env=managed_env)
    return argv, env
