from collections.abc import Mapping

from miles.backends.dynamo_utils.config import Address, DynamoConfig, EngineBinding, launch_args, runtime_env


def sidecar_launch(
    config: DynamoConfig,
    *,
    binding: EngineBinding,
    inherited_env: Mapping[str, str],
    executable: str = "dynamo-sglang-sidecar",
    system_host: str = "0.0.0.0",
) -> tuple[list[str], dict[str, str]]:
    if binding not in config.engines:
        raise ValueError("engine binding is not part of this run")
    if not executable.strip():
        raise ValueError("sidecar requires an executable")
    listener = Address(host=system_host, port=binding.sidecar.port)
    if listener.host not in ("0.0.0.0", "::", binding.sidecar.host):
        raise ValueError("sidecar system listener does not match the binding")
    managed = {
        "--namespace": config.namespace,
        "--component": "backend",
        "--endpoint": "generate",
        "--grpc-endpoint": binding.grpc.url,
        "--controller-managed": None,
        "--unregister-on-pause": "true",
    }
    managed_env = dict(
        # Dynamo's system listener constructs host:port without adding IPv6 brackets.
        DYN_SYSTEM_HOST=f"[{listener.host}]" if ":" in listener.host else listener.host,
        DYN_SYSTEM_PORT=str(listener.port),
        DYN_COMPONENT="backend",
        DYN_ENDPOINT="generate",
        DYN_SIDECAR_GRPC_ENDPOINT=binding.grpc.url,
        DYN_SGLANG_CONTROLLER_MANAGED="true",
        DYN_SGLANG_UNREGISTER_ON_PAUSE="true",
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
    argv = [executable, *launch_args(config.sidecar, managed=managed)]
    env = runtime_env(config, inherited_env=inherited_env, options=config.sidecar, managed_env=managed_env)
    return argv, env
