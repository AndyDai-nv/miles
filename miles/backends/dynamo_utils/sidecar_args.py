from collections.abc import Mapping

from miles.backends.dynamo_utils.config import Address, DynamoConfig, EngineBinding, runtime_env


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
    argv = [
        executable,
        "--namespace",
        config.namespace,
        "--component",
        "backend",
        "--endpoint",
        "generate",
        "--grpc-endpoint",
        binding.grpc.url,
        "--controller-managed",
        "--unregister-on-pause",
        "true",
    ]
    env = runtime_env(config, inherited_env=inherited_env)
    env.update(
        # Dynamo's system listener constructs host:port without adding IPv6 brackets.
        DYN_SYSTEM_HOST=f"[{listener.host}]" if ":" in listener.host else listener.host,
        DYN_SYSTEM_PORT=str(listener.port),
        DYN_SGLANG_POLICY_VERSION_TAINTS="false",
        DYN_ENABLE_RL="false",
    )
    return argv, env
