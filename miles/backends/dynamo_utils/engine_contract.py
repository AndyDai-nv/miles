from collections.abc import Mapping

from miles.backends.dynamo_utils.config import Address, DynamoConfig, EngineBinding

_SUPPORTED_OPTIONS = {
    "dp_size": 1,
    "pp_size": 1,
    "nnodes": 1,
    "node_rank": 0,
    "tokenizer_worker_num": 1,
    "disaggregation_mode": "null",
    "enable_dp_attention": False,
    "enable_lora": False,
    "smg_grpc_mode": False,
    "grpc_mode": False,
    "use_ray": False,
    "encoder_only": False,
    "api_key": None,
    "admin_api_key": None,
    "sidecar": None,
    "sidecar_args": None,
}


def configure_engine_args(
    config: DynamoConfig, *, binding: EngineBinding, overrides: Mapping[str, object]
) -> dict[str, object]:
    """Merge explicit overrides before SGLang defaults; use server_args_to_argv to render."""
    required = _required_args(config, binding=binding)
    result = dict(overrides)
    for name, value in required.items():
        if name in result and result[name] is not None and not _matches(result[name], value):
            raise ValueError(f"SGLang {name} conflicts with the sidecar launch contract")
        result[name] = value
    result.setdefault("host", "::" if ":" in binding.http.host else "0.0.0.0")
    for name, value in _SUPPORTED_OPTIONS.items():
        result.setdefault(name, value)
    validate_engine_info(config, binding=binding, server_info=result)
    return result


def validate_engine_info(config: DynamoConfig, *, binding: EngineBinding, server_info: Mapping[str, object]) -> None:
    """Check resolved server-info fields, not RPC availability or serving admission."""
    required = {**_required_args(config, binding=binding), **_SUPPORTED_OPTIONS}
    for name, expected in required.items():
        if name not in server_info or not _matches(server_info[name], expected):
            raise ValueError(f"SGLang {name} does not satisfy the sidecar launch contract")
    listener = Address(host=server_info.get("host", ""), port=binding.http.port)
    if listener.host not in ("0.0.0.0", "::", binding.http.host):
        raise ValueError("SGLang listener does not match the engine binding")
    occupied_ports = {binding.http.port, binding.grpc.port}
    if binding.sidecar.host == binding.http.host:
        occupied_ports.add(binding.sidecar.port)
    for name in ("nccl_port", "metrics_port", "metrics_http_port", "engine_info_bootstrap_port", "gated_launch_port"):
        if server_info.get(name) in occupied_ports:
            raise ValueError(f"SGLang {name} overlaps an engine or sidecar port")


def _required_args(config: DynamoConfig, *, binding: EngineBinding) -> dict[str, object]:
    if binding not in config.engines:
        raise ValueError("engine binding is not part of this run")
    return {
        "model_path": config.model_path,
        "port": binding.http.port,
        "grpc_port": binding.grpc.port,
        "tp_size": binding.tensor_parallel_size,
        "incremental_streaming_output": True,
    }


def _matches(actual: object, expected: object) -> bool:
    return type(actual) is type(expected) and actual == expected
