from copy import deepcopy

import pytest
from tests.fast.backends.dynamo_utils.test_config import config_dict

from miles.backends.dynamo_utils.config import DynamoConfig
from miles.backends.dynamo_utils.engine_contract import configure_engine_args, validate_engine_info


def _config():
    return DynamoConfig.model_validate(config_dict())


def test_native_tp2_overrides_are_immutable_and_idempotent():
    config = _config()
    overrides = {"device": "cuda", "mem_fraction_static": 0.7, "base_gpu_id": 2, "trust_remote_code": True}
    before = deepcopy(overrides)
    args = configure_engine_args(config, binding=config.engines[0], overrides=overrides)
    assert overrides == before
    assert args["tp_size"] == 2
    assert args["port"] == 30000
    assert args["grpc_port"] == 30001
    assert args["incremental_streaming_output"] is True
    assert args["base_gpu_id"] == 2
    assert args["mem_fraction_static"] == 0.7
    assert args["sidecar"] is None
    assert configure_engine_args(config, binding=config.engines[0], overrides=args) == args
    validate_engine_info(config, binding=config.engines[0], server_info=args)


@pytest.mark.parametrize(
    "name,value",
    [
        ("model_path", "another-model"),
        ("port", 9000),
        ("grpc_port", 9000),
        ("tp_size", 4),
        ("incremental_streaming_output", False),
        ("incremental_streaming_output", 1),
        ("tokenizer_worker_num", 2),
        ("dp_size", 2),
        ("pp_size", 2),
        ("nnodes", 2),
        ("node_rank", 1),
        ("enable_dp_attention", True),
        ("enable_lora", True),
        ("disaggregation_mode", "prefill"),
        ("smg_grpc_mode", True),
        ("grpc_mode", True),
        ("use_ray", True),
        ("encoder_only", True),
        ("sidecar", "dynamo.sglang.sidecar"),
        ("sidecar_args", []),
        ("api_key", "secret"),
        ("admin_api_key", "secret"),
        ("host", "wrong-host"),
    ],
)
def test_conflicting_options_fail_before_launch(name, value):
    config = _config()
    with pytest.raises(ValueError):
        configure_engine_args(config, binding=config.engines[0], overrides={name: value})


@pytest.mark.parametrize(
    "field",
    [
        "model_path",
        "port",
        "grpc_port",
        "tp_size",
        "incremental_streaming_output",
        "tokenizer_worker_num",
        "smg_grpc_mode",
        "host",
    ],
)
def test_external_readback_must_report_contract_fields(field):
    config = _config()
    args = configure_engine_args(config, binding=config.engines[0], overrides={})
    del args[field]
    with pytest.raises(ValueError):
        validate_engine_info(config, binding=config.engines[0], server_info=args)


@pytest.mark.parametrize(
    "port_name", ["nccl_port", "metrics_port", "metrics_http_port", "engine_info_bootstrap_port", "gated_launch_port"]
)
@pytest.mark.parametrize("port", [30000, 30001, 8081])
def test_auxiliary_port_collision(port_name, port):
    config = _config()
    with pytest.raises(ValueError, match="overlaps"):
        configure_engine_args(config, binding=config.engines[0], overrides={port_name: port})


def test_binding_must_belong_to_run():
    config = _config()
    foreign = config.engines[0].model_copy(update={"name": "another-run"})
    with pytest.raises(ValueError, match="not part"):
        configure_engine_args(config, binding=foreign, overrides={})


def test_ipv6_binding():
    data = config_dict()
    data["engines"] = data["engines"][:1]
    for key in ("http", "grpc", "sidecar"):
        data["engines"][0][key]["host"] = "::1"
    config = DynamoConfig.model_validate(data)
    args = configure_engine_args(config, binding=config.engines[0], overrides={"host": "::"})
    validate_engine_info(config, binding=config.engines[0], server_info=args)
