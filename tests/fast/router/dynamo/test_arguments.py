import pytest
from tests.fast.fixtures.args_fixtures import parse_fsdp_test_config
from tests.fast.ray.rollout.conftest import make_args

from miles.router.dynamo.arguments import validate_dynamo_args
from miles.router.dynamo.config import DynamoConfig
from miles.utils.args.runtime import InferenceControllerConfig


@pytest.fixture
def configured_args(tmp_path, monkeypatch):
    config = DynamoConfig(
        namespace="test-run",
        discovery={"backend": "etcd", "endpoints": [{"host": "localhost", "port": 2379}]},
        pins={"dynamo": "a" * 40, "sglang": "b" * 40},
    )
    path = tmp_path / "dynamo.json"
    path.write_text(config.model_dump_json())
    monkeypatch.setattr("miles.router.dynamo.arguments.use_legacy_rollout_v1", lambda: False)
    return make_args(router="dynamo", dynamo_config=str(path))


def test_fault_tolerance_and_api_key_are_accepted(configured_args):
    configured_args.ft_components = ["rollout"]
    configured_args.sglang_api_key = "test-key"
    validate_dynamo_args(configured_args, sglang=configured_args.sglang)
    assert configured_args.dynamo_config.namespace == "test-run"
    assert configured_args.sglang_api_key == "test-key"


def test_dynamo_configuration_survives_controller_config_transport(configured_args):
    parsed = parse_fsdp_test_config(
        "--router", "dynamo", "--dynamo-config", configured_args.dynamo_config, "--rollout-num-gpus", "1"
    )
    config = InferenceControllerConfig.slice_from(parsed)

    restored = InferenceControllerConfig.model_validate_json(config.model_dump_json())

    assert restored.dynamo_config.namespace == "test-run"
    assert restored.dynamo_config == parsed.dynamo_config
    assert restored.router == "dynamo"


def test_dynamo_router_requires_configuration(configured_args):
    configured_args.dynamo_config = None
    with pytest.raises(ValueError, match="requires --dynamo-config"):
        validate_dynamo_args(configured_args, sglang=configured_args.sglang)


@pytest.mark.parametrize(
    "name,value",
    [
        ("rollout_external", True),
        ("router", "miles"),
        ("router", "sglang"),
        ("eval_uses_snapshots", True),
        ("sglang_grpc_port", 1234),
        ("sglang_sidecar_args", ["--namespace", "wrong"]),
        ("sglang_kv_events_config", "{}"),
    ],
)
def test_incompatible_options_fail_at_startup(configured_args, name, value):
    setattr(configured_args, name, value)
    with pytest.raises(ValueError):
        validate_dynamo_args(configured_args, sglang=configured_args.sglang)


@pytest.mark.parametrize("policy", ["manual", "consistent_hashing", "cache_aware", "round_robin", "random", ""])
def test_sglang_router_policies_point_to_dynamo_configuration(configured_args, policy):
    configured_args.sglang_router_policy = policy
    with pytest.raises(ValueError, match="frontend.router_mode"):
        validate_dynamo_args(configured_args, sglang=configured_args.sglang)


@pytest.mark.parametrize(
    "base_dp,override_dp,accepted", [(1, None, True), (2, None, False), (1, 2, False), (2, 1, True)]
)
def test_dp_validation_uses_resolved_per_engine_value(configured_args, base_dp, override_dp, accepted):
    args = make_args(router="dynamo", dynamo_config=configured_args.dynamo_config, sglang_dp_size=base_dp)
    model = args.sglang.models[0]
    group = model.server_groups[0]
    overrides = group.overrides | ({"dp_size": override_dp} if override_dp is not None else {})
    model = model.model_copy(update={"server_groups": [group.model_copy(update={"overrides": overrides})]})
    sglang = args.sglang.model_copy(update={"models": [model]})
    if accepted:
        validate_dynamo_args(args, sglang=sglang)
        assert isinstance(args.dynamo_config, DynamoConfig)
    else:
        with pytest.raises(ValueError, match="per-engine dp_size=1"):
            validate_dynamo_args(args, sglang=sglang)
