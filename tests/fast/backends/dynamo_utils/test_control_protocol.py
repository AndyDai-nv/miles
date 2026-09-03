import pytest

from miles.backends.dynamo_utils.control_protocol import (
    CALL_TOKENIZER_MANAGER_ROUTE,
    DIRECT_ENGINE_ROUTES,
    SUPPORTED_CONTROL_OPERATIONS,
    DynamoControlRequest,
    build_dynamo_control_request,
    tokenizer_manager_request,
    typed_io_struct,
)


@pytest.mark.parametrize("operation", sorted(DIRECT_ENGINE_ROUTES))
def test_first_class_engine_routes_are_not_wrapped(operation):
    payload = {"marker": operation}

    request = build_dynamo_control_request(operation, payload)

    assert request == DynamoControlRequest(route=operation, body=payload)
    assert request.path == f"/engine/{operation}"
    assert request.body is not payload


@pytest.mark.parametrize(
    ("operation", "method", "io_struct_name", "payload"),
    [
        ("begin_weight_update", "begin_weight_update", "BeginWeightUpdateReqInput", {"selector": "target"}),
        (
            "check_weights",
            "check_weights",
            "CheckWeightsReqInput",
            {
                "action": "checksum",
                "allow_quant_error": True,
                "selector": "all",
                "skip_tensor_list": ["lm_head"],
            },
        ),
        (
            "continue_generation",
            "continue_generation",
            "ContinueGenerationReqInput",
            {"torch_empty_cache": False},
        ),
        (
            "destroy_weights_update_group",
            "destroy_weights_update_group",
            "DestroyWeightsUpdateGroupReqInput",
            {"group_name": "rollout"},
        ),
        ("end_weight_update", "end_weight_update", "EndWeightUpdateReqInput", {}),
        (
            "init_weights_update_group",
            "init_weights_update_group",
            "InitWeightsUpdateGroupReqInput",
            {
                "master_address": "10.0.0.1",
                "master_port": 23456,
                "rank_offset": 2,
                "world_size": 4,
                "group_name": "rollout",
                "backend": "nccl",
            },
        ),
        ("pause_generation", "pause_generation", "PauseGenerationReqInput", {"mode": "retract"}),
        (
            "pull_weights",
            "pull_weights",
            "PullWeightsReqInput",
            {
                "local_checkpoint_dir": "/models/local",
                "source_dir": "/models/published",
                "target_version": 7,
            },
        ),
        (
            "start_profile",
            "start_profile",
            "ProfileReq",
            {
                "output_dir": "/tmp/profiles",
                "start_step": 2,
                "num_steps": 5,
                "activities": ["CPU", "GPU"],
                "profile_by_stage": True,
                "with_stack": False,
                "record_shapes": True,
            },
        ),
    ],
)
def test_typed_tokenizer_manager_operations(operation, method, io_struct_name, payload):
    request = build_dynamo_control_request(operation, payload)

    assert request.route == CALL_TOKENIZER_MANAGER_ROUTE
    assert request.path == "/engine/call_tokenizer_manager"
    assert request.body == {
        "method": method,
        "args": [{f"io_struct.{io_struct_name}": payload}],
        "kwargs": {},
    }
    assert request.body["args"][0][f"io_struct.{io_struct_name}"] is not payload


def test_flush_cache_uses_keyword_arguments_instead_of_io_struct():
    request = build_dynamo_control_request("flush_cache", {"timeout_s": 5.0})

    assert request.body == {
        "method": "flush_cache",
        "args": [],
        "kwargs": {"timeout_s": 5.0},
    }


def test_payload_defaults_to_empty_mapping():
    direct = build_dynamo_control_request("stop_profile")
    typed = build_dynamo_control_request("continue_generation")
    plain = build_dynamo_control_request("flush_cache")

    assert direct.body == {}
    assert typed.body["args"] == [{"io_struct.ContinueGenerationReqInput": {}}]
    assert plain.body["kwargs"] == {}


def test_low_level_tokenizer_manager_request_preserves_wire_shape():
    args = [typed_io_struct("PauseGenerationReqInput", {"mode": "abort"})]
    kwargs = {"request_id": "request-1"}

    request = tokenizer_manager_request("pause_generation", args=args, kwargs=kwargs)

    assert request == DynamoControlRequest(
        route="call_tokenizer_manager",
        body={"method": "pause_generation", "args": args, "kwargs": kwargs},
    )
    assert request.body["args"] is not args
    assert request.body["kwargs"] is not kwargs


@pytest.mark.parametrize("class_name", ["", "io_struct.CheckWeightsReqInput", "not-a-class"])
def test_typed_io_struct_rejects_invalid_class_names(class_name):
    with pytest.raises(ValueError, match="invalid SGLang io_struct class name"):
        typed_io_struct(class_name)


@pytest.mark.parametrize("method", ["", "tokenizer_manager.flush_cache", "not-a-method"])
def test_tokenizer_manager_request_rejects_invalid_method_names(method):
    with pytest.raises(ValueError, match="invalid tokenizer-manager method name"):
        tokenizer_manager_request(method)


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (lambda: typed_io_struct("CheckWeightsReqInput", []), "fields must be a mapping or None"),
        (lambda: tokenizer_manager_request("flush_cache", args="bad"), "args must be a non-string sequence"),
        (lambda: tokenizer_manager_request("flush_cache", kwargs=[]), "kwargs must be a mapping or None"),
        (lambda: build_dynamo_control_request("stop_profile", []), "payload must be a mapping or None"),
    ],
)
def test_invalid_container_types_fail_before_http(call, message):
    with pytest.raises(TypeError, match=message):
        call()


def test_unknown_operation_fails_closed_with_supported_operations():
    with pytest.raises(ValueError, match="unsupported Dynamo control operation 'get_server_info'") as exc_info:
        build_dynamo_control_request("get_server_info")

    for operation in SUPPORTED_CONTROL_OPERATIONS:
        assert operation in str(exc_info.value)
