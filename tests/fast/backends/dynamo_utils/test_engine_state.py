import json
import os
import shutil
import subprocess
from pathlib import Path

import grpc
import pytest
from google.protobuf import descriptor_pb2, descriptor_pool, message_factory
from tests.fast.backends.dynamo_utils.test_config import config_dict

from miles.backends.dynamo_utils.config import DynamoConfig
from miles.backends.dynamo_utils.engine_contract import configure_engine_args
from miles.backends.dynamo_utils.engine_state import observe_engine
from miles.backends.dynamo_utils.engine_state_pb2 import EngineState


@pytest.mark.asyncio
@pytest.mark.parametrize("problem", [None, "empty", "unhealthy", "paused", "model", "contract"])
async def test_real_grpc_initial_snapshot_and_failures(problem):
    server = grpc.aio.server()
    port = server.add_insecure_port("127.0.0.1:0")
    data = config_dict()
    data["engines"] = data["engines"][:1]
    data["engines"][0]["grpc"] = {"host": "127.0.0.1", "port": port}
    data["engines"][0]["http"]["host"] = "127.0.0.1"
    config = DynamoConfig.model_validate(data)
    binding = config.engines[0]
    info = configure_engine_args(config, binding=binding, overrides={})
    if problem == "contract":
        info["incremental_streaming_output"] = False
    state = EngineState(instance_id=123, revision=1, healthy=problem != "unhealthy", is_pause=problem == "paused")
    state.server_info.json_info = json.dumps(info)
    state.model_info.json_info = json.dumps(
        {"model_path": "/other" if problem == "model" else config.model_path, "weight_version": "17"}
    )

    async def watch(request, context):
        if problem != "empty":
            yield state.SerializeToString()

    server.add_generic_rpc_handlers(
        (
            grpc.method_handlers_generic_handler(
                "sglang.runtime.v1.SglangService", {"WatchEngineState": grpc.unary_stream_rpc_method_handler(watch)}
            ),
        )
    )
    await server.start()
    try:
        if problem is None:
            result = await observe_engine(config, binding=binding, timeout=1)
            assert (result.instance_id, result.weight_version) == (123, "17")
        else:
            with pytest.raises(ValueError):
                await observe_engine(config, binding=binding, timeout=1)
    finally:
        await server.stop(None)


def test_wire_projection_against_real_sglang_proto(tmp_path):
    root = os.environ.get("MILES_TEST_SGLANG_SOURCE")
    if root is None or shutil.which("protoc") is None:
        pytest.skip("set MILES_TEST_SGLANG_SOURCE and install protoc for upstream wire verification")
    descriptor = tmp_path / "sglang.desc"
    subprocess.run(
        [
            "protoc",
            f"--proto_path={Path(root) / 'proto'}",
            f"--descriptor_set_out={descriptor}",
            "--include_imports",
            "sglang/runtime/v1/sglang.proto",
        ],
        check=True,
    )
    pool = descriptor_pool.DescriptorPool()
    for file in descriptor_pb2.FileDescriptorSet.FromString(descriptor.read_bytes()).file:
        pool.Add(file)
    native = message_factory.GetMessageClass(pool.FindMessageTypeByName("sglang.runtime.v1.EngineStateSnapshot"))
    state = native(instance_id=123, revision=7, healthy=True, is_pause=False)
    state.model_info.json_info = '{"weight_version":"17"}'
    state.server_info.json_info = '{"grpc_port":30001}'
    parsed = EngineState.FromString(state.SerializeToString())
    assert (parsed.instance_id, parsed.revision, parsed.healthy, parsed.is_pause) == (123, 7, True, False)
    assert json.loads(parsed.model_info.json_info)["weight_version"] == "17"
    assert json.loads(parsed.server_info.json_info)["grpc_port"] == 30001
