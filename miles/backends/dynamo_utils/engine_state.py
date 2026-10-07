import asyncio
import json
from dataclasses import dataclass

import grpc

from miles.backends.dynamo_utils.config import DynamoConfig, EngineBinding
from miles.backends.dynamo_utils.engine_contract import validate_engine_info
from miles.backends.dynamo_utils.engine_state_pb2 import EngineState

_WATCH_METHOD = "/sglang.runtime.v1.SglangService/WatchEngineState"


@dataclass(frozen=True)
class EngineObservation:
    instance_id: int
    weight_version: str | None


async def observe_engine(config: DynamoConfig, *, binding: EngineBinding, timeout: float) -> EngineObservation:
    host = f"[{binding.grpc.host}]" if ":" in binding.grpc.host else binding.grpc.host
    async with grpc.aio.insecure_channel(f"{host}:{binding.grpc.port}") as channel:
        watch = channel.unary_stream(_WATCH_METHOD, response_deserializer=EngineState.FromString)(b"", timeout=timeout)
        try:
            state = await watch.read()
            if state is grpc.aio.EOF or not state.instance_id or not state.revision:
                raise ValueError("SGLang did not report an engine incarnation")
            if not state.healthy or state.is_pause:
                raise ValueError("SGLang engine is unhealthy or paused")
            server_info = json.loads(state.server_info.json_info)
            model_info = json.loads(state.model_info.json_info)
            validate_engine_info(config, binding=binding, server_info=server_info)
            if not isinstance(model_info, dict) or model_info.get("model_path") != config.model_path:
                raise ValueError("SGLang live model does not match the configured model")
            version = model_info.get("weight_version")
            if version is not None and (not isinstance(version, str) or not version.strip()):
                raise ValueError("invalid engine weight version")
            return EngineObservation(instance_id=state.instance_id, weight_version=version)
        finally:
            watch.cancel()


async def observe_fleet(config: DynamoConfig, *, timeout: float) -> tuple[EngineObservation, ...]:
    async with asyncio.TaskGroup() as group:
        tasks = [
            group.create_task(observe_engine(config, binding=binding, timeout=timeout)) for binding in config.engines
        ]
    return tuple(task.result() for task in tasks)
