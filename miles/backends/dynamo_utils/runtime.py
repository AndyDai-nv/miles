import asyncio

from miles.backends.dynamo_utils.config import DynamoConfig
from miles.backends.dynamo_utils.engine_state import observe_fleet
from miles.backends.dynamo_utils.serving_registry import SidecarDiscoveryRegistry
from miles.backends.dynamo_utils.sync_gate import SyncRolloutGate
from miles.backends.sglang_utils.sglang_api_client import SGLangApiClient


class DynamoRuntime:
    def __init__(self, config: DynamoConfig):
        self._clients = tuple(SGLangApiClient(binding.http.url) for binding in config.engines)
        self.gate = SyncRolloutGate(
            observe=lambda: observe_fleet(config, timeout=config.control_timeout_seconds),
            drain=self._drain,
            timeout=config.control_timeout_seconds,
        )
        self.registry = SidecarDiscoveryRegistry(config=config, on_remove=self.gate.fail)

    async def _drain(self) -> None:
        async with asyncio.TaskGroup() as group:
            for client in self._clients:
                group.create_task(client.flush_cache())
