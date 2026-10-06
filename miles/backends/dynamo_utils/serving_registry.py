from collections.abc import Callable

from miles.backends.dynamo_utils.config import DynamoConfig


class SidecarDiscoveryRegistry:
    """Upstream sidecars own discovery; Miles owns the exclusive client's rollout gate."""

    def __init__(self, *, config: DynamoConfig, on_remove: Callable[[], None]):
        self._worker_urls = frozenset(binding.http.url for binding in config.engines)
        self._on_remove = on_remove

    async def register(self, *, worker_url: str, worker_type: str, bootstrap_port: int | None) -> None:
        if worker_url not in self._worker_urls or worker_type != "regular" or bootstrap_port is not None:
            raise ValueError("Dynamo registry requires a configured aggregated engine")
        # No SGLang-router add_worker call: registration is already sidecar-owned.
        # This callback is not an admission grant; only the versioned gate permits dispatch.

    async def unregister(self, *, worker_url: str) -> None:
        self._on_remove()
        if worker_url not in self._worker_urls:
            raise ValueError("unknown Dynamo engine")
        # External processes remain externally owned. Do not claim discovery withdrawal.
