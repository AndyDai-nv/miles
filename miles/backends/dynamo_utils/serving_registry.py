from collections.abc import Callable

from miles.backends.dynamo_utils.config import DynamoConfig


class SidecarDiscoveryRegistry:
    """Sidecars own discovery; these callbacks only validate bindings and close Miles dispatch."""

    def __init__(self, *, config: DynamoConfig, on_remove: Callable[[], None]):
        self._worker_urls = frozenset(binding.http.url for binding in config.engines)
        self._on_remove = on_remove

    async def register(self, *, worker_url: str, worker_type: str, bootstrap_port: int | None) -> None:
        # Aggregated SGLang also reports its default bootstrap port; it is unused here.
        if worker_url not in self._worker_urls or worker_type != "regular":
            raise ValueError("Dynamo registry requires a configured aggregated engine")

    async def unregister(self, *, worker_url: str) -> None:
        self._on_remove()
        if worker_url not in self._worker_urls:
            raise ValueError("unknown Dynamo engine")
