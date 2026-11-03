from unittest.mock import AsyncMock, MagicMock

import pytest
from tests.fast.ray.rollout.conftest import make_args, track_server_cell

from miles.backends.dynamo_utils.config import DynamoConfig
from miles.backends.dynamo_utils.serving_registry import DynamoServingRegistry, DynamoServingRegistryFactory
from miles.backends.sglang_utils.sglang_api_client import WorkerType
from miles.ray.rollout.cell_state import CellAddrInfo, StateDisposed, StatePendingWeights, StateServing
from miles.ray.rollout.rollout_server import create_rollout_servers
from miles.ray.rollout.server_cell import ServerCell, ServerCellMetadata
from miles.ray.rollout.serving_registry import SGLangServingRegistry
from miles.utils.context_lock import ContextLock
from miles.utils.workers.worker_spec import HostAndPort

pytestmark = [pytest.mark.asyncio, pytest.mark.usefixtures("dispose_tracked_server_cells")]


class _RecordingRegistry:
    def __init__(self):
        self.calls: list[tuple[str, dict]] = []

    async def register(self, *, worker_url: str, worker_type: WorkerType, bootstrap_port: int | None) -> None:
        self.calls.append(
            ("register", dict(worker_url=worker_url, worker_type=worker_type, bootstrap_port=bootstrap_port))
        )

    async def unregister(self, *, worker_url: str) -> None:
        self.calls.append(("unregister", dict(worker_url=worker_url)))


async def test_injected_registry_owns_registration_and_cleanup():
    registry = _RecordingRegistry()
    cell = track_server_cell(
        ServerCell(
            args=make_args(),
            meta=ServerCellMetadata(
                model_id="default",
                worker_type=WorkerType.PREFILL,
                cell_id="cell-0",
                num_gpus_per_engine=1,
                gpu_offset=0,
                sglang_api_key=None,
                worker_name="cell-0-0",
                needs_offload=False,
                update_weights=True,
                workers_hash="test-hash",
            ),
            provider=AsyncMock(),
            serving_registry=registry,
        )
    )
    cell._state = StatePendingWeights(
        addr_info=CellAddrInfo(server_url="http://engine:30000", bootstrap_port=1234, gate_url=None)
    )

    await cell.mark_weights_ready()

    assert isinstance(cell._state, StateServing)
    assert registry.calls == [
        ("register", dict(worker_url="http://engine:30000", worker_type=WorkerType.PREFILL, bootstrap_port=1234))
    ]

    await cell.dispose()

    assert isinstance(cell._state, StateDisposed)
    assert registry.calls == [
        ("register", dict(worker_url="http://engine:30000", worker_type=WorkerType.PREFILL, bootstrap_port=1234)),
        ("unregister", dict(worker_url="http://engine:30000")),
    ]


@pytest.mark.parametrize("router", ["sglang", "dynamo"])
async def test_registry_is_injected_at_creation_and_isolates_engine_incarnations(router):
    config = (
        DynamoConfig(
            namespace="test-run",
            discovery={"backend": "etcd", "endpoints": [{"host": "localhost", "port": 2379}]},
            pins={"dynamo": "a" * 40, "sglang": "b" * 40},
        )
        if router == "dynamo"
        else None
    )
    args = make_args(router=router, dynamo_config=config, colocate=True, ft_components=[])
    model_name = args.sglang.models[0].name
    provider = MagicMock()
    provider.expected_num_cells.return_value = 2
    lock = ContextLock(name="test-registry")
    servers = await create_rollout_servers(
        args,
        lock,
        engine_provider=provider,
        router_addrs={model_name: HostAndPort(host="localhost", port=30000)},
    )
    server = servers[model_name]
    assert isinstance(server.serving_registry, DynamoServingRegistryFactory if config else SGLangServingRegistry)
    meta = ServerCellMetadata(
        model_id=model_name,
        worker_type=WorkerType.REGULAR,
        cell_id="cell-0",
        num_gpus_per_engine=1,
        gpu_offset=0,
        sglang_api_key="engine-key",
        worker_name="cell-0-0",
        needs_offload=True,
        update_weights=True,
        workers_hash="first-incarnation",
    )
    async with lock:
        try:
            await server.add_cell(meta)
            await server.add_cell(meta.model_copy(update=dict(cell_id="cell-1", worker_name="cell-1-0", gpu_offset=1)))
            first = server.all_server_cells["cell-0"].serving_registry
            second = server.all_server_cells["cell-1"].serving_registry
            if config:
                assert isinstance(first, DynamoServingRegistry)
                assert first.provider is provider
                assert first.worker_name == "cell-0-0"
                assert first.api_key == "engine-key"
                first._session = "old-session"
                first._expected_version = "old-version"
                assert second is not first
                assert second._session is None
                assert second._expected_version is None
            else:
                assert first is second is server.serving_registry
            await server.remove_cell("cell-0")
            await server.add_cell(meta.model_copy(update=dict(workers_hash="replacement-incarnation")))
            replacement = server.all_server_cells["cell-0"].serving_registry
            if config:
                assert replacement is not first
                assert replacement._session is None
                assert replacement._expected_version is None
            else:
                assert replacement is first
        finally:
            await server.dispose()
