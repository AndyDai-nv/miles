import asyncio
from argparse import Namespace
from typing import Any
from urllib.parse import urlsplit

from miles.backends.dynamo_utils.config import Address, DynamoConfig, EngineBinding
from miles.backends.dynamo_utils.engine_contract import validate_engine_info
from miles.backends.sglang_utils.sglang_api_client import SGLangApiClient
from miles.ray.rollout.external_engine_provider import StaticInferenceEngineWorkerProvider
from miles.utils.workers.worker_info import WorkerInfo
from miles.utils.workers.worker_provider.base import BaseWorkerProvider, CellInfo, CellReconcileFn, StopWatchFn
from miles.utils.workers.worker_spec import HostAndPort, NamedHostAndPorts


class DynamoExternalEngineProvider(BaseWorkerProvider):
    """Static bindings, not restart detection or proof of a sidecar's live gRPC peer."""

    def __init__(self, *, config: DynamoConfig, args: Any) -> None:
        if not args.rollout_external or args.colocate or args.rollout_external_router_pd:
            raise ValueError("Dynamo bindings require external, non-colocated aggregated engines")
        urls = [binding.http.url for binding in config.engines]
        declared = args.rollout_external_engine_addrs
        if declared and [_address(url) for url in declared] != [binding.http for binding in config.engines]:
            raise ValueError("external engine addresses conflict with the ordered Dynamo bindings")
        self._config = config
        self._api_key = args.sglang_api_key
        # Manifest owns the ordered address book; never mutate the caller's args.
        self._provider = StaticInferenceEngineWorkerProvider(
            args=Namespace(**{**vars(args), "rollout_external_engine_addrs": urls})
        )
        self._bindings: dict[str, EngineBinding] | None = None

    async def init(self) -> None:
        self._bindings = None
        await self._provider.init()
        await asyncio.gather(*[self._validate_engine(binding) for binding in self._config.engines])
        by_address = {binding.http: binding for binding in self._config.engines}
        bindings = {}
        for cell in self._provider.cell_infos:
            if len(cell.worker_names) != 1 or cell.meta["worker_type"] != "regular":
                raise ValueError("expected one aggregated external worker per binding")
            worker = cell.worker_names[0]
            primary = (await self._provider.get_addrs(worker))["primary"]
            address = Address(host=primary.host.strip("[]"), port=primary.port)
            binding = by_address.pop(address, None)
            if binding is None or cell.meta["num_gpus_per_engine"] != binding.tensor_parallel_size:
                raise ValueError("discovered external cell does not match a unique TP binding")
            bindings[worker] = binding
        if by_address:
            raise ValueError("external provider did not discover every binding")
        self._bindings = bindings

    async def _validate_engine(self, binding: EngineBinding) -> None:
        info = await SGLangApiClient(server_url=binding.http.url, api_key=self._api_key).get_server_info()
        validate_engine_info(self._config, binding=binding, server_info=info)
        if info.get("disaggregation_mode") not in (None, "null"):
            raise ValueError("Dynamo static bindings currently require aggregated engines")

    def _binding(self, worker: str) -> EngineBinding:
        if self._bindings is None:
            raise RuntimeError("Dynamo external provider must successfully initialize first")
        return self._bindings[worker]

    def _addrs(self, worker: str, primary_addrs: NamedHostAndPorts) -> NamedHostAndPorts:
        binding = self._binding(worker)
        return {
            **primary_addrs,
            "sglang_grpc": _host_and_port(binding.grpc),
            "dynamo_sidecar": _host_and_port(binding.sidecar),
        }

    @property
    def cell_infos(self) -> list[CellInfo]:
        if self._bindings is None:
            raise RuntimeError("Dynamo external provider must successfully initialize first")
        return self._provider.cell_infos

    async def get_addrs(self, worker_name: str) -> NamedHostAndPorts:
        self._binding(worker_name)
        return self._addrs(worker_name, await self._provider.get_addrs(worker_name))

    def get_worker_infos(self, *, cell_ids: list[str]) -> list[list[WorkerInfo]]:
        _ = self.cell_infos
        return [
            [info.model_copy(update={"self_addrs": self._addrs(info.name, info.self_addrs)}) for info in group]
            for group in self._provider.get_worker_infos(cell_ids=cell_ids)
        ]

    async def watch_cells(self, reconcile: CellReconcileFn) -> StopWatchFn:
        _ = self.cell_infos
        return await self._provider.watch_cells(reconcile)

    def expected_num_cells(self, *, group_id: str) -> int:
        _ = self.cell_infos
        return self._provider.expected_num_cells(group_id=group_id)


def _host_and_port(address: Address) -> HostAndPort:
    host = f"[{address.host}]" if ":" in address.host else address.host
    return HostAndPort(host=host, port=address.port)


def _address(raw: str) -> Address:
    url = urlsplit(raw if "://" in raw else f"http://{raw}")
    if (
        url.scheme != "http"
        or not url.hostname
        or url.port is None
        or url.username is not None
        or url.password is not None
        or url.path not in ("", "/")
        or url.query
        or url.fragment
    ):
        raise ValueError("external engine address must be an HTTP host and port")
    return Address(host=url.hostname, port=url.port)
