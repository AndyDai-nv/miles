import copy
from argparse import Namespace

import pytest
from tests.fast.backends.dynamo_utils.test_config import config_dict

from miles.backends.dynamo_utils.config import DynamoConfig
from miles.backends.dynamo_utils.external_provider import DynamoExternalEngineProvider
from miles.backends.sglang_utils.sglang_api_client import SGLangApiClient

pytestmark = pytest.mark.asyncio


def _args(**overrides):
    return Namespace(
        **{
            "rollout_external": True,
            "colocate": False,
            "rollout_external_router_pd": False,
            "rollout_external_engine_addrs": [],
            "rollout_num_gpus": 4,
            "rollout_num_gpus_per_engine": 2,
            "sglang_api_key": None,
            **overrides,
        }
    )


def _install(monkeypatch, config, *, overrides=None):
    seen = []

    async def info(self):
        seen.append(self.server_url)
        binding = next(binding for binding in config.engines if binding.http.url == self.server_url)
        return {
            "host": "0.0.0.0",
            "port": binding.http.port,
            "grpc_port": binding.grpc.port,
            "model_path": config.model_path,
            "tp_size": binding.tensor_parallel_size,
            "incremental_streaming_output": True,
            "disaggregation_mode": "null",
            "internal_states": [{"world_size": binding.tensor_parallel_size}],
            **(overrides or {}),
        }

    monkeypatch.setattr(SGLangApiClient, "get_server_info", info)
    return seen


async def test_two_tp2_cells_preserve_native_primary_and_metadata(monkeypatch):
    config = DynamoConfig.model_validate(config_dict())
    args = _args()
    before = copy.deepcopy(vars(args))
    seen = _install(monkeypatch, config)
    provider = DynamoExternalEngineProvider(config=config, args=args)
    await provider.init()
    assert vars(args) == before
    assert provider.expected_num_cells(group_id="default") == 2
    assert provider.expected_num_cells(group_id="other") == 0
    cells = provider.cell_infos
    assert [cell.meta["gpu_offset"] for cell in cells] == [0, 2]
    assert all(cell.meta["num_gpus_per_engine"] == 2 and cell.meta["update_weights"] for cell in cells)
    for cell, binding in zip(cells, config.engines, strict=True):
        addrs = await provider.get_addrs(cell.worker_names[0])
        assert addrs["primary"].addr == binding.http.url
        assert addrs["sglang_grpc"].addr == binding.grpc.url
        assert addrs["dynamo_sidecar"].addr == binding.sidecar.url
        info = provider.get_worker_infos(cell_ids=[cell.cell_id])[0][0]
        assert info.self_addrs == addrs
        assert cell.workers_hash == binding.http.url  # Static provider semantics, not an incarnation fence.
    observed = []

    async def reconcile(cell_id, info):
        observed.append(info)

    stop = await provider.watch_cells(reconcile)
    assert observed == cells
    await stop()
    assert set(seen) == {binding.http.url for binding in config.engines}


@pytest.mark.parametrize(
    "override",
    [
        {"grpc_port": 9999},
        {"tp_size": 1},
        {"model_path": "/other"},
        {"incremental_streaming_output": False},
        {"internal_states": [{"world_size": 1}]},
    ],
)
async def test_live_contract_mismatch_is_not_published(monkeypatch, override):
    config = DynamoConfig.model_validate(config_dict())
    _install(monkeypatch, config, overrides=override)
    provider = DynamoExternalEngineProvider(config=config, args=_args())
    with pytest.raises((ValueError, AssertionError)):
        await provider.init()
    with pytest.raises(RuntimeError):
        _ = provider.cell_infos


@pytest.mark.parametrize(
    "overrides",
    [
        {"rollout_external": False},
        {"colocate": True},
        {"rollout_external_router_pd": True},
        {"rollout_external_engine_addrs": ["http://frontend:8000"]},
    ],
)
async def test_invalid_topology_or_address_book(overrides):
    with pytest.raises(ValueError):
        DynamoExternalEngineProvider(config=DynamoConfig.model_validate(config_dict()), args=_args(**overrides))


async def test_ipv6_and_equivalent_declared_addresses(monkeypatch):
    data = config_dict()
    data["engines"] = data["engines"][:1]
    for address in ("http", "grpc", "sidecar"):
        data["engines"][0][address]["host"] = "::1"
    config = DynamoConfig.model_validate(data)
    _install(monkeypatch, config, overrides={"host": "::"})
    provider = DynamoExternalEngineProvider(
        config=config,
        args=_args(rollout_external_engine_addrs=["[::1]:30000"], rollout_num_gpus=2),
    )
    await provider.init()
    addrs = await provider.get_addrs(provider.cell_infos[0].worker_names[0])
    assert addrs["primary"].addr == "http://[::1]:30000"
    assert addrs["sglang_grpc"].addr == "http://[::1]:30001"


async def test_access_before_init():
    provider = DynamoExternalEngineProvider(config=DynamoConfig.model_validate(config_dict()), args=_args())
    with pytest.raises(RuntimeError):
        await provider.get_addrs("unknown")
    with pytest.raises(RuntimeError):
        provider.get_worker_infos(cell_ids=[])
