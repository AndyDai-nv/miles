import shlex
from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from tests.fast.backends.dynamo_utils.test_config import config_dict

from miles.backends.dynamo_utils.config import DynamoConfig
from miles.backends.dynamo_utils.launch import external_engine_provider, frontend_spec
from miles.backends.dynamo_utils.serving_registry import SidecarDiscoveryRegistry
from miles.utils.workers.worker_spec import HostAndPort


@pytest.mark.parametrize("configured", [False, True])
def test_frontend_runs_as_cpu_command_worker(configured, monkeypatch):
    data = config_dict()
    data["frontend"] = {"router_mode": "kv", "extra_args": ["--metrics-prefix", "rl"]}
    if configured:
        data["frontend"]["address"] = {"host": "0.0.0.0", "port": 8123}
    config = DynamoConfig.model_validate(data)
    args = Namespace(pin_rollout_manager_to_head=True)
    spec = frontend_spec(args, config=config, pool_name="inference-router-0")
    ctx = SimpleNamespace(self_addrs={"primary": HostAndPort(host="10.0.0.2", port=8123)})
    argv = shlex.split(spec.launch_command(ctx))
    assert "dynamo.frontend" in argv and "sglang_router.launch_router" not in argv
    assert argv[argv.index("--http-port") + 1] == "8123"
    assert argv[argv.index("--http-host") + 1] == "0.0.0.0"
    assert argv[argv.index("--router-mode") + 1] == "kv"
    assert spec.scheduling.num_gpus_per_worker == 0
    assert spec.env_var(ctx)["DYN_SGLANG_ENABLE_GENERATE"] == "1"
    assert spec.port_infos[0].static_port == (8123 if configured else 8000)
    assert spec.port_infos[0].allow_dynamic is not configured


def test_provider_factory_uses_resolved_config():
    config = DynamoConfig.model_validate(config_dict())
    args = Namespace(
        dynamo_config=config,
        rollout_external=True,
        colocate=False,
        rollout_external_router_pd=False,
        rollout_external_engine_addrs=None,
        sglang_api_key=None,
    )
    provider = external_engine_provider(args, capability=None)
    assert provider._config is config


@pytest.mark.asyncio
async def test_sidecar_owns_discovery_but_dispose_closes_dispatch():
    config = DynamoConfig.model_validate(config_dict())
    close = Mock()
    registry = SidecarDiscoveryRegistry(config=config, on_remove=close)
    await registry.register(worker_url=config.engines[0].http.url, worker_type="regular", bootstrap_port=None)
    close.assert_not_called()
    await registry.unregister(worker_url=config.engines[0].http.url)
    close.assert_called_once_with()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "worker_url,worker_type,bootstrap",
    [("http://unknown:30000", "regular", None), ("http://engine-0:30000", "prefill", 9999)],
)
async def test_registry_rejects_unbound_or_pd_workers(worker_url, worker_type, bootstrap):
    registry = SidecarDiscoveryRegistry(config=DynamoConfig.model_validate(config_dict()), on_remove=Mock())
    with pytest.raises(ValueError):
        await registry.register(worker_url=worker_url, worker_type=worker_type, bootstrap_port=bootstrap)
