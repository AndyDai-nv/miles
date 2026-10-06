from argparse import ArgumentParser
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from tests.fast.backends.dynamo_utils.test_config import config_dict
from tests.fast.ray.rollout.conftest import make_args
from tests.fast.ray.rollout.test_inference_controller import _FakeUpdatableCell, _make_controller, _RecordingServer

from miles.backends.dynamo_utils.arguments import PROVIDER_PATH, ROLLOUT_PATH, configure_dynamo
from miles.backends.dynamo_utils.config import DynamoConfig
from miles.backends.dynamo_utils.engine_state import EngineObservation
from miles.backends.dynamo_utils.sync_gate import SyncRolloutGate
from miles.ray.specs.inference import specs_router
from miles.utils.arguments import get_miles_extra_args_provider, miles_validate_args


def _args():
    data = config_dict()
    data["exclusive_run"] = True
    return make_args(
        dynamo_config=DynamoConfig.model_validate(data),
        hf_checkpoint="/models/policy",
        rollout_num_gpus=4,
        rollout_num_gpus_per_engine=2,
        rollout_function_path=None,
        eval_function_path=None,
        custom_inference_engine_provider_path=None,
        rollout_submission_granularity=None,
        eval_interval=None,
    )


def test_config_selects_native_paths_without_launching_gpu_workers(tmp_path):
    args = _args()
    path = tmp_path / "dynamo.json"
    path.write_text(args.dynamo_config.model_dump_json())
    args.dynamo_config = str(path)
    configure_dynamo(args)
    assert args.rollout_function_path == ROLLOUT_PATH
    assert args.custom_inference_engine_provider_path == PROVIDER_PATH
    spec = specs_router(args)[0]
    assert spec.name == "inference-router-0"
    assert spec.scheduling.num_gpus_per_worker == 0


def test_real_argument_validation_derives_external_rollout(tmp_path):
    config = _args().dynamo_config
    path = tmp_path / "dynamo.json"
    path.write_text(config.model_dump_json())
    parser = ArgumentParser()
    get_miles_extra_args_provider()(parser)
    args = parser.parse_args(
        [
            "--dynamo-config",
            str(path),
            "--hf-checkpoint",
            config.model_path,
            "--rollout-batch-size",
            "4",
            "--n-samples-per-prompt",
            "4",
            "--rollout-num-gpus-per-engine",
            "2",
            "--num-rollout",
            "2",
        ]
    )
    miles_validate_args(args)
    assert args.rollout_external is True
    assert args.rollout_num_gpus == 4
    assert args.custom_inference_engine_provider_path == PROVIDER_PATH
    assert args.rollout_function_path == args.eval_function_path == ROLLOUT_PATH


@pytest.mark.parametrize(
    "name,value",
    [
        ("fully_async", True),
        ("ft_components", ["rollout"]),
        ("megatron_config", "multi-policy.yaml"),
        ("sglang_api_key", "unsupported-key"),
        ("partial_rollout", True),
        ("eval_interval", 1),
        ("colocate", True),
        ("use_sampling_support_replay", True),
        ("custom_generate_function_path", "other.generate"),
        ("dynamic_sampling_filter_path", "other.filter"),
        ("rollout_num_gpus", 8),
        ("rollout_num_gpus_per_engine", 4),
        ("hf_checkpoint", "wrong-model"),
        ("rollout_function_path", "other.rollout"),
        ("update_weight_transfer_mode", "disk-delta"),
    ],
)
def test_incompatible_paths_fail_at_argument_resolution(name, value):
    args = _args()
    setattr(args, name, value)
    with pytest.raises(ValueError):
        configure_dynamo(args)


def test_explicit_exclusive_ownership_is_required():
    args = _args()
    args.dynamo_config = args.dynamo_config.model_copy(update={"exclusive_run": False})
    with pytest.raises(ValueError, match="exclusive_run"):
        configure_dynamo(args)


def _controller():
    cell = _FakeUpdatableCell("static-address")
    server = _RecordingServer({"cell": cell}, update_weights=True)
    server.api_clients, server.engine_gpu_counts, server.engine_gpu_offsets = ["sglang-http"], [2], [0]
    controller = _make_controller({"default": server})
    observations = [EngineObservation(123, None)]
    gate = SyncRolloutGate(observe=AsyncMock(side_effect=lambda: tuple(observations)), drain=AsyncMock(), timeout=1)
    controller._dynamo_runtime = SimpleNamespace(gate=gate)
    return controller, observations, cell


@pytest.mark.asyncio
async def test_controller_checks_versions_on_initial_and_already_serving_updates():
    controller, observations, cell = _controller()
    for version in ("1", "2"):
        info = await controller.start_update_weights()
        assert info.rollout_engines == ["sglang-http"]
        observations[0] = EngineObservation(123, version)
        await controller.end_update_weights(info.snapshot_cell_id_to_hashes, weight_version=version)
        cell.is_pending_weights = False
        ticket = await controller.dynamo_acquire_rollout(version=version)
        await controller.dynamo_release_rollout(ticket=ticket, success=True)
    assert cell.marked_ready == 1
    assert not controller.context_lock.locked


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["stale_snapshot", "wrong_version", "missing_version", "restart", "trainer_abort"])
async def test_controller_errors_close_gate_and_release_context_lock(failure):
    controller, observations, _ = _controller()
    info = await controller.start_update_weights()
    observations[0] = EngineObservation(124 if failure == "restart" else 123, "1")
    if failure == "trainer_abort":
        await controller.abort_update_weights()
    else:
        version = None if failure == "missing_version" else "2" if failure == "wrong_version" else "1"
        snapshot = {} if failure == "stale_snapshot" else info.snapshot_cell_id_to_hashes
        with pytest.raises((ValueError, RuntimeError)):
            await controller.end_update_weights(snapshot, weight_version=version)
    assert not controller.context_lock.locked
    with pytest.raises(RuntimeError, match="failed"):
        await controller.dynamo_acquire_rollout(version="1")


@pytest.mark.asyncio
async def test_driver_passes_trainer_version_before_publishing_to_executor(monkeypatch):
    import miles.ray.placement_group as module

    args = _args()
    controller, observations, _ = _controller()
    events = []

    async def update(**kwargs):
        events.append("transfer")
        observations[0] = EngineObservation(123, "17")
        return 17

    async def publish(version, **kwargs):
        ticket = await controller.dynamo_acquire_rollout(version=str(version))
        await controller.dynamo_release_rollout(ticket=ticket, success=True)
        events.append("publish")

    monkeypatch.setattr(module.FTTestActionOrchestrationExecutor, "from_args", lambda *a, **kw: None)
    monkeypatch.setattr(module, "_maybe_log_inference_engine_weight_checksums", AsyncMock())
    await module.update_weights(
        args, SimpleNamespace(update_weights=update), SimpleNamespace(set_weight_version=publish), controller
    )
    assert events == ["transfer", "publish"]
