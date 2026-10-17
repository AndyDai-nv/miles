import shlex
from types import SimpleNamespace

import pytest
from tests.fast.backends.sglang_utils.conftest import make_engine_args
from tests.fast.ray.rollout.conftest import make_args
from tests.fast.ray.specs.test_inference import _make_model_cfg, _make_router_ctx

from miles.backends.dynamo_utils.config import DynamoConfig, runtime_env
from miles.backends.dynamo_utils.launch import sidecar_env, sidecar_overrides
from miles.backends.sglang_utils.server_args_utils import parse_server_args_argv
from miles.backends.sglang_utils.sglang_api_client import WorkerType
from miles.backends.sglang_utils.sglang_engine import compute_engine_launch_cmd
from miles.ray.specs.inference import _compute_spec_inference_engine, _compute_spec_router
from miles.utils.workers.launch_gate import GATE_PORT_NAME
from miles.utils.workers.worker_spec import HostAndPort, LaunchCommandContext, WorkerLaunchContext


@pytest.fixture
def config():
    return DynamoConfig(
        namespace="test-run",
        discovery={"backend": "etcd", "endpoints": [{"host": "localhost", "port": 2379}]},
        pins={"dynamo": "a" * 40, "sglang": "b" * 40},
    )


def test_namespace_and_nats_are_run_owned(config):
    with pytest.raises(ValueError, match="global"):
        DynamoConfig(**(config.model_dump() | {"namespace": "dynamo"}))
    assert runtime_env(config, inherited_env={"NATS_SERVER": "nats://wrong:4222"})["NATS_SERVER"] == config.nats_server
    with pytest.raises(ValueError, match="NATS_SERVER"):
        runtime_env(config.model_copy(update={"env": {"NATS_SERVER": "wrong"}}), inherited_env={})


def test_frontend_is_a_router_spec_with_native_options(config):
    config = config.model_copy(
        update={"frontend": config.frontend.model_copy(update={"extra_args": ("--router-mode", "kv")})}
    )
    args = make_args(dynamo_config=config, sglang_router_port=None)
    spec = _compute_spec_router(args, 0, _make_model_cfg("regular"))
    argv = shlex.split(spec.launch_command(_make_router_ctx()))
    assert "dynamo.frontend" in argv
    assert argv[argv.index("--http-port") + 1] == "20000"
    assert argv[-2:] == ["--router-mode", "kv"]
    assert (
        spec.env_var(WorkerLaunchContext(cell_index=0, worker_in_cell_index=0, gpu_ids=[]))["NATS_SERVER"]
        == config.nats_server
    )


def test_miles_allocates_and_launches_sidecar_with_engine(config, monkeypatch):
    captured = {}

    def launch(**kwargs):
        captured.update(kwargs)
        return "python -m sglang.launch_server"

    monkeypatch.setattr("miles.ray.specs.inference.compute_engine_launch_cmd", launch)
    model = _make_model_cfg("regular")
    args = make_args(dynamo_config=config, sglang_api_key="test-key")
    spec = _compute_spec_inference_engine(args, 0, 0, model, model.server_groups[0])
    ctx = LaunchCommandContext(
        cell_index=0,
        worker_in_cell_index=0,
        gpu_ids=[4, 5],
        local_gpu_ids=[4, 5],
        pool_addrs={},
        self_addrs={
            port.name: HostAndPort(host="127.0.0.1", port=20000 + index) for index, port in enumerate(spec.port_infos)
        },
    )
    command = shlex.split(spec.launch_command(ctx))
    assert command[0] == "env"
    assert "sglang.launch_server" in command
    overrides = captured["sglang_overrides"]
    assert overrides["sidecar"] == "dynamo.sglang.sidecar"
    assert overrides["grpc_port"] == ctx.self_addrs["dynamo_grpc"].port
    assert overrides["incremental_streaming_output"] is True
    assert "--controller-managed" in overrides["sidecar_args"]
    assert captured["gated_launch_port"] == ctx.self_addrs[GATE_PORT_NAME].port
    assert spec.env_var(ctx)["DYN_SGLANG_API_KEY"] == "test-key"
    assert spec.category == "inference_engine"
    assert spec.scheduling.pg_name == "rollout"


def test_sidecar_does_not_accept_conflicting_admission_env(config):
    config = config.model_copy(
        update={"sidecar": config.sidecar.model_copy(update={"env": {"DYN_SGLANG_CONTROLLER_MANAGED": "false"}})}
    )
    with pytest.raises(ValueError, match="CONTROLLER_MANAGED"):
        sidecar_env(config, api_key=None)


def test_native_sglang_parser_accepts_embedded_sidecar(config):
    ctx = SimpleNamespace(
        self_addrs={
            "primary": HostAndPort(host="127.0.0.1", port=18000),
            "dynamo_grpc": HostAndPort(host="127.0.0.1", port=18001),
            "dynamo_kv": HostAndPort(host="127.0.0.1", port=18002),
        }
    )
    overrides = sidecar_overrides(config, overrides={}, ctx=ctx)
    command = compute_engine_launch_cmd(
        make_engine_args(sglang_api_key="test-key"),
        interpreter_prefix=["python"],
        node_rank=0,
        worker_type=WorkerType.REGULAR,
        base_gpu_id=0,
        sglang_overrides=overrides,
        num_gpus_per_engine=2,
        dist_init_addr="127.0.0.1:18003",
        nccl_port=18004,
        host="127.0.0.1",
        port=18000,
        disaggregation_bootstrap_port=None,
        engine_info_bootstrap_port=18005,
        gated_launch_port=18006,
        random_seed=42,
    )
    parsed = parse_server_args_argv(shlex.split(command)[3:])
    assert parsed.sidecar == "dynamo.sglang.sidecar"
    assert parsed.sidecar_args == overrides["sidecar_args"]
    assert parsed.grpc_port == 18001
    assert parsed.api_key == "test-key"
    assert parsed.tp_size == 2
