import shlex
from types import SimpleNamespace
import pytest
from sglang.srt.arg_groups.serving_hook import handle_deprecated_args
from tests.fast.backends.sglang_utils.conftest import make_engine_args
from tests.fast.ray.rollout.conftest import make_args
from miles.backends.dynamo_utils.config import DynamoConfig
from miles.backends.dynamo_utils.launch import sidecar_env, sidecar_overrides
from miles.backends.sglang_utils.server_args_utils import parse_server_args_argv
from miles.backends.sglang_utils.sglang_api_client import WorkerType
from miles.backends.sglang_utils.sglang_engine import compute_engine_launch_cmd
from miles.ray.specs.inference import InferenceEngineSpec, RouterSpec
from miles.router.backend import DynamoRouterBackend, make_router_backend
from miles.utils.args.configs.scaling import ScalingConfig
from miles.utils.workers.launch_gate import GATE_PORT_NAME
from miles.utils.workers.worker_spec import HostAndPort, LaunchCommandContext, WorkerLaunchContext


@pytest.fixture
def config():
    return DynamoConfig(
        namespace="test-run",
        discovery={"backend": "etcd", "endpoints": [{"host": "localhost", "port": 2379}]},
        pins={"dynamo": "a" * 40, "sglang": "b" * 40},
    )


def test_miles_allocates_and_launches_sidecar_with_engine(config, monkeypatch):
    captured = {}

    def launch(**kwargs):
        captured.update(kwargs)
        return "python -m sglang.launch_server"

    monkeypatch.setattr("miles.ray.specs.inference.compute_engine_launch_cmd", launch)
    args = make_args(dynamo_config=config, sglang_api_key="test-key", rollout_num_gpus_per_engine=2)
    scaling = ScalingConfig.model_validate({name: getattr(args, name) for name in ScalingConfig.model_fields})
    spec = InferenceEngineSpec.create(args)[0]
    ctx = LaunchCommandContext(
        args=args,
        scaling=scaling,
        cell_index=0,
        worker_in_cell_index=0,
        num_workers_per_cell=1,
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
    assert spec.scheduling(scaling).pg_name == "rollout"


def test_sidecar_does_not_accept_conflicting_admission_env(config):
    config = config.model_copy(
        update={"sidecar": config.sidecar.model_copy(update={"env": {"DYN_SGLANG_CONTROLLER_MANAGED": "false"}})}
    )
    with pytest.raises(ValueError, match="CONTROLLER_MANAGED"):
        sidecar_env(config, api_key=None)


@pytest.mark.parametrize("api_key", [None, "test-key"])
def test_native_sidecar_arguments_and_upstream_auth_validation(config, api_key):
    ctx = SimpleNamespace(
        self_addrs={
            "primary": HostAndPort(host="127.0.0.1", port=18000),
            "dynamo_grpc": HostAndPort(host="127.0.0.1", port=18001),
            "dynamo_kv": HostAndPort(host="127.0.0.1", port=18002),
        }
    )
    overrides = sidecar_overrides(config, overrides={}, ctx=ctx)
    command = compute_engine_launch_cmd(
        make_engine_args(sglang_api_key=api_key),
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
    assert parsed.api_key == api_key
    assert parsed.tp_size == 2
    handle_deprecated_args(parsed)


@pytest.mark.parametrize("host", ["127.0.0.1", "[::1]", "0.0.0.0"])
def test_frontend_uses_shared_router_spec_and_preallocation_environment(config, monkeypatch, host):
    monkeypatch.setenv("NATS_SERVER", "nats://other-run:4222")
    config = config.model_copy(
        update={
            "frontend": config.frontend.model_copy(
                update={"router_mode": "kv", "extra_args": ("--log-level", "debug")}
            )
        }
    )
    args = make_args(dynamo_config=config)
    spec = RouterSpec.create(args)[0]
    scaling = ScalingConfig.model_validate({name: getattr(args, name) for name in ScalingConfig.model_fields})
    env_context = WorkerLaunchContext(
        args=args, cell_index=0, worker_in_cell_index=0, num_workers_per_cell=1, gpu_ids=[]
    )

    env = spec.env_var(env_context)

    assert isinstance(make_router_backend(args), DynamoRouterBackend)
    assert spec.name == "inference-router-0"
    assert spec.scheduling(scaling).num_gpus_per_worker == 0
    assert env["NATS_SERVER"] == config.nats_server
    assert env["DYN_NAMESPACE"] == config.namespace
    assert env["DYN_ROUTER_MIN_INITIAL_WORKERS"] == "0"
    assert env["DYN_SGLANG_ENABLE_GENERATE"] == "1"
    assert "DYN_HTTP_HOST" not in env
    assert "DYN_HTTP_PORT" not in env

    context = LaunchCommandContext(
        **env_context.model_dump(),
        scaling=scaling,
        local_gpu_ids=[],
        pool_addrs={},
        self_addrs={"primary": HostAndPort(host=host, port=20000)},
    )
    argv = shlex.split(spec.launch_command(context))

    assert argv[: len(spec.interpreter_prefix)] == spec.interpreter_prefix
    assert argv[len(spec.interpreter_prefix) : len(spec.interpreter_prefix) + 2] == ["-m", "dynamo.frontend"]
    assert argv[argv.index("--http-host") + 1] == host.strip("[]")
    assert argv[argv.index("--http-port") + 1] == "20000"
    assert argv[argv.index("--router-min-initial-workers") + 1] == "0"
    assert argv[argv.index("--router-mode") + 1] == "kv"
    assert argv[-2:] == ["--log-level", "debug"]


def test_frontend_rejects_explicit_environment_conflicts(config):
    config = config.model_copy(update={"env": {"NATS_SERVER": "nats://other-run:4222"}})
    args = make_args(dynamo_config=config)
    spec = RouterSpec.create(args)[0]
    context = WorkerLaunchContext(args=args, cell_index=0, worker_in_cell_index=0, num_workers_per_cell=1, gpu_ids=[])

    with pytest.raises(ValueError, match="NATS_SERVER"):
        spec.env_var(context)


def test_frontend_rejects_pd_disaggregation(config):
    args = make_args(dynamo_config=config, rollout_external_router_pd=True)
    backend = make_router_backend(args)

    with pytest.raises(AssertionError, match="dynamo router does not support PD disaggregation"):
        backend.launch_argv(None, model_cfg=args.sglang.models[0], interpreter_prefix=["python"])
