import shlex
from types import SimpleNamespace

import pytest
from tests.fast.backends.sglang_utils.conftest import make_engine_args
from tests.fast.ray.rollout.conftest import make_args

from miles.backends.sglang_utils.server_args_utils import parse_server_args_argv
from miles.backends.sglang_utils.sglang_api_client import WorkerType
from miles.backends.sglang_utils.sglang_config import ServerGroupConfig
from miles.backends.sglang_utils.sglang_engine import compute_engine_launch_cmd
from miles.ray.specs.inference import InferenceEngineSpec, RouterSpec
from miles.router.backend import DynamoRouterBackend, make_router_backend
from miles.router.dynamo.config import DynamoConfig
from miles.router.dynamo.sidecar import sidecar_command, sidecar_env, sidecar_overrides, sidecar_ports
from miles.utils.args.configs.scaling import ScalingConfig
from miles.utils.external_utils.command_utils.helm_backend.launcher.values.builder import (
    build_values,
    compute_static_connections,
)
from miles.utils.external_utils.command_utils.helm_backend.launcher.values.misc import LaunchPlan
from miles.utils.workers.launch_gate import GATE_PORT_NAME
from miles.utils.workers.worker_spec import HostAndPort, LaunchCommandContext, WorkerLaunchContext


@pytest.fixture
def config():
    return DynamoConfig(
        namespace="test-run",
        discovery={"backend": "etcd", "endpoints": [{"host": "localhost", "port": 2379}]},
        pins={"dynamo": "a" * 40, "sglang": "b" * 40},
    )


@pytest.mark.parametrize("nats_server", [None, "nats://nats:4222"])
def test_miles_allocates_and_launches_sidecar_with_engine(config, monkeypatch, nats_server):
    config = config.model_copy(update={"nats_server": nats_server})
    monkeypatch.setenv("NATS_SERVER", "nats://other-run:4222")
    captured = {}

    def launch(**kwargs):
        captured.update(kwargs)
        return "python -m sglang.launch_server"

    monkeypatch.setattr("miles.ray.specs.inference.compute_engine_launch_cmd", launch)
    args = make_args(router="dynamo", dynamo_config=config, sglang_api_key="test-key", rollout_num_gpus_per_engine=2)
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
    if nats_server is None:
        assert "NATS_SERVER" not in spec.env_var(ctx)
    else:
        assert spec.env_var(ctx)["NATS_SERVER"] == nats_server
    assert spec.category == "inference_engine"
    assert spec.scheduling(scaling).pg_name == "rollout"


def test_sidecar_does_not_accept_conflicting_admission_env(config):
    config = config.model_copy(
        update={"sidecar": config.sidecar.model_copy(update={"env": {"DYN_SGLANG_CONTROLLER_MANAGED": "false"}})}
    )
    with pytest.raises(ValueError, match="CONTROLLER_MANAGED"):
        sidecar_env(config, api_key=None)


@pytest.mark.parametrize("api_key", [None, "test-key"])
def test_sidecar_env_uses_engine_credentials(config, monkeypatch, api_key):
    monkeypatch.setenv("DYN_SGLANG_API_KEY", "inherited-key")
    env = sidecar_env(config, api_key=api_key)
    assert env.get("DYN_SGLANG_API_KEY") == api_key


def test_allocated_listener_replaces_inherited_address_without_changing_engine_argv(config, monkeypatch):
    monkeypatch.setenv("DYN_SYSTEM_HOST", "old-host")
    monkeypatch.setenv("DYN_SYSTEM_PORT", "9999")
    env = sidecar_env(config, api_key=None)
    assert "DYN_SYSTEM_HOST" not in env
    assert "DYN_SYSTEM_PORT" not in env

    ctx = SimpleNamespace(self_addrs={"dynamo_system": HostAndPort(host="[::1]", port=18003)})
    engine_argv = ["python", "-m", "sglang.launch_server", "--model-path", "/models/a b;$HOME"]
    command = sidecar_command(shlex.join(engine_argv), ctx=ctx)
    assert shlex.split(command) == ["env", "DYN_SYSTEM_PORT=18003", "DYN_SYSTEM_HOST=::1", *engine_argv]


@pytest.mark.parametrize("api_key", [None, "test-key"])
@pytest.mark.parametrize("dp_size", [1, 2, 4])
def test_native_sidecar_arguments_preserve_api_key(config, api_key, sidecar_ctx, group, dp_size):
    ctx = sidecar_ctx
    args = make_engine_args(sglang_api_key=api_key, sglang_dp_size=dp_size)
    overrides = sidecar_overrides(config, sglang=args.sglang, group=group, ctx=ctx)
    command = compute_engine_launch_cmd(
        args,
        interpreter_prefix=["python"],
        node_rank=0,
        worker_type=WorkerType.REGULAR,
        base_gpu_id=0,
        sglang_overrides=overrides,
        num_gpus_per_engine=8,
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
    assert parsed.tp_size == 8
    assert parsed.dp_size == dp_size


@pytest.mark.parametrize("host", ["127.0.0.1", "[::1]", "0.0.0.0"])
@pytest.mark.parametrize("nats_server", [None, "nats://nats:4222"])
def test_frontend_uses_shared_router_spec_and_preallocation_environment(config, monkeypatch, host, nats_server):
    monkeypatch.setenv("NATS_SERVER", "nats://other-run:4222")
    config = config.model_copy(
        update={
            "nats_server": nats_server,
            "frontend": config.frontend.model_copy(
                update={"router_mode": "kv", "extra_args": ("--log-level", "debug")}
            ),
        }
    )
    args = make_args(router="dynamo", dynamo_config=config)
    spec = RouterSpec.create(args)[0]
    scaling = ScalingConfig.model_validate({name: getattr(args, name) for name in ScalingConfig.model_fields})
    env_context = WorkerLaunchContext(
        args=args, cell_index=0, worker_in_cell_index=0, num_workers_per_cell=1, gpu_ids=[]
    )

    env = spec.env_var(env_context)

    assert isinstance(make_router_backend(args), DynamoRouterBackend)
    assert make_router_backend(args).routing_headers("sample-key", args.sglang.models) == {}
    assert make_router_backend(args).routing_headers(None, []) == {}
    assert spec.name == "inference-router-0"
    assert spec.scheduling(scaling).num_gpus_per_worker == 0
    if nats_server is None:
        assert "NATS_SERVER" not in env
    else:
        assert env["NATS_SERVER"] == nats_server
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


@pytest.mark.parametrize(
    "key,value",
    [("NATS_SERVER", "nats://other-run:4222"), ("DYN_HTTP_HOST", "127.0.0.1"), ("DYN_HTTP_PORT", "20000")],
)
def test_frontend_rejects_explicit_environment_conflicts(config, key, value):
    config = config.model_copy(update={"env": {key: value}})
    args = make_args(router="dynamo", dynamo_config=config)
    spec = RouterSpec.create(args)[0]
    context = WorkerLaunchContext(args=args, cell_index=0, worker_in_cell_index=0, num_workers_per_cell=1, gpu_ids=[])

    with pytest.raises(ValueError, match=key):
        spec.env_var(context)


def test_frontend_rejects_pd_disaggregation(config):
    args = make_args(router="dynamo", dynamo_config=config, rollout_external_router_pd=True)
    backend = make_router_backend(args)

    with pytest.raises(AssertionError, match="dynamo router does not support PD disaggregation"):
        backend.launch_argv(None, model_cfg=args.sglang.models[0], interpreter_prefix=["python"])


def test_frontend_command_and_environment_render_through_shared_helm_path(config):
    args = make_args(router="dynamo", dynamo_config=config, sglang_router_port=31000)
    specs = RouterSpec.create(args)
    scaling = ScalingConfig.model_validate({name: getattr(args, name) for name in ScalingConfig.model_fields})
    plan = LaunchPlan(
        run_id="260101-000000-000",
        release="miles-run-test",
        namespace="rl",
        state_file="",
        orchestrator_command=[],
        worker_argv=[],
    )

    values = build_values(
        specs,
        plan,
        scaling=scaling,
        static_connections=compute_static_connections(specs, scaling=scaling),
    )

    (entry,) = values.run.static_workers
    assert entry.command[entry.command.index("-m") + 1] == "dynamo.frontend"
    assert entry.command[entry.command.index("--http-host") + 1] == "0.0.0.0"
    assert entry.command[entry.command.index("--http-port") + 1] == "31000"
    assert "NATS_SERVER" not in entry.env
    assert entry.env["DYN_NAMESPACE"] == config.namespace


@pytest.fixture
def sidecar_ctx():
    return SimpleNamespace(
        self_addrs={
            "primary": HostAndPort(host="127.0.0.1", port=18000),
            "dynamo_grpc": HostAndPort(host="127.0.0.1", port=18001),
            "dynamo_kv": HostAndPort(host="127.0.0.1", port=18002),
        }
    )


@pytest.mark.parametrize("scope", ["base", "group"])
@pytest.mark.parametrize(
    "name,value",
    [
        ("grpc_mode", True),
        ("smg_grpc_mode", True),
        ("sidecar", "other.sidecar"),
        ("grpc_port", 9999),
        ("kv_events_config", '{"publisher": "other"}'),
        ("sidecar_args", ["--namespace", "other"]),
    ],
)
def test_sidecar_rejects_conflicting_effective_settings(config, sidecar_ctx, scope, name, value, group):
    args = make_engine_args(**{f"sglang_{name}": value}) if scope == "base" else make_engine_args()
    if scope == "group":
        group = group.model_copy(update={"overrides": {**group.overrides, name: value}})
    with pytest.raises(ValueError, match="legacy gRPC mode" if name.endswith("grpc_mode") else name):
        sidecar_overrides(config, sglang=args.sglang, group=group, ctx=sidecar_ctx)


def test_sidecar_uses_group_precedence_without_mutating_config(config, sidecar_ctx, group):
    args = make_engine_args(sglang_grpc_mode=True, sglang_sidecar="other.sidecar")
    group = group.model_copy(
        update={"overrides": {**group.overrides, "grpc_mode": False, "sidecar": "dynamo.sglang.sidecar"}}
    )
    overrides = sidecar_overrides(config, sglang=args.sglang, group=group, ctx=sidecar_ctx)
    assert overrides["sidecar"] == "dynamo.sglang.sidecar"
    assert args.sglang.base_args["grpc_mode"] is True
    assert "grpc_port" not in group.overrides


def test_sidecar_rejects_group_streaming_opt_out(config, sidecar_ctx, group):
    args = make_engine_args()
    group = group.model_copy(update={"overrides": {"incremental_streaming_output": False}})
    with pytest.raises(ValueError, match="incremental_streaming_output"):
        sidecar_overrides(config, sglang=args.sglang, group=group, ctx=sidecar_ctx)


@pytest.fixture
def group():
    return ServerGroupConfig(worker_type=WorkerType.REGULAR, num_gpus_per_engine=2, needs_offload=False)


@pytest.mark.parametrize("dp_size", [1, 2, 4])
def test_kv_event_ports_reserve_every_dp_rank(dp_size):
    ports = {port.name: port for port in sidecar_ports(dp_size=dp_size)}
    kv = ports["dynamo_kv"]
    assert kv.allow_dynamic
    assert kv.num_consecutive == dp_size
    first = kv.effective_static_port(worker_in_pod_index=0)
    second = kv.effective_static_port(worker_in_pod_index=1)
    assert first + dp_size <= second
    assert ports["dynamo_grpc"].num_consecutive == 1
    assert ports["dynamo_system"].num_consecutive == 1


@pytest.mark.parametrize("scope", ["base", "group"])
@pytest.mark.parametrize("dp_size", [1, 2])
def test_engine_spec_reserves_effective_dp_ports(config, scope, dp_size):
    args = make_args(dynamo_config=config, sglang_dp_size=dp_size if scope == "base" else 1)
    if scope == "group":
        model = args.sglang.models[0]
        group = model.server_groups[0]
        group = group.model_copy(update={"overrides": {**group.overrides, "dp_size": dp_size}})
        args.sglang = args.sglang.model_copy(update={"models": [model.model_copy(update={"server_groups": [group]})]})
    spec = InferenceEngineSpec.create(args)[0]
    ports = {port.name: port for port in spec.port_infos}
    assert ports["dynamo_kv"].num_consecutive == dp_size
    assert ports["dist_init"].num_consecutive == 30 + dp_size
