import os
import shlex

from miles.backends.dynamo_utils.config import Address, DynamoConfig
from miles.backends.dynamo_utils.external_provider import DynamoExternalEngineProvider
from miles.backends.dynamo_utils.frontend_args import frontend_env, frontend_launch
from miles.utils.workers.argv_utils import python_argv_prefix
from miles.utils.workers.worker_spec import CommandWorkerSpec, PortInfo, SchedulingSpec


def external_engine_provider(args, *, capability):
    return DynamoExternalEngineProvider(config=args.dynamo_config, args=args)


def frontend_spec(args, *, config: DynamoConfig, pool_name: str) -> CommandWorkerSpec:
    def launch(ctx):
        primary = ctx.self_addrs["primary"]
        address = config.frontend.address or Address(
            host="::" if ":" in primary.host else "0.0.0.0", port=primary.port
        )
        argv, env = frontend_launch(
            config, address=address, interpreter_prefix=python_argv_prefix(), inherited_env=os.environ
        )
        if any(token.startswith("--tls-") for token in argv) or any(
            env.get(key) for key in ("DYN_TLS_CERT_PATH", "DYN_TLS_KEY_PATH", "DYN_TLS_CLIENT_CA_CERT_PATH")
        ):
            raise ValueError("Miles Dynamo rollout currently requires a private HTTP frontend without HTTP TLS")
        return argv, env

    configured = config.frontend.address
    return CommandWorkerSpec(
        name=pool_name,
        port_infos=[
            PortInfo(
                name="primary", static_port=configured.port if configured else 8000, allow_dynamic=configured is None
            )
        ],
        env_var=lambda _ctx: frontend_env(config, inherited_env=os.environ),
        scheduling=SchedulingSpec.single(num_gpus_per_worker=0, pin_to_head=args.pin_rollout_manager_to_head),
        launch_command=lambda ctx: shlex.join(launch(ctx)[0]),
    )
