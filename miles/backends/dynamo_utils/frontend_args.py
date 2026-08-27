"""Render the ``dynamo.frontend`` launch command.

`dynamo.frontend` is the Dynamo backend's counterpart to `sglang_router.launch_router`:
one HTTP process per model that routes generation requests to the workers. This module
mirrors the shape the sglang side uses -- `compute_*_args` produces a mapping, `*_to_argv`
renders it, and the launch command is `shlex.join`-ed -- so both backends can be driven
by the same launcher once one exists.

Everything here is a pure function of :class:`DynamoConfig` plus an address. No process is
started, no cluster is contacted, and the output for a given input never varies, so the
rendered command line can be asserted in a unit test and rendered offline by a manifest
generator.
"""

from __future__ import annotations

import shlex
import sys
from collections.abc import Mapping
from typing import Any

from miles.backends.dynamo_utils.arguments import DYNAMO_UPSTREAM_DEFAULTS
from miles.backends.dynamo_utils.dynamo_config import DynamoConfig

FRONTEND_MODULE = "dynamo.frontend"

# Dynamo's own defaults (DYNAMO_UPSTREAM_DEFAULTS), restated rather than omitted: miles pins
# the transports it was validated against, so a dynamo release that moves a default surfaces
# as a test diff rather than a silently different deployment. Neither plane needs an external
# broker -- NATS is opt-in on both since dynamo v0.8.
REQUEST_PLANE = DYNAMO_UPSTREAM_DEFAULTS["request-plane"]
EVENT_PLANE = DYNAMO_UPSTREAM_DEFAULTS["event-plane"]

# Serves SGLang's native `/generate` on the frontend, which is the API NVIDIA documents for
# SGLang RL rollouts. Without it the frontend only speaks the OpenAI-compatible routes.
ENV_ENABLE_SGLANG_GENERATE = "DYN_SGLANG_ENABLE_GENERATE"
ENV_FILE_KV = "DYN_FILE_KV"


def compute_dynamo_frontend_args(config: DynamoConfig, *, host: str, port: int) -> dict[str, Any]:
    """The frontend's flag values, before rendering.

    Returned as a mapping rather than a list so a caller can inspect or override a single
    flag without parsing a command line.
    """
    args: dict[str, Any] = {
        "http-host": host,
        "http-port": port,
        "namespace": config.namespace,
        # `None` when the run expressed no preference: dropped by the renderer so dynamo
        # applies its own default rather than miles choosing one for it.
        "discovery-backend": config.discovery_backend,
        "request-plane": REQUEST_PLANE,
        "event-plane": EVENT_PLANE,
        "router-mode": config.router_mode,
    }

    if config.uses_kv_routing:
        # Only meaningful under KV routing: the other modes keep no cache index to feed.
        args["router-kv-events"] = config.router_kv_events
        if config.router_kv_events:
            if config.router_predicted_ttl_secs is not None:
                args["router-predicted-ttl-secs"] = config.router_predicted_ttl_secs
        else:
            args["router-ttl-secs"] = config.router_ttl_secs

    if config.router_min_initial_workers:
        args["router-min-initial-workers"] = config.router_min_initial_workers

    if config.router_queue_threshold is not None:
        args["router-queue-threshold"] = config.router_queue_threshold

    return args


def dynamo_frontend_args_to_argv(args: Mapping[str, Any]) -> list[str]:
    """Render flag values as argv, preserving insertion order.

    A ``bool`` renders as its presence flag or dynamo's ``--no-`` negation; ``None`` is
    dropped so an unset option is never rendered as the string ``"None"``.
    """
    argv: list[str] = []
    for name, value in args.items():
        if value is None:
            continue
        if isinstance(value, bool):
            argv.append(f"--{name}" if value else f"--no-{name}")
            continue
        argv += [f"--{name}", str(value)]
    return argv


def compute_dynamo_frontend_launch_cmd(config: DynamoConfig, *, host: str, port: int) -> str:
    """The full shell command that starts one model's frontend."""
    frontend_args = compute_dynamo_frontend_args(config, host=host, port=port)
    return shlex.join([sys.executable, "-m", FRONTEND_MODULE, *dynamo_frontend_args_to_argv(frontend_args)])


def compute_dynamo_frontend_env_vars(config: DynamoConfig) -> dict[str, str]:
    """Environment the frontend process needs on top of its command line.

    ``DYN_SGLANG_ENABLE_GENERATE`` is off in dynamo by default; miles turns it on because
    serving SGLang's native ``/generate`` is the path NVIDIA documents for SGLang RL
    rollouts, and it keeps the rollout code on SGLang's own request and response shapes.
    """
    env_vars = {ENV_ENABLE_SGLANG_GENERATE: "1"}
    if config.file_kv_path is not None:
        env_vars[ENV_FILE_KV] = config.file_kv_path
    return env_vars
