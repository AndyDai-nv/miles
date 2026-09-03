"""Render the launch contract for a ``dynamo.sglang`` rollout worker.

The rollout refactor renders Miles' resolved SGLang ``ServerArgs`` separately.
This module deliberately accepts that rendered argv instead of duplicating the
topology, GPU, PD-disaggregation, LoRA, or launch-gate calculations.  The
Dynamo adapter only owns the wrapper module, Dynamo runtime arguments, and the
environment needed by Dynamo's system/control server.

All helpers are pure.  They do not import Dynamo or SGLang, start a process, or
depend on Ray worker specs, so this contract can land before the rollout
refactor and be wired into its ``CommandWorkerSpec`` later.
"""

from __future__ import annotations

import json
import shlex
import sys
from collections.abc import Mapping, Sequence
from typing import Any

from miles.backends.dynamo_utils.arguments import DYNAMO_UPSTREAM_DEFAULTS
from miles.backends.dynamo_utils.dynamo_config import DynamoConfig, compute_dynamo_model_namespace

DYNAMO_SGLANG_MODULE = "dynamo.sglang"
DYNAMO_WORKER_COMPONENT = "backend"
DYNAMO_WORKER_ENDPOINT = "generate"

# Read the same pinned contract as the frontend renderer without making the
# worker-side module depend on frontend implementation details.
REQUEST_PLANE = DYNAMO_UPSTREAM_DEFAULTS["request-plane"]
EVENT_PLANE = DYNAMO_UPSTREAM_DEFAULTS["event-plane"]

ENV_DISCOVERY_BACKEND = "DYN_DISCOVERY_BACKEND"
ENV_ENDPOINT = "DYN_ENDPOINT"
ENV_EVENT_PLANE = "DYN_EVENT_PLANE"
ENV_FILE_KV = "DYN_FILE_KV"
ENV_NAMESPACE = "DYN_NAMESPACE"
ENV_REQUEST_PLANE = "DYN_REQUEST_PLANE"
ENV_SYSTEM_PORT = "DYN_SYSTEM_PORT"

_KV_EVENTS_FLAG = "--kv-events-config"
_DYNAMO_OWNED_FLAGS = frozenset(
    {
        "--namespace",
        "--endpoint",
        "--discovery-backend",
        "--request-plane",
        "--event-plane",
        "--enable-rl",
        "--no-enable-rl",
    }
)


def compute_dynamo_worker_endpoint(config: DynamoConfig, *, model_id: str) -> str:
    """Return the discovery endpoint shared by one model's frontend and workers."""
    namespace = compute_dynamo_model_namespace(config.namespace, model_id)
    return f"dyn://{namespace}.{DYNAMO_WORKER_COMPONENT}.{DYNAMO_WORKER_ENDPOINT}"


def compute_dynamo_engine_args(config: DynamoConfig, *, model_id: str) -> dict[str, Any]:
    """Return only the arguments owned by the Dynamo worker wrapper.

    SGLang arguments are intentionally absent.  The caller appends the output
    of the refactor's ``server_args_to_argv`` unchanged.
    """
    namespace = compute_dynamo_model_namespace(config.namespace, model_id)
    return {
        "namespace": namespace,
        "endpoint": compute_dynamo_worker_endpoint(config, model_id=model_id),
        "discovery-backend": config.discovery_backend,
        "request-plane": REQUEST_PLANE,
        "event-plane": EVENT_PLANE,
        "enable-rl": config.enable_rl,
    }


def dynamo_engine_args_to_argv(args: Mapping[str, Any]) -> list[str]:
    """Render Dynamo-owned worker arguments while preserving insertion order."""
    argv: list[str] = []
    for name, value in args.items():
        if value is None:
            continue
        if isinstance(value, bool):
            argv.append(f"--{name}" if value else f"--no-{name}")
            continue
        argv += [f"--{name}", str(value)]
    return argv


def compute_dynamo_engine_argv(
    config: DynamoConfig,
    *,
    model_id: str,
    sglang_argv: Sequence[str],
) -> list[str]:
    """Compose a complete ``dynamo.sglang`` process argv.

    ``sglang_argv`` must already have been produced by Miles' SGLang renderer.
    Keeping it opaque is what guarantees that both backends use the same model
    and topology calculation after the rollout refactor lands.
    """
    sglang_tokens = _validate_sglang_argv(sglang_argv)
    _validate_kv_events_contract(config, sglang_tokens)
    dynamo_argv = dynamo_engine_args_to_argv(compute_dynamo_engine_args(config, model_id=model_id))
    return [sys.executable, "-m", DYNAMO_SGLANG_MODULE, *dynamo_argv, *sglang_tokens]


def compute_dynamo_engine_launch_cmd(
    config: DynamoConfig,
    *,
    model_id: str,
    sglang_argv: Sequence[str],
) -> str:
    """Return the shell-safe command for one Dynamo SGLang worker."""
    return shlex.join(compute_dynamo_engine_argv(config, model_id=model_id, sglang_argv=sglang_argv))


def compute_dynamo_engine_env_vars(
    config: DynamoConfig,
    *,
    model_id: str,
    system_port: int,
    sglang_port: int,
) -> dict[str, str]:
    """Return Dynamo-specific environment to add to Miles' SGLang env.

    The system port exposes Dynamo ``/health`` and ``/engine/*`` routes.  It is
    deliberately distinct from SGLang's internal HTTP port; conflating them was
    a limitation of the pre-refactor proof of concept.
    """
    system_port = _validate_port("system_port", system_port)
    sglang_port = _validate_port("sglang_port", sglang_port)
    if system_port == sglang_port:
        raise ValueError("Dynamo system_port must be different from the SGLang internal port")

    namespace = compute_dynamo_model_namespace(config.namespace, model_id)
    env_vars = {
        ENV_SYSTEM_PORT: str(system_port),
        ENV_NAMESPACE: namespace,
        ENV_ENDPOINT: compute_dynamo_worker_endpoint(config, model_id=model_id),
        ENV_REQUEST_PLANE: REQUEST_PLANE,
        ENV_EVENT_PLANE: EVENT_PLANE,
        # SGLang's KV block hashes cross process boundaries. Dynamo also sets
        # this in its module entrypoint, but making it explicit keeps offline
        # manifests and live launches identical.
        "PYTHONHASHSEED": "0",
    }
    if config.discovery_backend is not None:
        env_vars[ENV_DISCOVERY_BACKEND] = config.discovery_backend
    if config.file_kv_path is not None:
        env_vars[ENV_FILE_KV] = config.file_kv_path
    return env_vars


def _validate_sglang_argv(sglang_argv: Sequence[str]) -> list[str]:
    if isinstance(sglang_argv, str):
        raise TypeError("sglang_argv must be a sequence of tokens, not a shell command string")

    tokens = list(sglang_argv)
    if any(not isinstance(token, str) for token in tokens):
        raise TypeError("every sglang_argv token must be a string")

    collisions = sorted(
        {
            option
            for token in tokens
            if token.startswith("--") and (option := token.partition("=")[0]) in _DYNAMO_OWNED_FLAGS
        }
    )
    if collisions:
        raise ValueError(
            "sglang_argv contains Dynamo-owned options: "
            f"{collisions}; configure them through DynamoConfig so the wrapper has one source of truth"
        )
    return tokens


def _validate_kv_events_contract(config: DynamoConfig, sglang_argv: Sequence[str]) -> None:
    values = _option_values(sglang_argv, _KV_EVENTS_FLAG)
    if len(values) > 1:
        raise ValueError(f"{_KV_EVENTS_FLAG} may be supplied at most once")

    if not config.uses_kv_routing or not config.router_kv_events:
        if values:
            raise ValueError(f"{_KV_EVENTS_FLAG} publishes cache events, but this Dynamo config does not consume them")
        return

    if not values:
        raise ValueError(
            "Dynamo KV routing with router_kv_events enabled requires SGLang argv to contain "
            f"{_KV_EVENTS_FLAG}; its endpoint must come from an allocated worker port"
        )

    try:
        value = json.loads(values[0])
    except json.JSONDecodeError as exception:
        raise ValueError(f"{_KV_EVENTS_FLAG} must contain valid JSON") from exception
    if not isinstance(value, dict):
        raise ValueError(f"{_KV_EVENTS_FLAG} must contain a JSON object")
    if value.get("publisher") != EVENT_PLANE:
        raise ValueError(
            f"{_KV_EVENTS_FLAG} publisher must be {EVENT_PLANE!r} to match the configured Dynamo event plane"
        )
    if not isinstance(value.get("endpoint"), str) or not value["endpoint"].strip():
        raise ValueError(f"{_KV_EVENTS_FLAG} must contain a non-empty endpoint")


def _option_values(argv: Sequence[str], flag: str) -> list[str]:
    values: list[str] = []
    index = 0
    while index < len(argv):
        token = argv[index]
        if token == flag:
            if index + 1 >= len(argv) or argv[index + 1].startswith("--"):
                raise ValueError(f"{flag} requires a value")
            values.append(argv[index + 1])
            index += 2
            continue
        if token.startswith(f"{flag}="):
            values.append(token.split("=", 1)[1])
        index += 1
    return values


def _validate_port(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 65535:
        raise ValueError(f"{name} must be an integer between 1 and 65535")
    return value
