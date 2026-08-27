"""Resolved configuration for the Dynamo rollout backend.

One resolution step, one immutable object. Every later stage (frontend argv,
engine argv, api client) reads this instead of poking at ``args``, so a default
is decided exactly once and cannot drift between the frontend and the engines.

That drift is not hypothetical: reading ``args.dynamo_router_mode`` directly
returns ``None`` rather than a usable default, and two call sites that each
apply their own fallback can disagree about whether KV routing is on.

Resolving does not mean inventing. Where miles has no reason to differ from
dynamo, the resolved value stays ``None`` and the flag is never rendered, so
dynamo applies its own default. See ``DYNAMO_UPSTREAM_DEFAULTS``.
"""

from __future__ import annotations

from typing import Literal

from miles.utils.pydantic_utils import FrozenStrictBaseModel

DiscoveryBackend = Literal["kubernetes", "etcd", "file", "mem"]
RouterMode = Literal[
    "round-robin",
    "random",
    "power-of-two",
    "kv",
    "direct",
    "least-loaded",
    "device-aware-weighted",
]

# Repeated from dynamo rather than chosen by miles; see DYNAMO_UPSTREAM_DEFAULTS.
DEFAULT_ROUTER_MODE: RouterMode = "round-robin"


class DynamoConfig(FrozenStrictBaseModel):
    """Everything the Dynamo backend needs, with no unresolved defaults left."""

    namespace: str
    # `None` means "miles has no opinion": the flag is not rendered and dynamo
    # applies its own default. Only set what a run actually asked for.
    discovery_backend: DiscoveryBackend | None
    file_kv_path: str | None

    router_mode: RouterMode
    router_kv_events: bool
    router_ttl_secs: float
    router_predicted_ttl_secs: float | None
    router_min_initial_workers: int
    router_queue_threshold: float | None

    enable_rl: bool

    @property
    def uses_kv_routing(self) -> bool:
        return self.router_mode == "kv"


def is_dynamo_backend(args) -> bool:
    return getattr(args, "rollout_backend", "sglang") == "dynamo"


def resolve_dynamo_config(args) -> DynamoConfig:
    """Turn ``args`` into a fully resolved :class:`DynamoConfig`.

    Raises if the arguments describe a configuration dynamo will not honour, so a
    contradiction surfaces before any process is launched rather than as a silently
    degraded router.
    """
    assert is_dynamo_backend(args), (
        "resolve_dynamo_config was called while --rollout-backend is "
        f"{getattr(args, 'rollout_backend', 'sglang')!r}; the dynamo configuration only "
        "describes runs that actually launch dynamo"
    )

    namespace = args.dynamo_namespace or _default_namespace(args)
    discovery_backend: DiscoveryBackend | None = args.dynamo_discovery_backend
    router_mode: RouterMode = args.dynamo_router_mode or DEFAULT_ROUTER_MODE

    if args.dynamo_router_predicted_ttl_secs is not None and not args.dynamo_router_kv_events:
        raise ValueError(
            "--dynamo-router-predicted-ttl-secs enables a predict-on-route side indexer that reads "
            "the KV event stream, so it cannot be combined with --no-dynamo-router-kv-events. Drop "
            "one of the two, or use --dynamo-router-ttl-secs for pure prediction-based routing."
        )

    if router_mode != "kv" and args.dynamo_router_predicted_ttl_secs is not None:
        raise ValueError(
            f"--dynamo-router-predicted-ttl-secs only applies to KV routing, but --dynamo-router-mode "
            f"is {router_mode!r}. Pass --dynamo-router-mode kv, or drop the predicted TTL."
        )

    if args.dynamo_file_kv_path is not None and discovery_backend != "file":
        raise ValueError(
            f"--dynamo-file-kv-path only backs the 'file' discovery backend, but "
            f"--dynamo-discovery-backend is {discovery_backend!r}. Pass "
            f"--dynamo-discovery-backend file, or drop the path."
        )

    return DynamoConfig(
        namespace=namespace,
        discovery_backend=discovery_backend,
        file_kv_path=args.dynamo_file_kv_path,
        router_mode=router_mode,
        router_kv_events=args.dynamo_router_kv_events,
        router_ttl_secs=args.dynamo_router_ttl_secs,
        router_predicted_ttl_secs=args.dynamo_router_predicted_ttl_secs,
        router_min_initial_workers=args.dynamo_router_min_initial_workers,
        router_queue_threshold=args.dynamo_router_queue_threshold,
        enable_rl=args.dynamo_enable_rl,
    )


def _default_namespace(args) -> str:
    """Scope discovery to this run so concurrent runs on one host stay invisible to each other.

    Dynamo's own default is the global ``dynamo``, but its Kubernetes operator overrides that
    per deployment (``{k8s_namespace}-{dgd_name}``). Miles has no operator on the Ray path, so
    it does the equivalent itself.
    """
    for attr in ("run_uuid", "run_id", "wandb_run_name"):
        if value := getattr(args, attr, None):
            return f"miles-{value}"
    return "miles"
