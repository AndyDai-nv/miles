"""Command-line surface for the Dynamo rollout backend.

Kept separate from ``miles.backends.sglang_utils.arguments`` on purpose: the two
backends resolve their own configuration and share no argument namespace, so
neither one has to reason about the other's defaults.

Nothing here changes behaviour unless ``--rollout-backend dynamo`` is passed.

Miles does not invent defaults for dynamo. Every option below either repeats
dynamo's own documented default, or defaults to ``None`` so the flag is simply
not passed and dynamo applies its default itself. The two places miles
deliberately differs are marked as such and justified where they are set.
"""

import argparse

# Dynamo v1.4.0 defaults, transcribed from
# docs/fern/pages/reference/components/{runtime,frontend}-configuration.mdx.
# `test_dynamo_config.py` asserts miles agrees with every entry, so a dynamo
# release that moves one of these fails a unit test instead of silently changing
# how miles deploys.
DYNAMO_UPSTREAM_DEFAULTS = {
    "namespace": "dynamo",
    "discovery-backend": "etcd",
    "request-plane": "tcp",
    "event-plane": "zmq",
    "router-mode": "round-robin",
    "router-kv-events": True,
    "router-ttl-secs": 120.0,
    "router-predicted-ttl-secs": None,
    "router-min-initial-workers": 0,
    "router-queue-threshold": None,
    "enable-rl": False,
}

# Values accepted by dynamo's `--discovery-backend`. `kubernetes` is set automatically by
# the Dynamo operator; `mem` is single-process only, so miles never selects it by default.
DISCOVERY_BACKENDS = ("kubernetes", "etcd", "file", "mem")

# Values accepted by dynamo.frontend's `--router-mode`.
ROUTER_MODES = (
    "round-robin",
    "random",
    "power-of-two",
    "kv",
    "direct",
    "least-loaded",
    "device-aware-weighted",
)


def add_dynamo_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Register the rollout-backend selector and every ``--dynamo-*`` option."""
    parser.add_argument(
        "--rollout-backend",
        type=str,
        default="sglang",
        choices=["sglang", "dynamo"],
        help=(
            "Inference stack used during rollout. 'sglang' launches sgl_router plus "
            "sglang.launch_server workers. 'dynamo' launches dynamo.frontend plus "
            "dynamo.sglang workers instead."
        ),
    )

    group = parser.add_argument_group("Dynamo rollout backend")

    group.add_argument(
        "--dynamo-namespace",
        type=str,
        default=None,
        help=(
            "Dynamo namespace (DYN_NAMESPACE) that scopes service discovery. Miles defaults to "
            "a per-run value rather than dynamo's global 'dynamo', mirroring what dynamo's own "
            "Kubernetes operator does ('{k8s_namespace}-{dgd_name}'), so two runs on one host "
            "never discover each other's workers."
        ),
    )
    group.add_argument(
        "--dynamo-discovery-backend",
        type=str,
        default=None,
        choices=list(DISCOVERY_BACKENDS),
        help=(
            "How dynamo workers advertise themselves. Left unset the flag is not passed and "
            "dynamo applies its own default ('etcd'); 'file' needs no external service and "
            "suits a single host or shared storage. Since dynamo v0.8 neither etcd nor NATS "
            "is required for the request and event planes."
        ),
    )
    group.add_argument(
        "--dynamo-file-kv-path",
        type=str,
        default=None,
        help=(
            "Directory backing --dynamo-discovery-backend=file (DYN_FILE_KV). Left unset, "
            "dynamo uses its own default ($TMPDIR/dynamo_store_kv). Must be on shared storage "
            "for a multi-node run."
        ),
    )

    group.add_argument(
        "--dynamo-router-mode",
        type=str,
        default=None,
        choices=list(ROUTER_MODES),
        help="dynamo.frontend --router-mode. Defaults to dynamo's own default, 'round-robin'.",
    )
    group.add_argument(
        "--dynamo-router-kv-events",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Consume KV cache events from the workers (dynamo.frontend --router-kv-events). "
            "Disabling it makes the router predict cache state from its own routing decisions."
        ),
    )
    group.add_argument(
        "--dynamo-router-ttl-secs",
        type=float,
        default=120.0,
        help="Block TTL for prediction-based routing. Only used with --no-dynamo-router-kv-events.",
    )
    group.add_argument(
        "--dynamo-router-predicted-ttl-secs",
        type=float,
        default=None,
        help=(
            "Enable the predict-on-route side indexer with this TTL. Requires KV events, so it "
            "cannot be combined with --no-dynamo-router-kv-events."
        ),
    )
    group.add_argument(
        "--dynamo-router-min-initial-workers",
        type=int,
        default=0,
        help="Workers dynamo.frontend waits for before it starts serving. 0 disables the wait.",
    )
    group.add_argument(
        "--dynamo-router-queue-threshold",
        type=float,
        default=None,
        help=(
            "dynamo.frontend --router-queue-threshold. Dynamo turns router queueing off by "
            "default since v1.4.0; set this to re-enable it."
        ),
    )

    group.add_argument(
        "--dynamo-enable-rl",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Pass --enable-rl to dynamo.sglang workers. On SGLang this registers the "
            "/engine/call_tokenizer_manager passthrough route and out-of-band meta_info upload; "
            "the built-in /engine/control/* routes do not need it."
        ),
    )

    return parser
