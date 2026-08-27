import argparse

import pytest
from pydantic import ValidationError

from miles.backends.dynamo_utils.arguments import DYNAMO_UPSTREAM_DEFAULTS, add_dynamo_arguments
from miles.backends.dynamo_utils.dynamo_config import (
    DEFAULT_ROUTER_MODE,
    is_dynamo_backend,
    resolve_dynamo_config,
)


def _parse(*argv: str) -> argparse.Namespace:
    parser = add_dynamo_arguments(argparse.ArgumentParser())
    args = parser.parse_args(list(argv))
    # `run_uuid` is supplied by the launcher; the tests pin it so namespaces are deterministic.
    args.run_uuid = "testrun"
    return args


# ------------------------------- backend selection -------------------------------


def test_backend_defaults_to_sglang():
    assert not is_dynamo_backend(_parse())


def test_dynamo_args_parse_without_selecting_the_backend():
    """The flags exist for every run; only --rollout-backend decides which stack launches."""
    args = _parse("--dynamo-router-mode", "kv")
    assert not is_dynamo_backend(args)


def test_resolve_refuses_a_non_dynamo_run():
    with pytest.raises(AssertionError, match="only describes runs that actually launch dynamo"):
        resolve_dynamo_config(_parse())


# ------------------------------- defaults -------------------------------


# Options miles mirrors: its argparse default must equal dynamo's documented default,
# so a miles run behaves the same whether or not the flag is passed.
MIRRORED_DEFAULTS = {
    "dynamo_router_mode": "router-mode",
    "dynamo_router_kv_events": "router-kv-events",
    "dynamo_router_ttl_secs": "router-ttl-secs",
    "dynamo_router_predicted_ttl_secs": "router-predicted-ttl-secs",
    "dynamo_router_min_initial_workers": "router-min-initial-workers",
    "dynamo_router_queue_threshold": "router-queue-threshold",
    "dynamo_enable_rl": "enable-rl",
}


@pytest.mark.parametrize(("dest", "upstream_flag"), sorted(MIRRORED_DEFAULTS.items()))
def test_mirrored_defaults_match_dynamo(dest, upstream_flag):
    """Miles must not invent a default where it mirrors one.

    `--dynamo-router-mode` is the exception: dynamo's default is applied by miles rather
    than by dynamo, because the value is also read on the miles side to decide whether the
    KV flags are rendered at all.
    """
    parsed = getattr(_parse(), dest)
    expected = DYNAMO_UPSTREAM_DEFAULTS[upstream_flag]
    if dest == "dynamo_router_mode":
        assert parsed is None and DEFAULT_ROUTER_MODE == expected
    else:
        assert parsed == expected


def test_options_miles_does_not_mirror_stay_unset():
    """Where miles has no opinion the value stays None, so dynamo applies its own default
    instead of miles picking one for it."""
    config = resolve_dynamo_config(_parse("--rollout-backend", "dynamo"))
    assert config.discovery_backend is None
    assert config.file_kv_path is None


def test_defaults_are_resolved_not_none():
    """Regression: reading args directly yields None, which then reaches subprocess env vars."""
    config = resolve_dynamo_config(_parse("--rollout-backend", "dynamo"))

    assert config.router_mode == DEFAULT_ROUTER_MODE
    assert config.router_kv_events is True
    assert config.router_predicted_ttl_secs is None
    assert config.enable_rl is False
    assert not config.uses_kv_routing

    # Everything miles resolves itself must come out concrete; the rest is deliberately unset.
    for name in (
        "namespace",
        "router_mode",
        "router_kv_events",
        "router_ttl_secs",
        "router_min_initial_workers",
        "enable_rl",
    ):
        value = getattr(config, name)
        assert value is not None, f"{name} resolved to None and would be rendered as a literal 'None'"


def test_namespace_is_scoped_to_the_run():
    config = resolve_dynamo_config(_parse("--rollout-backend", "dynamo"))
    assert config.namespace == "miles-testrun"


def test_namespace_falls_back_when_the_launcher_supplied_no_id():
    args = _parse("--rollout-backend", "dynamo")
    del args.run_uuid
    assert resolve_dynamo_config(args).namespace == "miles"


def test_file_kv_path_is_carried_through_when_requested():
    config = resolve_dynamo_config(
        _parse(
            "--rollout-backend",
            "dynamo",
            "--dynamo-discovery-backend",
            "file",
            "--dynamo-file-kv-path",
            "/shared/dynamo-kv",
        )
    )
    assert config.discovery_backend == "file"
    assert config.file_kv_path == "/shared/dynamo-kv"


# ------------------------------- overrides -------------------------------


def test_explicit_values_win():
    config = resolve_dynamo_config(
        _parse(
            "--rollout-backend",
            "dynamo",
            "--dynamo-namespace",
            "custom-ns",
            "--dynamo-discovery-backend",
            "etcd",
            "--dynamo-router-mode",
            "kv",
            "--dynamo-router-min-initial-workers",
            "4",
            "--dynamo-router-queue-threshold",
            "16.0",
            "--dynamo-enable-rl",
        )
    )
    assert config.namespace == "custom-ns"
    assert config.discovery_backend == "etcd"
    assert config.router_mode == "kv"
    assert config.uses_kv_routing
    assert config.router_min_initial_workers == 4
    assert config.router_queue_threshold == 16.0
    assert config.enable_rl is True


def test_kv_events_can_be_turned_off():
    config = resolve_dynamo_config(_parse("--rollout-backend", "dynamo", "--no-dynamo-router-kv-events"))
    assert config.router_kv_events is False
    assert config.router_ttl_secs == 120.0


def test_config_is_immutable():
    """A later stage must not be able to edit a resolved value out from under another."""
    config = resolve_dynamo_config(_parse("--rollout-backend", "dynamo"))
    with pytest.raises(ValidationError, match="frozen"):
        config.router_mode = "kv"


# ------------------------------- rejected combinations -------------------------------


def test_predicted_ttl_requires_kv_events():
    with pytest.raises(ValueError, match="cannot be combined with --no-dynamo-router-kv-events"):
        resolve_dynamo_config(
            _parse(
                "--rollout-backend",
                "dynamo",
                "--dynamo-router-mode",
                "kv",
                "--dynamo-router-predicted-ttl-secs",
                "30",
                "--no-dynamo-router-kv-events",
            )
        )


def test_predicted_ttl_requires_kv_router_mode():
    with pytest.raises(ValueError, match="only applies to KV routing"):
        resolve_dynamo_config(_parse("--rollout-backend", "dynamo", "--dynamo-router-predicted-ttl-secs", "30"))


def test_file_kv_path_rejected_for_other_discovery_backends():
    with pytest.raises(ValueError, match="only backs the 'file' discovery backend"):
        resolve_dynamo_config(
            _parse(
                "--rollout-backend",
                "dynamo",
                "--dynamo-discovery-backend",
                "etcd",
                "--dynamo-file-kv-path",
                "/tmp/whatever",
            )
        )
