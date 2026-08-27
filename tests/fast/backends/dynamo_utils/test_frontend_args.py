import argparse

from miles.backends.dynamo_utils.arguments import add_dynamo_arguments
from miles.backends.dynamo_utils.dynamo_config import resolve_dynamo_config
from miles.backends.dynamo_utils.frontend_args import (
    compute_dynamo_frontend_args,
    compute_dynamo_frontend_env_vars,
    compute_dynamo_frontend_launch_cmd,
    dynamo_frontend_args_to_argv,
)

HOST = "10.0.0.7"
PORT = 30000


def _config(*argv: str):
    parser = add_dynamo_arguments(argparse.ArgumentParser())
    args = parser.parse_args(["--rollout-backend", "dynamo", *argv])
    args.run_uuid = "testrun"
    return resolve_dynamo_config(args)


def _argv(*argv: str) -> list[str]:
    return dynamo_frontend_args_to_argv(compute_dynamo_frontend_args(_config(*argv), host=HOST, port=PORT))


# ------------------------------- rendered argv -------------------------------


def test_default_argv_is_exact():
    """Pinned in full: the command line is what a run reproduces from, so a silent
    change to any flag or its ordering should fail here rather than in a cluster."""
    assert _argv() == [
        "--http-host",
        "10.0.0.7",
        "--http-port",
        "30000",
        "--namespace",
        "miles-testrun",
        "--request-plane",
        "tcp",
        "--event-plane",
        "zmq",
        "--router-mode",
        "round-robin",
    ]


def test_kv_routing_argv_is_exact():
    assert _argv("--dynamo-router-mode", "kv", "--dynamo-router-predicted-ttl-secs", "45") == [
        "--http-host",
        "10.0.0.7",
        "--http-port",
        "30000",
        "--namespace",
        "miles-testrun",
        "--request-plane",
        "tcp",
        "--event-plane",
        "zmq",
        "--router-mode",
        "kv",
        "--router-kv-events",
        "--router-predicted-ttl-secs",
        "45.0",
    ]


def test_kv_routing_without_events_renders_the_negation_and_a_ttl():
    argv = _argv("--dynamo-router-mode", "kv", "--no-dynamo-router-kv-events")
    assert "--no-router-kv-events" in argv
    assert "--router-kv-events" not in argv
    assert argv[argv.index("--router-ttl-secs") + 1] == "120.0"


def test_non_kv_modes_render_no_kv_flags():
    """The other router modes keep no cache index, so a KV flag there is noise at best."""
    argv = _argv("--dynamo-router-mode", "power-of-two")
    assert not [flag for flag in argv if "kv-events" in flag or "ttl-secs" in flag]


def test_optional_flags_are_absent_by_default():
    argv = _argv()
    assert "--router-min-initial-workers" not in argv
    assert "--router-queue-threshold" not in argv


def test_optional_flags_render_when_set():
    argv = _argv("--dynamo-router-min-initial-workers", "4", "--dynamo-router-queue-threshold", "16")
    assert argv[argv.index("--router-min-initial-workers") + 1] == "4"
    assert argv[argv.index("--router-queue-threshold") + 1] == "16.0"


# ------------------------------- renderer semantics -------------------------------


def test_none_is_dropped_rather_than_stringified():
    """Regression: an unresolved default reaching argv as the literal 'None' is the
    failure mode the config object exists to prevent; the renderer refuses it too."""
    assert dynamo_frontend_args_to_argv({"a": None, "b": 1}) == ["--b", "1"]


def test_booleans_render_as_presence_or_negation():
    assert dynamo_frontend_args_to_argv({"router-kv-events": True}) == ["--router-kv-events"]
    assert dynamo_frontend_args_to_argv({"router-kv-events": False}) == ["--no-router-kv-events"]


def test_insertion_order_is_preserved():
    assert dynamo_frontend_args_to_argv({"z": 1, "a": 2}) == ["--z", "1", "--a", "2"]


# ------------------------------- launch command -------------------------------


def test_launch_command_is_shell_safe_and_names_the_module():
    cmd = compute_dynamo_frontend_launch_cmd(_config(), host=HOST, port=PORT)
    assert "-m dynamo.frontend" in cmd
    assert "--http-port 30000" in cmd


def test_launch_command_is_deterministic():
    """It has to be: an offline manifest and the live launch must agree."""
    config = _config("--dynamo-router-mode", "kv")
    assert compute_dynamo_frontend_launch_cmd(config, host=HOST, port=PORT) == compute_dynamo_frontend_launch_cmd(
        config, host=HOST, port=PORT
    )


def test_launch_command_quotes_a_hostile_namespace():
    cmd = compute_dynamo_frontend_launch_cmd(_config("--dynamo-namespace", "a b;rm -rf /"), host=HOST, port=PORT)
    assert "'a b;rm -rf /'" in cmd


# ------------------------------- environment -------------------------------


def test_env_enables_the_sglang_generate_route():
    assert compute_dynamo_frontend_env_vars(_config())["DYN_SGLANG_ENABLE_GENERATE"] == "1"


def test_discovery_backend_is_rendered_only_when_the_run_asked_for_one():
    """Unset means dynamo picks; miles must not smuggle in a choice of its own."""
    assert "--discovery-backend" not in _argv()
    argv = _argv("--dynamo-discovery-backend", "file")
    assert argv[argv.index("--discovery-backend") + 1] == "file"


def test_env_carries_the_file_kv_path_only_when_one_was_given():
    assert "DYN_FILE_KV" not in compute_dynamo_frontend_env_vars(_config())
    env = compute_dynamo_frontend_env_vars(
        _config("--dynamo-discovery-backend", "file", "--dynamo-file-kv-path", "/shared/kv")
    )
    assert env["DYN_FILE_KV"] == "/shared/kv"
