"""Optional contract checks against the installed Dynamo frontend.

Miles' regular unit tests do not require Dynamo. When a compatible Dynamo is
present in an integration-test image, these checks turn our parser-default pins
and minimum route capability into actual upstream drift detectors.
"""

import argparse
import json
import os
import shlex
import socket
import subprocess
import time
import urllib.request

import pytest

from miles.backends.dynamo_utils.arguments import DYNAMO_SGLANG_GENERATE_MINIMUM_COMMIT, DYNAMO_UPSTREAM_DEFAULTS
from miles.backends.dynamo_utils.dynamo_config import DynamoConfig
from miles.backends.dynamo_utils.frontend_args import (
    compute_dynamo_frontend_env_vars,
    compute_dynamo_frontend_launch_cmd,
)


def test_frontend_defaults_match_installed_dynamo(monkeypatch):
    frontend_args = pytest.importorskip("dynamo.frontend.frontend_args")

    env_vars = {
        "DYN_DISCOVERY_BACKEND",
        "DYN_REQUEST_PLANE",
        "DYN_EVENT_PLANE",
        "DYN_ROUTER_MODE",
        "DYN_ROUTER_KV_EVENTS",
        "DYN_ROUTER_TTL_SECS",
        "DYN_ROUTER_PREDICTED_TTL_SECS",
        "DYN_ROUTER_MIN_INITIAL_WORKERS",
        "DYN_ROUTER_QUEUE_THRESHOLD",
    }
    for name in env_vars:
        monkeypatch.delenv(name, raising=False)

    parser = argparse.ArgumentParser()
    frontend_args.FrontendArgGroup().add_arguments(parser)
    parsed = parser.parse_args([])

    assert parsed.discovery_backend == DYNAMO_UPSTREAM_DEFAULTS["discovery-backend"]
    assert parsed.request_plane == DYNAMO_UPSTREAM_DEFAULTS["request-plane"]
    assert parsed.router_mode == DYNAMO_UPSTREAM_DEFAULTS["router-mode"]
    assert parsed.use_kv_events == DYNAMO_UPSTREAM_DEFAULTS["router-kv-events"]
    assert parsed.router_ttl_secs == DYNAMO_UPSTREAM_DEFAULTS["router-ttl-secs"]
    assert parsed.router_predicted_ttl_secs == DYNAMO_UPSTREAM_DEFAULTS["router-predicted-ttl-secs"]
    assert parsed.min_initial_workers == DYNAMO_UPSTREAM_DEFAULTS["router-min-initial-workers"]
    assert parsed.router_queue_threshold == DYNAMO_UPSTREAM_DEFAULTS["router-queue-threshold"]

    # The parser leaves this unset; Dynamo's runtime resolves the effective
    # event plane to ZMQ. Keep the distinction explicit in the contract test.
    assert parsed.event_plane is None
    assert DYNAMO_UPSTREAM_DEFAULTS["event-plane"] == "zmq"


def test_installed_dynamo_mounts_sglang_generate_route(tmp_path):
    """Start the real frontend so an accepted-but-ignored env var cannot pass.

    Dynamo v1.4.0 accepts the environment variable simply because unknown
    variables are ignored, but has no ``POST /generate`` route. Inspecting
    OpenAPI distinguishes that build from one containing upstream PR #11640
    without requiring a model or worker to be available.
    """
    pytest.importorskip("dynamo.frontend.frontend_args")

    port = _unused_local_port()
    config = DynamoConfig(
        namespace="miles-contract",
        discovery_backend="file",
        file_kv_path=str(tmp_path / "discovery"),
        router_mode="round-robin",
        router_kv_events=True,
        router_ttl_secs=120.0,
        router_predicted_ttl_secs=None,
        router_min_initial_workers=0,
        router_queue_threshold=None,
        enable_rl=True,
    )
    command = compute_dynamo_frontend_launch_cmd(
        config,
        model_id="contract/model",
        host="127.0.0.1",
        port=port,
    )
    env = os.environ | compute_dynamo_frontend_env_vars(config)
    log_path = tmp_path / "dynamo-frontend.log"

    with log_path.open("w+") as log:
        process = subprocess.Popen(
            shlex.split(command),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            _wait_for_listening_frontend(process, port, log)
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/openapi.json",
                timeout=2,
            ) as response:
                openapi = json.load(response)

            assert "/generate" in openapi["paths"], (
                "the installed Dynamo accepted DYN_SGLANG_ENABLE_GENERATE but did not mount "
                f"/generate; build Dynamo at {DYNAMO_SGLANG_GENERATE_MINIMUM_COMMIT} or later"
            )
        finally:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


def _unused_local_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def _wait_for_listening_frontend(process: subprocess.Popen, port: int, log) -> None:
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        if process.poll() is not None:
            log.seek(0)
            pytest.fail(f"Dynamo frontend exited during startup:\n{log.read()}")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                return
        except OSError:
            time.sleep(0.1)

    log.flush()
    log.seek(0)
    pytest.fail(f"Dynamo frontend did not listen within 15 seconds:\n{log.read()}")
