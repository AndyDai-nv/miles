#!/usr/bin/env python3
"""Fail fast when an image does not satisfy the Miles × Dynamo contract."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path

if __package__:
    from .build_wheels import validate_commit_sha
else:
    from build_wheels import validate_commit_sha

REQUIRED_SGLANG_SERVER_ARGS = (
    "base_gpu_id",
    "dist_init_addr",
    "dp_size",
    "enable_memory_saver",
    "engine_info_bootstrap_port",
    "host",
    "model_path",
    "node_rank",
    "pp_size",
    "port",
    "tp_size",
)

# These weight-update session objects and methods are Miles fork contracts;
# checking them is what distinguishes sglang-miles from a merely compatible
# stock version number.
REQUIRED_SGLANG_IO_STRUCTS = (
    "BeginWeightUpdateReqInput",
    "EndWeightUpdateReqInput",
    "UpdateWeightFromDiskReqInput",
    "UpdateWeightsFromDistributedReqInput",
    "UpdateWeightsFromIPCReqInput",
    "UpdateWeightsFromTensorReqInput",
    "UpdateWeightVersionReqInput",
)
REQUIRED_TOKENIZER_MANAGER_METHODS = (
    "begin_weight_update",
    "check_weights",
    "continue_generation",
    "destroy_weights_update_group",
    "end_weight_update",
    "flush_cache",
    "init_weights_update_group",
    "pause_generation",
    "update_weights_from_disk",
    "update_weights_from_distributed",
    "update_weights_from_ipc",
    "update_weights_from_tensor",
)
REQUIRED_DYNAMO_CONTROL_METHODS = (
    "release_memory_occupation",
    "resume_memory_occupation",
    "start_profile",
    "stop_profile",
    "update_weight_version",
    "update_weights_from_disk",
    "update_weights_from_distributed",
    "update_weights_from_ipc",
    "update_weights_from_tensor",
)


class ContractError(RuntimeError):
    """The built image cannot support Miles' Dynamo rollout control path."""


def missing_attributes(value: object, names: Iterable[str]) -> list[str]:
    return sorted(name for name in names if not hasattr(value, name))


def read_build_manifest(path: Path, *, expected_dynamo_commit: str) -> dict:
    expected = validate_commit_sha(expected_dynamo_commit, name="expected_dynamo_commit")
    try:
        manifest = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read Dynamo wheel manifest {path}: {exc}") from exc

    if manifest.get("schema_version") != 1:
        raise ContractError("Dynamo wheel manifest schema_version must be 1")
    actual = manifest.get("dynamo_commit")
    if actual != expected:
        raise ContractError(f"Dynamo wheels were built from {actual!r}, expected {expected!r}")
    wheels = manifest.get("wheels")
    if not isinstance(wheels, list) or not wheels:
        raise ContractError("Dynamo wheel manifest must contain a non-empty wheels list")
    invalid_names = [
        name for name in wheels if not isinstance(name, str) or Path(name).name != name or not name.endswith(".whl")
    ]
    if invalid_names:
        raise ContractError(f"Dynamo wheel manifest contains invalid wheel names: {invalid_names!r}")
    missing_files = [name for name in wheels if not (path.parent / name).is_file()]
    if missing_files:
        raise ContractError("Dynamo wheel manifest names files absent from its directory: " + ", ".join(missing_files))
    return manifest


def verify_sglang_checkout(source_dir: Path, *, expected_commit: str) -> str:
    expected = validate_commit_sha(expected_commit, name="expected_sglang_commit")
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=source_dir,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ContractError(f"cannot inspect SGLang checkout at {source_dir}: {exc}") from exc
    actual = result.stdout.strip()
    if actual != expected:
        raise ContractError(f"installed SGLang checkout is {actual}, expected {expected}")
    return actual


def _require_attributes(label: str, value: object, names: Iterable[str]) -> None:
    missing = missing_attributes(value, names)
    if missing:
        raise ContractError(f"{label} is missing required Miles contracts: {', '.join(missing)}")


def probe_installed_contract(*, sglang_source_dir: Path | None = None) -> dict[str, str]:
    """Import and inspect the exact Python objects used by future adapters."""
    try:
        from dynamo.common.configuration.groups.runtime_args import DynamoRuntimeArgGroup
        from dynamo.frontend.frontend_args import FrontendArgGroup
        from dynamo.sglang.backend_args import DynamoSGLangArgGroup
        from dynamo.sglang.request_handlers.handler_base import BaseWorkerHandler, RLMixin
        from sglang.srt.managers import io_struct
        from sglang.srt.managers.tokenizer_manager import TokenizerManager
        from sglang.srt.server_args import ServerArgs
    except ImportError as exc:
        raise ContractError(f"cannot import the Dynamo/SGLang integration surface: {exc}") from exc

    if sglang_source_dir is not None:
        import sglang

        installed_module = Path(sglang.__file__).resolve()
        expected_module_root = (sglang_source_dir / "python").resolve()
        if not installed_module.is_relative_to(expected_module_root):
            raise ContractError(f"sglang imports from {installed_module}, not pinned source {expected_module_root}")

    import argparse as _argparse
    import dataclasses as _dataclasses

    frontend_parser = _argparse.ArgumentParser()
    FrontendArgGroup().add_arguments(frontend_parser)
    frontend = frontend_parser.parse_args(
        [
            "--namespace",
            "miles-contract-model-00000000",
            "--http-host",
            "127.0.0.1",
            "--http-port",
            "30000",
            "--discovery-backend",
            "file",
            "--request-plane",
            "tcp",
            "--event-plane",
            "zmq",
            "--router-mode",
            "round-robin",
        ]
    )
    if frontend.namespace != "miles-contract-model-00000000":
        raise ContractError("Dynamo frontend parser did not preserve the requested namespace")

    worker_parser = _argparse.ArgumentParser()
    DynamoRuntimeArgGroup().add_arguments(worker_parser)
    DynamoSGLangArgGroup().add_arguments(worker_parser)
    worker = worker_parser.parse_args(
        [
            "--namespace",
            "miles-contract-model-00000000",
            "--discovery-backend",
            "file",
            "--enable-rl",
        ]
    )
    if not worker.enable_rl:
        raise ContractError("Dynamo SGLang parser did not enable its RL route surface")

    server_arg_names = {field.name for field in _dataclasses.fields(ServerArgs)}
    missing_server_args = sorted(set(REQUIRED_SGLANG_SERVER_ARGS) - server_arg_names)
    if missing_server_args:
        raise ContractError("SGLang ServerArgs is missing fields required by Miles: " + ", ".join(missing_server_args))

    _require_attributes("sglang.srt.managers.io_struct", io_struct, REQUIRED_SGLANG_IO_STRUCTS)
    _require_attributes("SGLang TokenizerManager", TokenizerManager, REQUIRED_TOKENIZER_MANAGER_METHODS)
    _require_attributes("Dynamo BaseWorkerHandler", BaseWorkerHandler, REQUIRED_DYNAMO_CONTROL_METHODS)
    _require_attributes("Dynamo RLMixin", RLMixin, ("call_tokenizer_manager",))

    return {
        "ai_dynamo_version": importlib.metadata.version("ai-dynamo"),
        "ai_dynamo_runtime_version": importlib.metadata.version("ai-dynamo-runtime"),
        "sglang_version": importlib.metadata.version("sglang"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel-manifest", type=Path, required=True)
    parser.add_argument("--expected-dynamo-commit", required=True)
    parser.add_argument("--sglang-source-dir", type=Path, required=True)
    parser.add_argument("--expected-sglang-commit", required=True)
    args = parser.parse_args()

    try:
        manifest = read_build_manifest(
            args.wheel_manifest,
            expected_dynamo_commit=args.expected_dynamo_commit,
        )
        sglang_commit = verify_sglang_checkout(
            args.sglang_source_dir,
            expected_commit=args.expected_sglang_commit,
        )
        versions = probe_installed_contract(sglang_source_dir=args.sglang_source_dir)
    except (ContractError, ValueError) as exc:
        print(f"Dynamo/SGLang compatibility check failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    print(
        json.dumps(
            {
                "status": "ok",
                "dynamo_commit": manifest["dynamo_commit"],
                "sglang_commit": sglang_commit,
                **versions,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
