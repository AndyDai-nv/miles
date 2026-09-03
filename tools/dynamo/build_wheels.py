#!/usr/bin/env python3
"""Build Dynamo wheels and record the exact source commit they came from."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path

MANIFEST_NAME = "dynamo-build-manifest.json"
_FULL_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")


def validate_commit_sha(value: str, *, name: str = "commit") -> str:
    normalized = value.lower()
    if _FULL_GIT_SHA.fullmatch(normalized) is None:
        raise ValueError(f"{name} must be a full 40-character Git SHA; got {value!r}")
    return normalized


def make_build_manifest(*, dynamo_commit: str, wheel_files: Sequence[str]) -> dict:
    wheels = sorted(wheel_files)
    if not any(name.startswith("ai_dynamo-") and name.endswith(".whl") for name in wheels):
        raise ValueError("wheel output does not contain an ai_dynamo wheel")
    if not any(name.startswith("ai_dynamo_runtime-") and name.endswith(".whl") for name in wheels):
        raise ValueError("wheel output does not contain an ai_dynamo_runtime wheel")
    return {
        "schema_version": 1,
        "dynamo_commit": validate_commit_sha(dynamo_commit, name="dynamo_commit"),
        "wheels": wheels,
    }


def _run_output(cmd: Sequence[str], *, cwd: Path) -> str:
    result = subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True)
    return result.stdout.strip()


def build_dynamo_wheels(
    *,
    dynamo_repo: Path,
    dynamo_commit: str,
    output_dir: Path,
    runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> Path:
    """Build both Dynamo Python wheels from a clean checkout at one exact commit."""
    repo = dynamo_repo.resolve()
    if not (repo / ".git").exists():
        raise ValueError(f"--dynamo-repo is not a Git checkout: {repo}")

    expected_commit = validate_commit_sha(dynamo_commit, name="dynamo_commit")
    actual_commit = _run_output(["git", "rev-parse", "HEAD"], cwd=repo)
    if actual_commit != expected_commit:
        raise ValueError(f"Dynamo checkout is at {actual_commit}, but --dynamo-commit is {expected_commit}")
    if _run_output(["git", "status", "--porcelain"], cwd=repo):
        raise ValueError("Dynamo checkout must be clean before building attributable wheels")

    output = output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError(f"wheel output directory must be empty: {output}")

    runner(
        ["uv", "build", "--wheel", "--out-dir", str(output)],
        cwd=repo,
        check=True,
    )
    runner(
        [
            "maturin",
            "build",
            "--release",
            "--features",
            "kv-indexer",
            "--out",
            str(output),
        ],
        cwd=repo / "lib" / "bindings" / "python",
        check=True,
    )

    manifest = make_build_manifest(
        dynamo_commit=actual_commit,
        wheel_files=[path.name for path in output.glob("*.whl")],
    )
    manifest_path = output / MANIFEST_NAME
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dynamo-repo", type=Path, required=True)
    parser.add_argument("--dynamo-commit", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    manifest_path = build_dynamo_wheels(
        dynamo_repo=args.dynamo_repo,
        dynamo_commit=args.dynamo_commit,
        output_dir=args.output_dir,
    )
    print(manifest_path)


if __name__ == "__main__":
    main()
