import json
import subprocess
from types import SimpleNamespace

import pytest

from tools.dynamo import build_wheels
from tools.dynamo.build_wheels import make_build_manifest, validate_commit_sha
from tools.dynamo.check_contract import ContractError, missing_attributes, read_build_manifest, verify_sglang_checkout

DYNAMO_SHA = "1" * 40
SGLANG_SHA = "2" * 40


def test_commit_pins_must_be_full_git_shas():
    assert validate_commit_sha(DYNAMO_SHA.upper()) == DYNAMO_SHA
    for invalid in ("main", "abc123", "g" * 40, ""):
        with pytest.raises(ValueError, match="full 40-character Git SHA"):
            validate_commit_sha(invalid)


def test_build_manifest_is_deterministic_and_contains_both_wheels():
    manifest = make_build_manifest(
        dynamo_commit=DYNAMO_SHA,
        wheel_files=[
            "ai_dynamo_runtime-1.4.0-cp311.whl",
            "ai_dynamo-1.4.0-py3-none-any.whl",
        ],
    )
    assert manifest == {
        "schema_version": 1,
        "dynamo_commit": DYNAMO_SHA,
        "wheels": [
            "ai_dynamo-1.4.0-py3-none-any.whl",
            "ai_dynamo_runtime-1.4.0-cp311.whl",
        ],
    }


@pytest.mark.parametrize(
    "wheels,missing",
    [
        (["ai_dynamo_runtime-1.whl"], "ai_dynamo wheel"),
        (["ai_dynamo-1.whl"], "ai_dynamo_runtime wheel"),
    ],
)
def test_build_manifest_requires_both_distributions(wheels, missing):
    with pytest.raises(ValueError, match=missing):
        make_build_manifest(dynamo_commit=DYNAMO_SHA, wheel_files=wheels)


def test_wheel_builder_requires_exact_clean_checkout_and_writes_manifest(tmp_path, monkeypatch):
    repo = tmp_path / "dynamo"
    (repo / ".git").mkdir(parents=True)
    (repo / "lib" / "bindings" / "python").mkdir(parents=True)
    output = tmp_path / "wheels"

    def fake_git_output(cmd, *, cwd):
        return DYNAMO_SHA if cmd[-1] == "HEAD" else ""

    commands = []

    def fake_runner(cmd, *, cwd, check):
        commands.append((cmd, cwd, check))
        wheel = "ai_dynamo-1.whl" if cmd[0] == "uv" else "ai_dynamo_runtime-1.whl"
        (output / wheel).touch()
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(build_wheels, "_run_output", fake_git_output)
    manifest_path = build_wheels.build_dynamo_wheels(
        dynamo_repo=repo,
        dynamo_commit=DYNAMO_SHA,
        output_dir=output,
        runner=fake_runner,
    )

    assert [cmd[0][0] for cmd in commands] == ["uv", "maturin"]
    assert json.loads(manifest_path.read_text())["dynamo_commit"] == DYNAMO_SHA


def test_read_build_manifest_rejects_a_different_source_commit(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dynamo_commit": "3" * 40,
                "wheels": ["ai_dynamo-1.whl", "ai_dynamo_runtime-1.whl"],
            }
        )
    )
    with pytest.raises(ContractError, match="expected"):
        read_build_manifest(path, expected_dynamo_commit=DYNAMO_SHA)


def test_read_build_manifest_accepts_the_attributed_wheels(tmp_path):
    path = tmp_path / "manifest.json"
    expected = make_build_manifest(
        dynamo_commit=DYNAMO_SHA,
        wheel_files=["ai_dynamo-1.whl", "ai_dynamo_runtime-1.whl"],
    )
    for wheel in expected["wheels"]:
        (tmp_path / wheel).touch()
    path.write_text(json.dumps(expected))
    assert read_build_manifest(path, expected_dynamo_commit=DYNAMO_SHA) == expected


def test_read_build_manifest_rejects_missing_or_unsafe_wheel_paths(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dynamo_commit": DYNAMO_SHA,
                "wheels": ["../other/ai_dynamo-1.whl"],
            }
        )
    )
    with pytest.raises(ContractError, match="invalid wheel names"):
        read_build_manifest(path, expected_dynamo_commit=DYNAMO_SHA)

    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dynamo_commit": DYNAMO_SHA,
                "wheels": ["ai_dynamo-1.whl"],
            }
        )
    )
    with pytest.raises(ContractError, match="files absent"):
        read_build_manifest(path, expected_dynamo_commit=DYNAMO_SHA)


def test_missing_attributes_reports_the_contract_gap():
    value = SimpleNamespace(begin_weight_update=object(), end_weight_update=object())
    assert missing_attributes(
        value,
        ("begin_weight_update", "check_weights", "end_weight_update"),
    ) == ["check_weights"]


def test_verify_sglang_checkout_checks_the_exact_commit(tmp_path, monkeypatch):
    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args[0], 0, stdout=f"{SGLANG_SHA}\n")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert verify_sglang_checkout(tmp_path, expected_commit=SGLANG_SHA) == SGLANG_SHA


def test_verify_sglang_checkout_rejects_version_skew(tmp_path, monkeypatch):
    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args[0], 0, stdout=f"{'4' * 40}\n")

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(ContractError, match="installed SGLang checkout"):
        verify_sglang_checkout(tmp_path, expected_commit=SGLANG_SHA)
