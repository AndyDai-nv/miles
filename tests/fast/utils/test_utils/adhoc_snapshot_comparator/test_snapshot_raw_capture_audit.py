import hashlib
import json
from pathlib import Path

import pytest
from tests.fast.utils.test_utils.adhoc_snapshot_comparator.capture_input_fixtures import _CaptureInputs, capture_inputs
from typer.testing import CliRunner

from miles.utils.test_utils.adhoc_snapshot_comparator.snapshot_capture_recovery import app
from miles.utils.test_utils.adhoc_snapshot_comparator.snapshot_provenance_evidence import (
    _build_provenance,
    _load_provenance,
)

__all__ = ["capture_inputs"]


class TestRawCaptureRecovery:
    @pytest.mark.parametrize(
        "mode", ["matched", "changed", "incomplete", "unknown_schema", "missing_case", "historical"]
    )
    def test_recovery_retains_raw_identity_and_cannot_promote_unmatched_or_incomplete_capture(
        self, tmp_path: Path, capture_inputs: _CaptureInputs, mode: str
    ) -> None:
        """Only an exact complete capture passes, while every failure retains raw-file identity and its reason."""
        inputs = capture_inputs
        argv = [
            "--raw-directory",
            str(inputs.raw),
            "--after",
            str(inputs.after),
            "--output",
            str(tmp_path / "report.json"),
        ]
        if mode == "changed":
            inputs.after_case.write_text(inputs.after_case.read_text().replace('"seed": 7', '"seed": 8'))
        elif mode == "incomplete":
            (inputs.raw / "attempt.json").write_text(json.dumps({"test": "tests/e2e/case.py", "completed": False}))
        elif mode == "unknown_schema":
            inputs.record.write_text('{"unknown": true}')
        elif mode == "missing_case":
            inputs.after_case.rename(inputs.after_case.with_name("other.yaml"))
        elif mode == "historical":
            replay = tmp_path / "replay.json"
            replay.write_text(
                json.dumps(
                    {
                        "source_sha256": {"normalizer.py": "historical-source"},
                        "attempts": [
                            {
                                "attempt": str(inputs.raw / "attempt.json"),
                                "raw_sha256": {
                                    str(inputs.record): hashlib.sha256(inputs.record.read_bytes()).hexdigest()
                                },
                                "expanded": inputs.expanded,
                            }
                        ],
                    }
                )
            )
            argv.extend(["--historical-replay", str(replay)])

        result = CliRunner().invoke(app, argv)

        assert result.exit_code == (0 if mode in {"matched", "historical"} else 1), result.exception
        report = json.loads((tmp_path / "report.json").read_text())
        [attempt] = report["attempts"]
        assert attempt["raw_files"] == [
            {"path": str(inputs.record), "sha256": hashlib.sha256(inputs.record.read_bytes()).hexdigest()}
        ]
        verdicts = {
            "matched": "exact-historical-after-recovered",
            "historical": "exact-historical-after-recovered",
            "changed": "historical-after-differs",
            "incomplete": "incomplete-attempt-retained",
            "unknown_schema": "historical-schema-replay-required",
            "missing_case": "no-paired-recorded-case",
        }
        assert attempt["verdict"] == verdicts[mode]
        if mode == "changed":
            [difference] = attempt["differences"]
            assert difference["path"][-1] == "seed"
            assert difference["before"] == {"present": True, "value": 7}
            assert difference["after"] == {"present": True, "value": 8}
        if mode == "historical":
            assert attempt["historical_normalizer_sources"] == {"normalizer.py": "historical-source"}


class TestRawProvenanceBuild:
    @pytest.mark.parametrize("complete", [False, True])
    @pytest.mark.parametrize("stage_matches", [False, True])
    def test_only_complete_exact_stage_produces_a_loadable_credential_witness(
        self, tmp_path: Path, capture_inputs: _CaptureInputs, complete: bool, stage_matches: bool
    ) -> None:
        """A raw credential origin cannot certify another stage or an unfinished attempt."""
        inputs = capture_inputs
        (inputs.raw / "attempt.json").write_text(json.dumps({"test": "tests/e2e/case.py", "completed": complete}))
        if not stage_matches:
            inputs.after_case.write_text(inputs.after_case.read_text().replace('"seed": 7', '"seed": 8'))
        output = tmp_path / "proof.json"

        _build_provenance(before=inputs.before, after=inputs.after, raw_directory=inputs.raw, output=output)

        report = json.loads(output.read_text())
        assert len(report["proof"]) == int(complete and stage_matches)
        assert report["ambiguous_or_changed_digests"] == []
        assert report["rejected_attempts"] == []
        assert len(_load_provenance(path=output, after_directory=inputs.after)) == int(complete and stage_matches)
        if complete and stage_matches:
            [row] = report["proof"]
            assert row["before"] == {"present": True, "value": "redacted-sha256:original"}
            assert row["after"] == {"present": True, "value": "$WANDB_KEY_environment"}
            assert row["witness"]["record"] == str(inputs.record)
            assert row["witness"]["sha256"] == hashlib.sha256(inputs.record.read_bytes()).hexdigest()
