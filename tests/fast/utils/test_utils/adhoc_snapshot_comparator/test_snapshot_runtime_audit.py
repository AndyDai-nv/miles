import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from miles.utils.test_utils.adhoc_snapshot_comparator.snapshot_runtime_evidence import app


class TestSnapshotRuntimeAudit:
    @pytest.mark.parametrize(
        "observed_cells,observed_role,passes", [(2, "critic", True), (3, "critic", False), (2, "actor", False)]
    )
    def test_runtime_replay_reports_and_rejects_changed_startup_counts_or_roles(
        self, tmp_path: Path, observed_cells: int, observed_role: str, passes: bool
    ) -> None:
        """The CLI retains exact mismatches and fails when a trainer's resource count or role changes."""
        before, after = tmp_path / "before", tmp_path / "after"
        for directory in (before, after):
            (directory / "parsed_config").mkdir(parents=True)
        args = {
            "trainer_id": "policy-critic",
            "indep_dp": True,
            "critic_num_nodes": 3,
            "critic_num_gpus_per_node": 8,
            "tensor_model_parallel_size": 2,
            "pipeline_model_parallel_size": 3,
            "context_parallel_size": 2,
        }
        (before / "parsed_config/case.yaml").write_text(json.dumps({"config": args}))
        (after / "parsed_config/case.yaml").write_text(
            json.dumps({"config": {"trainer_init_expected_num_cells": observed_cells, "trainer_role": observed_role}})
        )
        output = tmp_path / "result.json"

        result = CliRunner().invoke(app, ["--before", str(before), "--after", str(after), "--output", str(output)])

        assert result.exit_code == (0 if passes else 1), result.exception
        report = json.loads(output.read_text())
        assert report["counts"] == {
            "inputs": 1,
            "skipped_non_trainer": 0,
            "targets_not_captured": 0,
            "cells_match": int(observed_cells == 2),
            "roles_match": int(observed_role == "critic"),
        }
        record = report["records"][0]
        assert record["expected_cells"] == 2
        assert record["observed_cells"] == observed_cells
        assert record["role"] == "critic"
        assert record["inputs"] == {name: value for name, value in args.items() if name != "trainer_id"}

    @pytest.mark.parametrize("observed_cells", [True, "2"])
    def test_non_integer_startup_counts_cannot_pass_through_python_equality(
        self, tmp_path: Path, observed_cells: bool | str
    ) -> None:
        """Boolean and string startup counts cannot be accepted as captured integer resource counts."""
        before, after = tmp_path / "before", tmp_path / "after"
        for directory in (before, after):
            (directory / "parsed_config").mkdir(parents=True)
        (before / "parsed_config/case.yaml").write_text(
            json.dumps({"config": {"trainer_id": "actor", "indep_dp": False}})
        )
        (after / "parsed_config/case.yaml").write_text(
            json.dumps({"config": {"trainer_init_expected_num_cells": observed_cells}})
        )
        output = tmp_path / "result.json"

        result = CliRunner().invoke(app, ["--before", str(before), "--after", str(after), "--output", str(output)])

        assert isinstance(result.exception, AssertionError)
        assert not output.exists()
