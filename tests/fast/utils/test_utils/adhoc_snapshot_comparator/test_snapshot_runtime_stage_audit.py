import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from miles.utils.test_utils.adhoc_snapshot_comparator.snapshot_runtime_evidence import app


class TestRuntimeStageAudit:
    @pytest.mark.parametrize("changed_checkpoint", [False, True])
    def test_every_runtime_stage_keeps_its_own_trainer_resource_input(
        self, tmp_path: Path, changed_checkpoint: bool
    ) -> None:
        """A matching base capture cannot conceal a changed checkpoint-stage startup count."""
        before, after = tmp_path / "before", tmp_path / "after"
        for root in (before, after):
            (root / "runtime_config").mkdir(parents=True)
        args = {
            "trainer_id": "policy-critic",
            "indep_dp": True,
            "critic_num_nodes": 3,
            "critic_num_gpus_per_node": 8,
            "tensor_model_parallel_size": 2,
            "pipeline_model_parallel_size": 3,
            "context_parallel_size": 2,
        }
        process = "run/critic"
        (before / "runtime_config/case.yaml").write_text(
            json.dumps(
                {
                    "processes": {
                        process: {
                            "ranks": [0],
                            "base": {"args": args},
                            "diffs": {
                                "checkpoint_load-0000": {"set": {"/args/critic_num_gpus_per_node": 16}, "remove": []}
                            },
                        }
                    }
                }
            )
        )
        observed = 5 if changed_checkpoint else 4
        (after / "runtime_config/case.yaml").write_text(
            json.dumps(
                {
                    "processes": {
                        process: {
                            "ranks": [0],
                            "base": {"args": {"trainer_init_expected_num_cells": 2, "trainer_role": "critic"}},
                            "diffs": {
                                "checkpoint_load-0000": {
                                    "set": {"/args/trainer_init_expected_num_cells": observed},
                                    "remove": [],
                                }
                            },
                        }
                    }
                }
            )
        )
        output = tmp_path / "report.json"

        result = CliRunner().invoke(app, ["--before", str(before), "--after", str(after), "--output", str(output)])

        assert result.exit_code == int(changed_checkpoint), result.exception
        report = json.loads(output.read_text())
        assert report["counts"] == {
            "inputs": 2,
            "skipped_non_trainer": 0,
            "targets_not_captured": 0,
            "cells_match": 1 if changed_checkpoint else 2,
            "roles_match": 2,
        }
        stages = {row["path"][3]: row for row in report["records"]}
        assert stages["process_config-0000"]["expected_cells"] == 2
        checkpoint = stages["checkpoint_load-0000"]
        assert checkpoint["expected_cells"] == 4
        assert checkpoint["observed_cells"] == observed
        assert checkpoint["inputs"]["critic_num_gpus_per_node"] == 16
