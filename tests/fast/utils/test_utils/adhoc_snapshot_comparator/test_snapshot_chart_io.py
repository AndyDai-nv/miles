import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from tests.fast.utils.test_utils.adhoc_snapshot_comparator.chart_observation_fixtures import chart_observations
from typer.testing import CliRunner

from miles.utils.test_utils.adhoc_snapshot_comparator.snapshot_chart_comparison import app
from miles.utils.test_utils.adhoc_snapshot_comparator.snapshot_chart_payload_evidence import _payloads

__all__ = ["chart_observations"]


class TestChartComparisonPreparation:
    def test_paired_observations_preserve_each_worker_input_and_rank_scope(
        self, tmp_path: Path, chart_observations: tuple[dict[str, Any], dict[str, Any]]
    ) -> None:
        """Preparation preserves each side's exact parser and worker inputs without fabricating execution."""
        before, after = chart_observations
        left, right, output = tmp_path / "before.json", tmp_path / "after.json", tmp_path / "prepared"
        left.write_text(json.dumps(before))
        right.write_text(json.dumps(after))

        result = CliRunner().invoke(app, ["--before", str(left), "--after", str(right), "--output", str(output)])

        assert result.exit_code == 0, result.exception
        for side, record in (("before", before), ("after", after)):
            root = output / side
            assert yaml.safe_load((root / "parsed_config/chart_fixture.yaml").read_text()) == {
                "config": record["parsed"]
            }
            processes = yaml.safe_load((root / "runtime_config/chart_fixture.yaml").read_text())["processes"]
            assert len(processes) == 4
            for owner, old_name in (
                ("rollout_executor", "rollout-executor"),
                ("inference_controller", "inference-controller"),
                ("trainer_controller_actor", "trainer-controller-actor"),
                ("actor_cell00000", "trainer-engine-actor"),
            ):
                process = processes["chart-fixture/run-0000/all/default/" + owner]
                assert process["ranks"] == ([0] if owner == "actor_cell00000" else [])
                assert process["base"] == {"args": record["workers"][old_name]["args"]}
                assert process["diffs"] == {}

    @pytest.mark.parametrize("mismatch", ["scenario_argv", "worker_argv", "legacy", "worker_class", "roster"])
    def test_unpaired_inputs_or_worker_identity_fail_before_writing_any_prepared_capture(
        self, tmp_path: Path, chart_observations: tuple[dict[str, Any], dict[str, Any]], mismatch: str
    ) -> None:
        """Different launch inputs or worker identities cannot masquerade as a paired chart observation."""
        before, after = chart_observations
        if mismatch in {"scenario_argv", "worker_argv"}:
            after[mismatch] = ["changed"]
        elif mismatch == "legacy":
            after["legacy"] = True
        elif mismatch == "worker_class":
            after["workers"]["rollout-executor"]["worker_class"] = "worker.Different"
        else:
            del after["workers"]["rollout-executor"]
        left, right, output = tmp_path / "before.json", tmp_path / "after.json", tmp_path / "prepared"
        left.write_text(json.dumps(before))
        right.write_text(json.dumps(after))

        result = CliRunner().invoke(app, ["--before", str(left), "--after", str(right), "--output", str(output)])

        assert isinstance(result.exception, AssertionError)
        assert not output.exists()


class TestChartPayloadExtraction:
    def test_nested_manifests_keep_exact_shell_and_argv_payloads_and_ignore_unrelated_configs(self) -> None:
        """Manifest extraction finds only served-worker configs while preserving JSON whitespace and quoting."""
        payload = '{"text": "value with spaces"}'
        manifest = {
            "items": [
                ["python", "-m", "miles.utils.workers.serving.serve", "--config", payload],
                'python -m miles.utils.workers.serving.serve --config \'{"text": "value with spaces"}\'',
                ["other", "--config", "ignored"],
            ]
        }

        found = list(_payloads(value=manifest, path=["manifest.yaml"]))

        assert found == [(["manifest.yaml", "items", "0"], payload), (["manifest.yaml", "items", "1"], payload)]
