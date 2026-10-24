import argparse
import json
from pathlib import Path

import pytest
from sglang_router.launch_router import RouterArgs
from typer.testing import CliRunner

from miles.utils.test_utils.adhoc_snapshot_comparator.snapshot_chart_evidence import app as chart_app
from miles.utils.test_utils.adhoc_snapshot_comparator.snapshot_megatron_evidence import app as megatron_app
from miles.utils.test_utils.adhoc_snapshot_comparator.snapshot_router_environment_evidence import app as router_app
from miles.utils.test_utils.adhoc_snapshot_comparator.snapshot_sglang_evidence import app as sglang_app
from miles.utils.test_utils.snapshot import snapshot_values


class TestCurrentChartObservation:
    def test_capture_observes_real_spec_inputs_instead_of_incomplete_worker_argv(self, tmp_path: Path) -> None:
        """Real CPU spec observation retains topology and model inputs absent from abbreviated worker argv."""
        output = tmp_path / "chart.json"

        result = CliRunner().invoke(chart_app, ["--output", str(output)])

        assert result.exit_code == 0, result.exception
        record = json.loads(output.read_text())
        assert record["legacy"] is False
        assert set(record["workers"]) == {
            "rollout-executor",
            "inference-controller",
            "trainer-controller-actor",
            "trainer-engine-actor",
        }
        trainer = record["workers"]["trainer-engine-actor"]["args"]
        assert trainer["backend"]["tensor_model_parallel_size"] == 2
        assert trainer["backend"]["context_parallel_size"] == 4
        assert trainer["backend"]["global_batch_size"] == 256
        assert record["workers"]["rollout-executor"]["args"]["rollout_max_response_len"] == 8192
        assert "<FIXTURE_DIR>" in output.read_text()
        assert "--tensor-model-parallel-size" in record["scenario_argv"]
        assert "--tensor-model-parallel-size" not in record["worker_argv"]


class TestRouterEnvironmentObservation:
    @pytest.mark.parametrize("changed", [False, True])
    def test_live_router_contract_mismatch_is_retained_and_fails_the_audit(
        self, tmp_path: Path, changed: bool
    ) -> None:
        """Router wheel drift remains a typed parser difference with recorded wheel and source identity."""
        parser = argparse.ArgumentParser(add_help=False)
        RouterArgs.add_cli_args(parser, use_router_prefix=True, exclude_host_port=True)
        actions = snapshot_values(parser)["state"]["_actions"]
        router_actions = [row for row in actions if row["dest"].startswith("router_") or row["dest"] == "worker_urls"]
        if changed:
            router_actions[0] = router_actions[0] | {"default": "deliberately-different-capture-default"}
        directory = tmp_path / "after"
        (directory / "argument_schema").mkdir(parents=True)
        (directory / "argument_schema/megatron.yaml").write_text(
            json.dumps({"schema": {"parser": {"state": {"_actions": router_actions}}}})
        )
        output = tmp_path / "router.json"

        result = CliRunner().invoke(router_app, ["--after", str(directory), "--output", str(output)])

        assert result.exit_code == int(changed), result.exception
        report = json.loads(output.read_text())
        assert report["installed_distribution"] == "sglang-router"
        assert report["capture_distribution_version"] is None
        assert isinstance(report["installed_version"], str) and report["installed_version"]
        assert len(report["installed_source_sha256"]) == 64
        assert report["verdict"] == (
            "router-environment-contract-mismatch" if changed else "router-declarations-match"
        )
        assert bool(report["differences"]) is changed
        if changed:
            assert report["differences"][0]["before"]["value"] == "deliberately-different-capture-default"


class TestHistoricalSourceHashGate:
    @pytest.mark.parametrize("application", [megatron_app, sglang_app])
    def test_unverified_historical_source_is_rejected_before_its_code_can_execute(
        self, tmp_path: Path, application: object
    ) -> None:
        """Resolver replay cannot execute source whose bytes do not match its fixed historical digest."""
        source = tmp_path / "wrong_source.py"
        source.write_text("raise RuntimeError('unverified source executed')\n")
        output = tmp_path / "replay.json"

        result = CliRunner().invoke(
            application,
            [
                "--before",
                str(tmp_path / "before"),
                "--after",
                str(tmp_path / "after"),
                "--before-source",
                str(source),
                "--output",
                str(output),
            ],
        )

        assert isinstance(result.exception, AssertionError)
        assert not output.exists()


class TestProductionChartPayloadDecode:
    def test_decoded_payloads_retain_a_deliberate_worker_input_disagreement(self, tmp_path: Path) -> None:
        """All historical served payloads pass through production decoding and retain a controlled typed mismatch."""
        from tests.fast.import_isolation_utils import REPO_ROOT

        from miles.utils.test_utils.adhoc_snapshot_comparator.snapshot_chart_payload_evidence import app as payload_app

        runner = CliRunner()
        observed = tmp_path / "observed.json"
        captured = runner.invoke(chart_app, ["--output", str(observed)])
        assert captured.exit_code == 0, captured.exception
        observation = json.loads(observed.read_text())
        observation["workers"]["trainer-engine-actor"]["args"]["backend"]["global_batch_size"] = 777
        observed.write_text(json.dumps(observation))
        comparison_input = tmp_path / "synthetic_legacy_header.json"
        comparison_input.write_text(json.dumps({"legacy": True, "parsed": {}}))
        output = tmp_path / "payloads.json"

        result = runner.invoke(
            payload_app,
            [
                "--snapshots",
                str(REPO_ROOT / "tests/snapshots/charts/miles-run"),
                "--observed",
                str(observed),
                "--before-observed",
                str(comparison_input),
                "--output",
                str(output),
            ],
        )

        assert result.exit_code == 1, result.exception
        report = json.loads(output.read_text())
        assert len(report["records"]) == 8
        trainers = [record for record in report["records"] if record["worker"] == "trainer-engine-actor"]
        assert len(trainers) == 2
        for record in trainers:
            [difference] = [item for item in record["differences"] if item["path"] == ["backend", "global_batch_size"]]
            assert difference["before"] == {"present": True, "value": 777}
            assert difference["after"] == {"present": True, "value": 256}
            assert len(record["evidence"]) == len(record["differences"])
