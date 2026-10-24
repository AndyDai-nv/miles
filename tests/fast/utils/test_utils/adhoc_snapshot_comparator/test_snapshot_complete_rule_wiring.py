import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from miles.utils.test_utils.adhoc_snapshot_comparator.snapshot_comparison import app as comparison_app
from miles.utils.test_utils.adhoc_snapshot_comparator.snapshot_hook_rules import _complete_hook_wraps
from miles.utils.test_utils.adhoc_snapshot_comparator.snapshot_ownership import _Ownership, _scope_moves
from miles.utils.test_utils.adhoc_snapshot_comparator.snapshot_refactor_rules import app as rules_app


class TestCompleteRuleWiring:
    def test_complete_live_ownership_rules_and_hook_wrappers_feed_the_strict_comparator(self, tmp_path: Path) -> None:
        """The complete generator preserves actual values and passes its move and wrapper rules to comparison."""
        before, after = tmp_path / "before", tmp_path / "after"
        hook = "miles.rollout.sft_rollout.generate_rollout"
        old = {"train_backend": "megatron", "lr": 1e-6, "rollout_function_path": hook}
        new = {
            "train_backend": "megatron",
            "raw_megatron": {"base_args": {"lr": 1e-6}},
            "rollout_function_path": {"path": hook, "config": None},
        }
        for root, config in ((before, old), (after, new)):
            (root / "parsed_config").mkdir(parents=True)
            (root / "parsed_config/case.yaml").write_text(json.dumps({"config": config}))
        rules, output = tmp_path / "rules.json", tmp_path / "report.json"
        runner = CliRunner()

        generated = runner.invoke(rules_app, ["--before", str(before), "--output", str(rules), "--complete-ownership"])
        assert generated.exit_code == 0, generated.exception
        compared = runner.invoke(
            comparison_app,
            ["--before", str(before), "--after", str(after), "--rules", str(rules), "--output", str(output)],
        )

        assert compared.exit_code == 0, compared.exception
        report = json.loads(output.read_text())
        assert report["verified_move_count"] == 1
        assert report["unverified_move_count"] == 0
        assert report["unexplained"] == []
        assert len(report["hook_wraps"]) == 1
        assert json.loads((before / "parsed_config/case.yaml").read_text()) == {"config": old}

    def test_hook_enumeration_keeps_each_case_process_and_stage_scope(self) -> None:
        """A hook found at one capture boundary cannot replace the independent rule for another boundary."""
        hook = "miles.rollout.sft_rollout.generate_rollout"
        snapshots = {
            "parsed_config/case.yaml": {"config": {"rollout_function_path": hook}},
            "runtime_config/case.yaml": {
                "run/worker": {
                    "stages": {
                        "process_config-0000": {"args": {"rollout_function_path": hook}},
                        "train_first_step-0000": {"args": {"rollout_function_path": hook}},
                    }
                }
            },
            "argument_schema/case.yaml": {"config": {"rollout_function_path": hook}},
        }

        wraps = _complete_hook_wraps(snapshots=snapshots)

        assert {tuple(rule.path) for rule in wraps} == {
            ("parsed_config/case.yaml", "config", "rollout_function_path"),
            (
                "runtime_config/case.yaml",
                "run/worker",
                "stages",
                "process_config-0000",
                "args",
                "rollout_function_path",
            ),
            (
                "runtime_config/case.yaml",
                "run/worker",
                "stages",
                "train_first_step-0000",
                "args",
                "rollout_function_path",
            ),
        }
        assert all(rule.function == hook and rule.config_fields is None for rule in wraps)

    @pytest.mark.parametrize("backend", ["fsdp", "torchtitan"])
    @pytest.mark.parametrize("trainer", [False, True])
    def test_native_backend_ownership_keeps_its_checkpoint_fields_and_correct_destination(
        self, backend: str, trainer: bool
    ) -> None:
        """Megatron-only transient-field removal cannot remove native backend checkpoint inputs."""
        ownership = _Ownership(
            backend_fields={backend: frozenset({"lr", "load"})},
            shared_fields=frozenset({"load"}),
            trainer_fields=frozenset(),
            sglang_fields=frozenset(),
            router_fields=frozenset(),
            checkpoint_fields=frozenset({"load"}),
        )

        moves = _scope_moves(
            args={"train_backend": backend, "lr": 1e-6, "load": "/checkpoint"},
            scope=["case"],
            ownership=ownership,
            trainer=trainer,
        )

        assert {move.source[-1] for move in moves} == {"lr", "load"}
        load = next(move for move in moves if move.source[-1] == "load")
        assert load.target == ["case", "backend" if trainer else "raw_" + backend, "load"]
        assert load.keep_source is not trainer
