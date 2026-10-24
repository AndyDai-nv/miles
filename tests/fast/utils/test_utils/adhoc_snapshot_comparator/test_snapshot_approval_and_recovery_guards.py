import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from miles.utils.test_utils.adhoc_snapshot_comparator.snapshot_comparison import app
from miles.utils.test_utils.adhoc_snapshot_comparator.snapshot_failure_recovery import _recover_actual_blocks


class TestSnapshotApprovalGuards:
    @pytest.mark.parametrize("invalid", ["duplicate_approval", "missing_human_decision", "duplicate_sort"])
    def test_otherwise_matching_rules_reject_ambiguous_or_undocumented_approval_metadata(
        self, tmp_path: Path, invalid: str
    ) -> None:
        """Duplicate or undocumented permissions cannot silently turn a strict difference into success."""
        before, after = tmp_path / "before", tmp_path / "after"
        case = "parsed_config/megatron_lora_targets.yaml"
        for root, targets, enabled in ((before, ["b", "a"], False), (after, ["a", "b"], True)):
            (root / "parsed_config").mkdir(parents=True)
            (root / case).write_text(json.dumps({"config": {"hf_lora_targets": targets, "enabled": enabled}}))
        approval = {
            "path": [case, "config", "enabled"],
            "before": {"present": True, "value": False},
            "after": {"present": True, "value": True},
            "human_decision": "Explicit test decision",
            "op": "test-op",
        }
        sort = {
            "path": [case, "config", "hf_lora_targets"],
            "op": "op48-15",
            "human_decision": "Explicit test decision",
        }
        rules = {"moves": [], "approved_differences": [approval], "list_sorts": [sort]}
        rule_path = tmp_path / "rules.json"
        rule_path.write_text(json.dumps(rules))
        runner = CliRunner()
        argv = ["--before", str(before), "--after", str(after), "--rules", str(rule_path)]
        accepted = runner.invoke(app, [*argv, "--output", str(tmp_path / "valid.json")])
        assert accepted.exit_code == 0, accepted.exception
        if invalid == "duplicate_approval":
            rules["approved_differences"].append(approval.copy())
        elif invalid == "missing_human_decision":
            approval["human_decision"] = ""
        else:
            rules["list_sorts"].append(sort.copy())
        rule_path.write_text(json.dumps(rules))
        output = tmp_path / "invalid.json"

        rejected = runner.invoke(app, [*argv, "--output", str(output)])

        assert isinstance(rejected.exception, AssertionError)
        assert not output.exists()


class TestTracebackRecoveryGuards:
    @pytest.mark.parametrize("invalid", ["traversal", "absolute", "wrong_prefix", "wrong_basename"])
    def test_malformed_traceback_blocks_cannot_create_a_recovery_path(self, invalid: str) -> None:
        """Recovery refuses escaping paths and malformed block identity or traceback prefixes."""
        relative = (
            "../outside.yaml" if invalid == "traversal" else "/outside.yaml" if invalid == "absolute" else "case.yaml"
        )
        basename = "other.yaml" if invalid == "wrong_basename" else Path(relative).name
        content = "wrong prefix" if invalid == "wrong_prefix" else "E           value: false"
        block = f"E           --- /root/miles/tests/snapshots/{relative}\nE           --- BEGIN ACTUAL {basename} ---\n{content}\nE           --- END ACTUAL ---\n"

        with pytest.raises(AssertionError):
            _recover_actual_blocks(text=block)
