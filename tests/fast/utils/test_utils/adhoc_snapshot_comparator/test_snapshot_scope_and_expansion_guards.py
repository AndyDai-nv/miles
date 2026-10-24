import json
from pathlib import Path

import pytest

from miles.utils.test_utils.adhoc_snapshot_comparator.snapshot_comparison_models import (
    SnapshotDifference,
    SnapshotValue,
)
from miles.utils.test_utils.adhoc_snapshot_comparator.snapshot_expansion import expand_snapshot_directory
from miles.utils.test_utils.adhoc_snapshot_comparator.snapshot_scope_evidence import _ScopeEvidence


class TestEndpointScopeWitnesses:
    @pytest.mark.parametrize("witness", ["matching", "typed_mismatch", "different_generation"])
    def test_endpoint_reset_reports_only_same_generation_exact_typed_value_evidence(self, witness: str) -> None:
        """A nullable trainer endpoint cannot borrow a numerically equal or different-generation discovery."""
        case = "runtime_config/case.yaml"
        owner = "other/main" if witness == "different_generation" else "run/main"
        observed = 8000.0 if witness == "typed_mismatch" else 8000
        difference = SnapshotDifference(
            path=[case, "run/actor", "stages", "process_config-0000", "args", "sglang_router_port"],
            before=SnapshotValue(present=True, value=8000),
            after=SnapshotValue(present=True, value=None),
        )
        evidence = _ScopeEvidence(
            actual={
                case: {
                    owner: {"stages": {"router_endpoints-0000": {"args": {"sglang_router_port": observed}}}},
                    "run/actor": {
                        "ranks": [0],
                        "stages": {"process_config-0000": {"args": {"sglang_router_port": None}}},
                    },
                }
            },
            fields={"all": frozenset({"sglang_router_port"}), "trainer": frozenset({"sglang_router_port"})},
        )

        result = evidence.explain(difference=difference)

        assert result["rule"] == "scope/endpoint-state-not-copied"
        assert result["verdict"] == (
            "same-generation-value-found" if witness == "matching" else "same-generation-value-not-found"
        )
        assert (result["witness"] is not None) is (witness == "matching")
        assert difference.after.present and difference.after.value is None


class TestExpansionInputRejections:
    @pytest.mark.parametrize("base", [None, "../outside", "/absolute"])
    def test_delta_base_must_be_a_sibling_name(self, tmp_path: Path, base: str | None) -> None:
        """Malformed delta metadata cannot direct expansion outside its recorded sibling inventory."""
        (tmp_path / "case.delta.yaml").write_text(json.dumps({"base": base}))

        with pytest.raises(AssertionError, match="Expected a sibling base name"):
            expand_snapshot_directory(directory=tmp_path)

    @pytest.mark.parametrize("invalid", ["empty_processes", "duplicate_base_stage"])
    def test_runtime_case_cannot_omit_processes_or_redeclare_the_base_stage(
        self, tmp_path: Path, invalid: str
    ) -> None:
        """Runtime inventory must contain a process and exactly one base stage per process."""
        runtime = tmp_path / "runtime_config"
        runtime.mkdir()
        processes = (
            {}
            if invalid == "empty_processes"
            else {
                "main": {
                    "ranks": [],
                    "base": {"args": {}},
                    "diffs": {"process_config-0000": {"set": {}, "remove": []}},
                }
            }
        )
        (runtime / "case.yaml").write_text(json.dumps({"processes": processes}))
        expected = "must contain processes" if invalid == "empty_processes" else "Base stage occurs twice"

        with pytest.raises(AssertionError, match=expected):
            expand_snapshot_directory(directory=tmp_path)
