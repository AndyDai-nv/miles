import pytest
from pydantic import JsonValue

from miles.utils.test_utils.adhoc_snapshot_comparator.snapshot_comparison_models import (
    SnapshotDifference,
    SnapshotValue,
)
from miles.utils.test_utils.adhoc_snapshot_comparator.snapshot_evidence_catalog import _describe_difference


class TestSnapshotRationaleBoundaries:
    @pytest.mark.parametrize(
        "field,value",
        [
            ("lora_B_init_method", "zero"),
            ("offload", False),
            ("hf_lora_targets", None),
            ("lora_adapter_targets", None),
            ("num_layers", None),
            ("custom_agent_function_path", None),
            ("rollout_indexer_topk_num_streams", None),
        ],
    )
    @pytest.mark.parametrize("mutation", ["before_present", "after_missing", "wrong_value"])
    def test_default_rationale_cannot_cover_existing_removed_or_different_values(
        self, field: str, value: JsonValue, mutation: str
    ) -> None:
        """A documented newly explicit default cannot justify any other presence or value transition."""
        difference = SnapshotDifference(
            path=["parsed_config/case.yaml", "config", field],
            before=SnapshotValue(present=mutation == "before_present", value=value),
            after=SnapshotValue(
                present=mutation != "after_missing",
                value=(0 if value is False else "unexpected") if mutation == "wrong_value" else value,
            ),
        )

        evidence = _describe_difference(difference=difference, category="presence/added-field")

        assert evidence["verdict"] == "requires-specific-witness"
        assert evidence["sources"] == []

    @pytest.mark.parametrize(
        "field,original", [("min_batch_collection_ratio", 1), ("tis_clip_low", 0), ("kl_coef", 0)]
    )
    @pytest.mark.parametrize("valid", [False, True])
    def test_numeric_rationale_requires_the_documented_direction_and_exact_pair(
        self, field: str, original: int, valid: bool
    ) -> None:
        """Only the specific int-to-float pairs may carry arithmetic-preservation rationale."""
        difference = SnapshotDifference(
            path=["runtime_config/case.yaml", "critic", "stages", "process_config-0000", "args", field],
            before=SnapshotValue(present=True, value=original if valid else float(original)),
            after=SnapshotValue(present=True, value=float(original) if valid else original),
        )

        evidence = _describe_difference(difference=difference, category="representation/numeric-type-change")

        assert evidence["verdict"] == ("consumer-arithmetic-preserved" if valid else "requires-specific-witness")
