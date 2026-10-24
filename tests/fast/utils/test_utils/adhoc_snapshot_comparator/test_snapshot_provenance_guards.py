import json
from pathlib import Path
from typing import Any

import pytest
from tests.fast.utils.test_utils.adhoc_snapshot_comparator.provenance_input_fixtures import provenance_inputs

from miles.utils.test_utils.adhoc_snapshot_comparator.snapshot_provenance_evidence import _load_provenance

__all__ = ["provenance_inputs"]


class TestProvenanceInputGuards:
    @pytest.mark.parametrize(
        "invalid",
        [
            "stage_match",
            "kind",
            "name",
            "plain_digest",
            "short_hash",
            "nonhex_hash",
            "point",
            "field",
            "before_missing",
            "after_missing",
            "token",
            "duplicate",
            "normalizer",
        ],
    )
    def test_provenance_load_rejects_each_broken_witness_precondition(
        self, tmp_path: Path, provenance_inputs: tuple[Path, dict[str, Any]], invalid: str
    ) -> None:
        """A valid proof stops being usable when any required origin, type, stage, or identity constraint breaks."""
        inventory, payload = provenance_inputs
        source = tmp_path / "proof.json"
        source.write_text(json.dumps(payload))
        assert len(_load_provenance(path=source, after_directory=inventory)) == 1
        row = payload["proof"][0]
        witness = row["witness"]
        if invalid == "stage_match":
            witness["full_normalized_stage_matches"] = False
        elif invalid == "kind":
            witness["registered_kind"] = "run_id"
        elif invalid == "name":
            witness["registered_name"] = "explicit"
        elif invalid == "plain_digest":
            row["before"]["value"] = witness["registered_redacted_value"] = "plain"
        elif invalid == "short_hash":
            witness["sha256"] = "a" * 63
        elif invalid == "nonhex_hash":
            witness["sha256"] = "g" * 64
        elif invalid == "point":
            witness["point"]["index"] = 1
        elif invalid == "field":
            row["path"][-1] = "unrelated"
        elif invalid == "before_missing":
            row["before"]["present"] = False
        elif invalid == "after_missing":
            row["after"]["present"] = False
        elif invalid == "token":
            row["after"]["value"] = "$WANDB_KEY_other"
        elif invalid == "duplicate":
            payload["proof"].append(row)
        else:
            payload["normalizer_sha256"] = "0" * 64
        source.write_text(json.dumps(payload))

        with pytest.raises(AssertionError):
            _load_provenance(path=source, after_directory=inventory)
