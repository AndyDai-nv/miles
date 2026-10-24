from pathlib import Path
from typing import Any

import pytest

from miles.utils.test_utils.adhoc_snapshot_comparator.snapshot_evidence_identity import _snapshot_identity
from miles.utils.test_utils.adhoc_snapshot_comparator.snapshot_provenance_evidence import _normalizer_hash


@pytest.fixture
def provenance_inputs(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    inventory = tmp_path / "inventory"
    inventory.mkdir()
    (inventory / "case.yaml").write_text("value: false\n")
    row = {
        "path": ["runtime_config/case", "run/main", "stages", "process_config-0000", "args", "wandb_key"],
        "before": {"present": True, "value": "redacted-sha256:original"},
        "after": {"present": True, "value": "$WANDB_KEY_environment"},
        "witness": {
            "full_normalized_stage_matches": True,
            "registered_kind": "wandb_key",
            "registered_name": "environment",
            "registered_redacted_value": "redacted-sha256:original",
            "sha256": "a" * 64,
            "point": {"stage": "process_config", "index": 0},
        },
    }
    return inventory, {
        "normalizer_sha256": _normalizer_hash(),
        "after_inventory": _snapshot_identity(directory=inventory),
        "proof": [row],
    }
