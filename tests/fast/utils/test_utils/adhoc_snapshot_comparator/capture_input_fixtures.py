import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from miles.utils.audit_utils.config_snapshot.models import (
    ConfigSnapshotContext,
    ConfigSnapshotGeneratedValue,
    ConfigSnapshotPoint,
    ConfigSnapshotRecord,
)
from miles.utils.audit_utils.process_identity import SimpleProcessIdentity


@dataclass(frozen=True)
class _CaptureInputs:
    before: Path
    after: Path
    raw: Path
    record: Path
    after_case: Path
    expanded: dict


@pytest.fixture
def capture_inputs(tmp_path: Path) -> _CaptureInputs:
    before, after, raw = tmp_path / "before", tmp_path / "after", tmp_path / "raw"
    for root in (before, after):
        (root / "runtime_config/tests/e2e").mkdir(parents=True)
    (raw / "records").mkdir(parents=True)
    (raw / "attempt.json").write_text(json.dumps({"test": "tests/e2e/case.py", "completed": True}))
    digest = "redacted-sha256:original"
    record = ConfigSnapshotRecord(
        context=ConfigSnapshotContext(
            name="test/run-0000",
            deploy_component="all",
            deploy_instance_id="default",
            source=SimpleProcessIdentity(component="main"),
            run_uuid="uuid",
            capture_id="capture",
        ),
        point=ConfigSnapshotPoint(stage="process_config", index=0),
        config={"args": {"seed": 7, "wandb_key": digest}},
        generated_values=[ConfigSnapshotGeneratedValue(kind="wandb_key", name="environment", value=digest)],
    )
    record_path = raw / "records/record.json"
    record_path.write_text(record.model_dump_json())
    process = "test/run-0000/all/default/main"
    expanded = {
        process: {
            "ranks": [],
            "stages": {"process_config-0000": {"args": {"seed": 7, "wandb_key": "$WANDB_KEY_environment"}}},
        }
    }
    for root, key in ((before, digest), (after, "$WANDB_KEY_environment")):
        (root / "runtime_config/tests/e2e/case.yaml").write_text(
            json.dumps(
                {"processes": {process: {"ranks": [], "base": {"args": {"seed": 7, "wandb_key": key}}, "diffs": {}}}}
            )
        )
    return _CaptureInputs(
        before=before,
        after=after,
        raw=raw,
        record=record_path,
        after_case=after / "runtime_config/tests/e2e/case.yaml",
        expanded=expanded,
    )
