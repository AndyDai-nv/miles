from typing import Any

import pytest


@pytest.fixture
def chart_observations() -> tuple[dict[str, Any], dict[str, Any]]:
    before = {
        "legacy": True,
        "scenario_argv": ["--seed", "7"],
        "worker_argv": ["--worker-option", "value"],
        "parsed": {"seed": 7},
        "workers": {
            name: {"worker_class": "worker." + name, "args": {"flag": False, "nested": {"value": 0}}}
            for name in (
                "rollout-executor",
                "inference-controller",
                "trainer-controller-actor",
                "trainer-engine-actor",
            )
        },
    }
    after = before | {"legacy": False, "parsed": {"seed": 7, "typed": True}}
    after["workers"] = {
        name: record | {"args": {"flag": False, "nested": {"value": 0}, "typed": True}}
        for name, record in before["workers"].items()
    }
    return before, after
