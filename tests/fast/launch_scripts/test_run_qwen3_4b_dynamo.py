import shlex

import pytest

from tests.fast.launch_scripts.py_harness import (
    REPO_ROOT,
    call_entrypoint,
    freeze_environment,
    import_launch_script,
    install_command_recorder,
)


@pytest.mark.parametrize("router", ["dynamo", "cache_aware"])
def test_comparison_starts_fresh_from_converted_weights(monkeypatch, tmp_path, router):
    freeze_environment(monkeypatch)
    recording = install_command_recorder(monkeypatch)
    module = import_launch_script(REPO_ROOT / "scripts/run_qwen3_4b_dynamo.py")
    call_entrypoint(module, "execute", {"router": router}, sandbox=tmp_path)
    command = next(command for command in recording.commands if "ray job submit" in command)
    argv = shlex.split(command)

    assert "--finetune" in argv
    assert argv[argv.index("--load") + 1].endswith("/Qwen3-4B_torch_dist")
    assert argv[argv.index("--actor-num-gpus-per-node") + 1] == "4"
    assert argv[argv.index("--rollout-num-gpus") + 1] == "4"
    assert argv[argv.index("--rollout-num-gpus-per-engine") + 1] == "2"
    assert ("--dynamo-config" in argv) == (router == "dynamo")
    assert ("--sglang-router-policy" in argv) == (router == "cache_aware")
