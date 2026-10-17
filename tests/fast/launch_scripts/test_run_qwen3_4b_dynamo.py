import shlex
from types import SimpleNamespace

import pytest

from tests.fast.launch_scripts.py_harness import (
    REPO_ROOT,
    call_entrypoint,
    freeze_environment,
    import_launch_script,
    install_command_recorder,
)

from miles.rollout.rm_hub import async_rm
from miles.utils.types import Sample


@pytest.mark.asyncio
@pytest.mark.parametrize("router", ["dynamo", "cache_aware"])
async def test_comparison_starts_fresh_from_converted_weights(monkeypatch, tmp_path, router):
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
    assert argv[argv.index("--rollout-max-response-len") + 1] == "4096"
    assert argv[argv.index("--max-tokens-per-gpu") + 1] == "8192"
    reward_args = SimpleNamespace(custom_rm_path=None, rm_type=argv[argv.index("--rm-type") + 1])
    assert await async_rm(reward_args, Sample(response=r"The answer is \boxed{6}.", label="6")) == 1
    assert await async_rm(reward_args, Sample(response=r"The answer is \boxed{7}.", label="6")) == 0
