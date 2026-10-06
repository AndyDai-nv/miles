import copy
import json
from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from tests.fast.backends.dynamo_utils.test_config import config_dict
from tests.fast.backends.dynamo_utils.test_stream_sample import _chunk

from miles.backends.dynamo_utils.config import DynamoConfig
from miles.backends.dynamo_utils.generate import GenerationContext, generate
from miles.backends.dynamo_utils.rollout import DynamoRolloutFn
from miles.rollout.base_types import GenerateFnInput, RolloutFnConstructorInput, RolloutFnTrainInput
from miles.utils.http_utils import GeneralHttpClientProvider
from miles.utils.types import Sample

pytestmark = pytest.mark.asyncio


def _state():
    return SimpleNamespace(
        args=Namespace(
            rollout_max_response_len=8,
            rollout_max_context_len=None,
            use_sampling_support_replay=False,
            use_rollout_routing_replay=False,
            use_rollout_indexer_replay=False,
            rollout_top_logprobs_num=0,
        ),
        processor=None,
        tokenizer=SimpleNamespace(encode=lambda *a, **kw: [1, 2]),
        dynamo_context=GenerationContext("http://frontend:8000/generate", "17", 1),
    )


async def test_end_to_end_payload_transport_mapping_without_retokenizing_output(monkeypatch):
    seen = []

    async def handle(request):
        seen.append(json.loads(request.content))
        chunks = [_chunk([3], count=1, text="hello", terminal={"type": "stop"})]
        content = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n"
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=content)

    sample = Sample(prompt="prompt")
    before = copy.deepcopy(sample.to_dict())
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        monkeypatch.setattr(GeneralHttpClientProvider, "client", lambda: client)
        result = await generate(GenerateFnInput(_state(), sample, {"max_new_tokens": 4, "temperature": 0.7}, False))
    assert seen[0]["input_ids"] == [1, 2]
    assert seen[0]["stream"] is True and seen[0]["return_logprob"] is True
    assert seen[0]["sampling_params"]["temperature"] == 0.7
    assert result.samples.tokens == [1, 2, 3]
    assert result.samples.rollout_log_probs == [-0.5]
    assert sample.to_dict() == before


@pytest.mark.parametrize("change", ["partial", "multimodal", "replay", "no_context", "tokens"])
async def test_rejects_incompatible_request_before_sending(change, monkeypatch):
    state, sample = _state(), Sample(prompt="prompt")
    if change == "partial":
        sample.response = "partial"
    elif change == "multimodal":
        sample.multimodal_inputs = {"images": ["image"]}
    elif change == "replay":
        state.args.use_rollout_routing_replay = True
    elif change == "no_context":
        state.dynamo_context = None
    else:
        sample.tokens = [99]

    def unexpected():
        pytest.fail("invalid request reached transport")

    monkeypatch.setattr(GeneralHttpClientProvider, "client", unexpected)
    with pytest.raises((ValueError, RuntimeError)):
        await generate(GenerateFnInput(state, sample, {"max_new_tokens": 4}, False))


def _rollout(monkeypatch):
    import miles.backends.dynamo_utils.rollout as module

    args = Namespace(
        dynamo_config=DynamoConfig.model_validate(config_dict()),
        rollout_batch_size=2,
        n_samples_per_prompt=2,
        reward_key=None,
        sglang_router_ip="::1",
        sglang_router_port=8000,
    )
    state = SimpleNamespace(args=args, sampling_params={"max_new_tokens": 8})
    monkeypatch.setattr(module, "GenerateState", lambda args: state)
    controller = SimpleNamespace(
        dynamo_acquire_rollout=AsyncMock(return_value="ticket"), dynamo_release_rollout=AsyncMock()
    )
    groups = [[Sample(prompt="p", index=i * 2 + j) for j in range(2)] for i in range(2)]
    source = SimpleNamespace(get_samples=lambda n: groups)
    fn = DynamoRolloutFn(RolloutFnConstructorInput(args=args, data_source=source, inference_controller=controller))
    return fn, controller, state, module


async def test_grpo_batch_uses_existing_group_reward_flow(monkeypatch):
    fn, controller, state, module = _rollout(monkeypatch)

    async def group(state, samples, params):
        assert state.dynamo_context.weight_version == "17"
        assert state.dynamo_context.url == "http://[::1]:8000/generate"
        for sample in samples:
            sample.reward = 1
        return samples

    monkeypatch.setattr(module, "generate_and_rm_group", group)
    result = await fn(RolloutFnTrainInput(rollout_id=0, weight_version=17))
    assert len(result.samples) == 2 and all(len(group) == 2 for group in result.samples)
    assert result.samples[0][0].reward == 1
    controller.dynamo_acquire_rollout.assert_awaited_once_with(version="17")
    controller.dynamo_release_rollout.assert_awaited_once_with(ticket="ticket", success=True)
    assert state.dynamo_context is None


async def test_request_failure_closes_gate_and_cancels_siblings(monkeypatch):
    import asyncio

    fn, controller, state, module = _rollout(monkeypatch)
    settled = []

    async def group(state, samples, params):
        try:
            if samples[0].index == 0:
                await asyncio.sleep(0)
                raise ValueError("wrong policy version")
            await asyncio.Event().wait()
        finally:
            settled.append(samples[0].index)

    monkeypatch.setattr(module, "generate_and_rm_group", group)
    with pytest.raises(ValueError, match="policy"):
        await fn(RolloutFnTrainInput(rollout_id=0, weight_version=17))
    assert sorted(settled) == [0, 2]
    controller.dynamo_release_rollout.assert_awaited_once_with(ticket="ticket", success=False)
    assert state.dynamo_context is None


async def test_missing_reward_discards_batch_and_fails_closed(monkeypatch):
    fn, controller, _, module = _rollout(monkeypatch)
    monkeypatch.setattr(module, "generate_and_rm_group", AsyncMock(side_effect=lambda state, group, params: group))
    with pytest.raises(ValueError, match="missing_reward"):
        await fn(RolloutFnTrainInput(rollout_id=0, weight_version=17))
    controller.dynamo_release_rollout.assert_awaited_once_with(ticket="ticket", success=False)


async def test_completed_samples_cannot_bypass_generation_version_checks(monkeypatch):
    fn, controller, _, module = _rollout(monkeypatch)
    fn.constructor_input.data_source.get_samples(2)[0][0].status = Sample.Status.COMPLETED
    generate_group = AsyncMock()
    monkeypatch.setattr(module, "generate_and_rm_group", generate_group)
    with pytest.raises(ValueError, match="fresh"):
        await fn(RolloutFnTrainInput(rollout_id=0, weight_version=17))
    generate_group.assert_not_awaited()
    controller.dynamo_release_rollout.assert_awaited_once_with(ticket="ticket", success=False)
