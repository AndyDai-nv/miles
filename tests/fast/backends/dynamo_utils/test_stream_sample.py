import copy

import pytest

from miles.backends.dynamo_utils.stream_sample import sample_from_stream
from miles.utils.types import Sample

pytestmark = pytest.mark.asyncio


async def _events(values):
    for value in values:
        yield value


def _chunk(ids, *, count, terminal=None, text="x", version="17"):
    return {
        "text": text,
        "output_ids": ids,
        "meta_info": {
            "id": "request-a",
            "prompt_tokens": 2,
            "completion_tokens": count,
            "finish_reason": terminal,
            "weight_version": version,
            "weight_versions": [{"version": version, "start": 0, "end": count}],
            "output_token_logprobs": [[-0.5, token, None] for token in ids],
        },
    }


async def _map(chunks, *, sample=None, **overrides):
    return await sample_from_stream(
        sample or Sample(tokens=[1, 2]),
        _events(chunks),
        **{"expected_weight_version": "17", "max_new_tokens": 8, **overrides},
    )


@pytest.mark.parametrize("finish,status", [("stop", Sample.Status.COMPLETED), ("length", Sample.Status.TRUNCATED)])
async def test_incremental_tokens_and_cumulative_spans(finish, status):
    sample = Sample(tokens=[1, 2], metadata={"keep": [1]})
    before = copy.deepcopy(sample.to_dict())
    result = await _map(
        [
            _chunk([3, 4], count=2, text="你"),
            _chunk([5], count=3, text="好", terminal={"type": finish}),
        ],
        sample=sample,
    )
    assert sample.to_dict() == before
    assert result.tokens == [1, 2, 3, 4, 5]
    assert result.response == "你好"
    assert result.response_length == 3
    assert result.rollout_log_probs == [-0.5] * 3
    assert result.status is status
    assert result.weight_versions[0].to_dicts() == [{"version": "17", "abs_start": 2, "abs_end": 5}]
    assert result.metadata == sample.metadata and result.metadata is not sample.metadata


async def test_version_scalar_fallback_and_empty_terminal_delta():
    first = _chunk([3], count=1)
    final = _chunk([], count=1, text="", terminal={"type": "stop"})
    for value in (first, final):
        value["meta_info"].pop("weight_versions")
    result = await _map([first, final])
    assert result.tokens == [1, 2, 3]
    assert len(result.all_weight_version_spans) == 1


async def test_zero_token_generation():
    result = await _map([_chunk([], count=0, text="", terminal={"type": "stop"})])
    assert result.response_length == 0
    assert result.all_weight_version_spans == []


async def test_cumulative_metrics_are_added_only_once():
    first = _chunk([3], count=1)
    final = _chunk([4], count=2, terminal={"type": "stop"})
    for chunk in (first, final):
        chunk["meta_info"].update(cached_tokens=1, spec_verify_ct=2, spec_num_correct_drafts=1)
    result = await _map([first, final])
    assert result.prefix_cache_info.cached_tokens == 1
    assert result.prefix_cache_info.total_prompt_tokens == 2
    assert result.spec_info.spec_verify_ct == 2


@pytest.mark.parametrize(
    "field,value", [("output_ids", [True]), ("output_ids", [-1]), ("text", None), ("meta_info", None)]
)
async def test_malformed_native_response(field, value):
    chunk = _chunk([3], count=1, terminal={"type": "stop"})
    chunk[field] = value
    with pytest.raises(ValueError):
        await _map([chunk])


@pytest.mark.parametrize(
    "field,value",
    [
        ("weight_version", None),
        ("weight_version", "18"),
        ("weight_version", 17),
        ("completion_tokens", 2),
        ("completion_tokens", True),
        ("prompt_tokens", 9),
        ("output_token_logprobs", []),
        ("output_token_logprobs", [[float("nan"), 3]]),
        ("output_token_logprobs", [[-0.5, 99]]),
        ("output_token_logprobs", [[0.5, 3]]),
        ("finish_reason", {"type": "abort"}),
        ("finish_reason", {"type": "unknown"}),
        ("weight_versions", [{"version": "16", "start": 0, "end": 1}]),
        ("weight_versions", [{"version": "17", "start": 1, "end": 1}]),
        ("weight_versions", []),
        ("routed_experts", "payload"),
        ("output_top_logprobs", []),
    ],
)
async def test_invalid_metadata_is_not_a_training_sample(field, value):
    chunk = _chunk([3], count=1, terminal={"type": "stop"})
    chunk["meta_info"][field] = value
    with pytest.raises(ValueError):
        await _map([chunk])


async def test_missing_terminal_changed_identity_and_duplicate_output():
    first = _chunk([3], count=1)
    for second in (_chunk([3], count=1), _chunk([4], count=2, terminal={"type": "stop"})):
        second["meta_info"]["id"] = "other"
        with pytest.raises(ValueError):
            await _map([first, second])
    with pytest.raises(ValueError, match="terminal"):
        await _map([first])
    final = _chunk([3], count=1, terminal={"type": "stop"})
    with pytest.raises(ValueError, match="terminal"):
        await _map([final, final])
    with pytest.raises(ValueError, match="token count"):
        await _map([first, _chunk([3], count=1, terminal={"type": "stop"})])


async def test_input_unchanged_on_stream_failure():
    sample = Sample(tokens=[1, 2])

    async def broken():
        yield _chunk([3], count=1)
        raise RuntimeError("connection lost")

    with pytest.raises(RuntimeError):
        await sample_from_stream(sample, broken(), expected_weight_version="17", max_new_tokens=4)
    assert sample.tokens == [1, 2] and sample.response == ""


@pytest.mark.parametrize("options", [{"expected_weight_version": ""}, {"max_new_tokens": 0}, {"max_new_tokens": True}])
async def test_invalid_contract(options):
    with pytest.raises(ValueError):
        await _map([], **options)


async def test_reject_partial_sample_and_token_overflow():
    with pytest.raises(ValueError, match="fresh"):
        await _map([], sample=Sample(tokens=[1, 2, 3], response="old", response_length=1))
    with pytest.raises(ValueError, match="token count"):
        await _map([_chunk([3, 4], count=2, terminal={"type": "stop"})], max_new_tokens=1)
