import copy
import math
from collections.abc import AsyncIterable
from dataclasses import replace
from typing import Any

from miles.utils.types import Sample, WeightVersionsPerCall

_UNSUPPORTED_META = (
    "output_top_logprobs",
    "output_token_sampling_mask",
    "routed_experts",
    "indexer_topk",
)


async def sample_from_stream(
    sample: Sample,
    events: AsyncIterable[dict[str, Any]],
    *,
    expected_weight_version: str,
    max_new_tokens: int,
) -> Sample:
    """Map a fully consumed, validated single-turn native stream; never mutate the input."""
    if not isinstance(expected_weight_version, str) or not expected_weight_version.strip():
        raise ValueError("expected_weight_version must come from the trainer")
    if type(max_new_tokens) is not int or max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be a positive integer")
    if (
        sample.status is not Sample.Status.PENDING
        or not sample.tokens
        or sample.response
        or sample.response_length
        or sample.rollout_log_probs
        or sample.weight_versions
        or sample.loss_mask is not None
        or sample.multimodal_inputs
        or sample.multimodal_train_inputs
    ):
        raise ValueError("expected a fresh single-turn text sample with prompt tokens")
    tokens, logprobs, texts = [], [], []
    terminal = None
    request_id = None
    final_meta = None
    async for event in events:
        if terminal is not None:
            raise ValueError("response after terminal event")
        meta, ids, probs, text = _validate_chunk(event)
        if request_id is not None and meta["id"] != request_id:
            raise ValueError("request identity changed within the stream")
        request_id = meta["id"]
        if meta["prompt_tokens"] != len(sample.tokens):
            raise ValueError("engine prompt token count does not match the sample")
        count = len(tokens) + len(ids)
        if count > max_new_tokens or meta["completion_tokens"] != count:
            raise ValueError("completion token count does not match incremental output")
        _validate_version(meta, expected=expected_weight_version, count=count)
        tokens.extend(ids)
        logprobs.extend(probs)
        texts.append(text)
        terminal = meta.get("finish_reason")
        final_meta = meta
    if terminal is None:
        raise ValueError("stream has no terminal response")
    meta = {**final_meta, "output_token_logprobs": [[p, t] for p, t in zip(logprobs, tokens, strict=True)]}
    result = replace(
        sample,
        tokens=[*sample.tokens, *tokens],
        response="".join(texts),
        response_length=len(tokens),
        rollout_log_probs=logprobs,
        weight_versions=[WeightVersionsPerCall.from_meta_info(meta, output_end=len(sample.tokens) + len(tokens))],
        status=Sample.Status.TRUNCATED if terminal["type"] == "length" else Sample.Status.COMPLETED,
        metadata=copy.deepcopy(sample.metadata),
        prefix_cache_info=copy.deepcopy(sample.prefix_cache_info),
        spec_info=copy.deepcopy(sample.spec_info),
    )
    result.prefix_cache_info.add(meta_info=meta)
    result.spec_info.add(meta_info=meta)
    return result


def _validate_chunk(event: dict) -> tuple[dict, list[int], list[float], str]:
    if not isinstance(event, dict) or "error" in event:
        raise ValueError("expected a native response")
    meta, ids, text = event.get("meta_info"), event.get("output_ids"), event.get("text")
    if not isinstance(meta, dict) or not isinstance(ids, list) or not isinstance(text, str):
        raise ValueError("response requires meta_info, output_ids and incremental text")
    if any(type(token) is not int or token < 0 for token in ids):
        raise ValueError("invalid output token ID")
    if not isinstance(meta.get("id"), str) or not meta["id"]:
        raise ValueError("missing request identity")
    if any(type(meta.get(key)) is not int or meta[key] < 0 for key in ("prompt_tokens", "completion_tokens")):
        raise ValueError("missing or invalid token counts")
    if any(meta.get(key) is not None for key in _UNSUPPORTED_META):
        raise ValueError("response contains unsupported replay/top-k metadata")
    reason = meta.get("finish_reason")
    if reason is not None and (not isinstance(reason, dict) or reason.get("type") not in ("stop", "length")):
        raise ValueError("generation aborted or returned an unsupported finish reason")
    pairs = meta.get("output_token_logprobs")
    if not isinstance(pairs, list) or len(pairs) != len(ids):
        raise ValueError("logprobs must cover every incremental output token")
    probs = []
    for token, pair in zip(ids, pairs, strict=True):
        if (
            not isinstance(pair, (list, tuple))
            or len(pair) < 2
            or type(pair[1]) is not int
            or pair[1] != token
            or type(pair[0]) not in (int, float)
            or not math.isfinite(pair[0])
            or pair[0] > 0
        ):
            raise ValueError("invalid token/logprob pair")
        probs.append(float(pair[0]))
    return meta, ids, probs, text


def _validate_version(meta: dict, *, expected: str, count: int) -> None:
    if meta.get("weight_version") != expected:
        raise ValueError("missing or unexpected weight version")
    # sglang-miles slices token/logprob deltas, but keeps version spans cumulative.
    if (spans := meta.get("weight_versions")) is None:
        return
    if not isinstance(spans, list):
        raise ValueError("invalid weight version spans")
    end = 0
    for span in spans:
        if (
            not isinstance(span, dict)
            or span.get("version") != expected
            or type(span.get("start")) is not int
            or type(span.get("end")) is not int
            or span["start"] != end
            or not end <= span["end"] <= count
            or (span["end"] == end and count != 0)
        ):
            raise ValueError("mixed, gapped or invalid weight version spans")
        end = span["end"]
    if end != count:
        raise ValueError("weight version spans do not cover the output")
