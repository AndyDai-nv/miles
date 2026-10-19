import asyncio
from functools import partial

import httpx

from miles.backends.dynamo_utils.generate_transport import GenerateProtocolError, _generate_events
from miles.utils.http_utils import post


async def post_generate(args, url: str, payload: dict, *, headers=None) -> dict:
    config = getattr(args, "dynamo_config", None)
    if config is None:
        return await post(url, payload, headers=headers)
    headers = dict(headers or {})
    if args.sglang_api_key:
        headers["Authorization"] = f"Bearer {args.sglang_api_key}"
    return await post(
        url,
        {**payload, "stream": True},
        headers=headers,
        response_decoder=partial(decode_response, timeout_seconds=config.request_timeout_seconds),
        response_timeout=config.request_timeout_seconds,
    )


async def decode_response(response: httpx.Response, *, timeout_seconds: float) -> dict:
    if response.headers.get("content-type", "").split(";")[0].strip().lower() != "text/event-stream":
        raise GenerateProtocolError("expected text/event-stream")
    texts, ids = [], []
    arrays = {}
    request_id = None
    final = None
    async with asyncio.timeout(timeout_seconds):
        async for event in _generate_events(response.aiter_bytes(), max_event_bytes=8 * 1024 * 1024):
            meta = event["meta_info"]
            if request_id is not None and meta.get("id") != request_id:
                raise GenerateProtocolError("request identity changed within the stream")
            request_id = meta.get("id")
            if (
                not request_id
                or not isinstance(event.get("text"), str)
                or not isinstance(event.get("output_ids"), list)
            ):
                raise GenerateProtocolError("missing native response identity, text or tokens")
            chunk_ids = event["output_ids"]
            if (meta.get("finish_reason") or {}).get("type") == "abort":
                # sglang-miles abort echoes cumulative text/logprobs, even in incremental mode.
                pairs = meta.get("output_token_logprobs", [])
                if len(pairs) != meta.get("completion_tokens"):
                    raise GenerateProtocolError("aborted response has incomplete token logprobs")
                ids = [pair[1] for pair in pairs]
                texts = [event["text"]]
                arrays = {}
                final = event
                continue
            pairs = meta.get("output_token_logprobs")
            if pairs is not None and (len(pairs) != len(chunk_ids) or [pair[1] for pair in pairs] != chunk_ids):
                raise GenerateProtocolError("incremental tokens and logprobs disagree")
            ids.extend(chunk_ids)
            texts.append(event["text"])
            if meta.get("completion_tokens") != len(ids):
                raise GenerateProtocolError("expected incremental token output")
            for name in (
                "output_token_logprobs",
                "output_top_logprobs",
                "output_token_ids_logprobs",
                "output_token_sampling_mask",
                "output_token_sampling_logprobs",
            ):
                if meta.get(name) is not None:
                    arrays.setdefault(name, []).extend(meta[name])
            final = event
    if final is None:
        raise GenerateProtocolError("empty generation response")
    # Version spans and usage are cumulative; token-aligned arrays and text are deltas.
    return {**final, "text": "".join(texts), "output_ids": ids, "meta_info": {**final["meta_info"], **arrays}}
