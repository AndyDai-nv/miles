import asyncio
import json
from functools import partial
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from tests.fast.ray.rollout.conftest import make_args

from miles.backends.dynamo_utils.generate import decode_response
from miles.rollout.base_types import GenerateFnInput
from miles.rollout.generate_hub.single_turn import generate
from miles.utils.http_utils import _post
from miles.utils.types import Sample


def chunk(token, count, *, text="x", terminal=False):
    return (
        "data: "
        + json.dumps(
            {
                "output_ids": [token],
                "text": text,
                "meta_info": {
                    "id": "request",
                    "completion_tokens": count,
                    "prompt_tokens": 1,
                    "output_token_logprobs": [[-0.5, token, text]],
                    "weight_version": "2",
                    "finish_reason": {"type": "stop"} if terminal else None,
                },
            }
        )
        + "\n\n"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("first_failure", ["truncated", "status", "error"])
async def test_native_retry_discards_incomplete_attempt(monkeypatch, first_failure):
    attempts = []

    def handle(request):
        attempts.append(request)
        if len(attempts) == 1:
            if first_failure == "status":
                return httpx.Response(503)
            body = chunk(99, 1)
            if first_failure == "error":
                body += 'event: error\ndata: {"error":"worker lost"}\n\n'
        else:
            body = chunk(10, 1, text="a") + chunk(11, 2, text="b", terminal=True) + "data: [DONE]\n\n"
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, text=body)

    monkeypatch.setattr("miles.utils.http_utils.asyncio.sleep", AsyncMock())
    payload = {"input_ids": [1], "stream": True}
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        result = await _post(
            client, "http://frontend/generate", payload, response_decoder=partial(decode_response, timeout_seconds=1)
        )
    assert len(attempts) == 2
    assert result["output_ids"] == [10, 11]
    assert result["text"] == "ab"
    assert result["meta_info"]["output_token_logprobs"] == [[-0.5, 10, "a"], [-0.5, 11, "b"]]
    assert payload == {"input_ids": [1], "stream": True}


@pytest.mark.asyncio
async def test_native_transport_keeps_sixty_attempts(monkeypatch):
    calls = []

    def handle(request):
        calls.append(request)
        return httpx.Response(503)

    monkeypatch.setattr("miles.utils.http_utils.asyncio.sleep", AsyncMock())
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        with pytest.raises(httpx.HTTPStatusError):
            await _post(
                client, "http://frontend/generate", {}, response_decoder=partial(decode_response, timeout_seconds=1)
            )
    assert len(calls) == 60


@pytest.mark.asyncio
async def test_cancel_does_not_retry():
    calls = []

    async def handle(request):
        calls.append(request)
        raise asyncio.CancelledError

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        with pytest.raises(asyncio.CancelledError):
            await _post(
                client, "http://frontend/generate", {}, response_decoder=partial(decode_response, timeout_seconds=1)
            )
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_timeout_also_covers_response_headers():
    async def handle(request):
        await asyncio.Event().wait()

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        with pytest.raises(TimeoutError):
            await _post(
                client,
                "http://frontend/generate",
                {},
                max_retries=1,
                response_decoder=partial(decode_response, timeout_seconds=1),
                response_timeout=0.01,
            )


@pytest.mark.asyncio
async def test_standard_generate_uses_stream_adapter_and_native_sample_mapping(monkeypatch):
    requests = []

    def handle(request):
        requests.append(request)
        body = chunk(10, 1, text="answer", terminal=True) + "data: [DONE]\n\n"
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, text=body)

    args = make_args(
        dynamo_config=SimpleNamespace(request_timeout_seconds=1),
        sglang_api_key="local-key",
        sglang_router_ip="frontend",
        sglang_router_port=8000,
    )
    state = SimpleNamespace(args=args, processor=None, tokenizer=SimpleNamespace(encode=lambda *a, **k: [1]))
    sample = Sample(prompt="question")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        monkeypatch.setattr("miles.utils.http_utils._rollout_client", lambda: client)
        output = await generate(
            GenerateFnInput(state=state, sample=sample, sampling_params={"max_new_tokens": 2}, evaluation=False)
        )
    assert output.samples is sample
    assert sample.tokens == [1, 10]
    assert sample.rollout_log_probs == [-0.5]
    assert sample.status is Sample.Status.COMPLETED
    assert json.loads(requests[0].content)["stream"] is True
    assert requests[0].headers["Authorization"] == "Bearer local-key"


@pytest.mark.asyncio
async def test_abort_echo_is_not_appended_twice():
    abort = {
        "output_ids": [10],
        "text": "a",
        "meta_info": {
            "id": "request",
            "completion_tokens": 1,
            "output_token_logprobs": [[-0.5, 10, "a"]],
            "finish_reason": {"type": "abort"},
        },
    }
    response = httpx.Response(
        200,
        headers={"content-type": "text/event-stream"},
        text=chunk(10, 1, text="a") + "data: " + json.dumps(abort) + "\n\ndata: [DONE]\n\n",
    )
    output = await decode_response(response, timeout_seconds=1)
    assert output["text"] == "a"
    assert output["output_ids"] == [10]
    assert output["meta_info"]["output_token_logprobs"] == [[-0.5, 10, "a"]]
