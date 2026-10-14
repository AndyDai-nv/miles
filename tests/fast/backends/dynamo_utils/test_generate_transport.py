import asyncio
import json

import httpx
import pytest

from miles.backends.dynamo_utils.generate_transport import GenerateProtocolError, stream_generate

pytestmark = pytest.mark.asyncio


class _Stream(httpx.AsyncByteStream):
    def __init__(self, chunks, *, block=False):
        self.chunks = chunks
        self.block = block
        self.closed = False
        self.waiting = asyncio.Event()

    async def __aiter__(self):
        for chunk in self.chunks:
            yield chunk
        if self.block:
            self.waiting.set()
            await asyncio.Event().wait()

    async def aclose(self):
        self.closed = True


def _event(*, terminal=False):
    return {"text": "你好", "output_ids": [12], "meta_info": {"finish_reason": {"type": "stop"} if terminal else None}}


def _wire(value):
    return b"data: " + json.dumps(value, ensure_ascii=False).encode() + b"\n\n"


async def _collect(chunks, *, status=200, content_type="text/event-stream", **kwargs):
    stream = _Stream(chunks)
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(status, headers={"content-type": content_type}, stream=stream)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        try:
            async with stream_generate(
                client, url="http://frontend/generate", payload={"input_ids": [1]}, timeout_seconds=1, **kwargs
            ) as events:
                result = [event async for event in events]
        finally:
            assert stream.closed
    assert len(requests) == 1
    assert json.loads(requests[0].content) == {"input_ids": [1], "stream": True}
    return result


@pytest.mark.parametrize("split", [1, 2, 7, 4096])
async def test_chunk_boundaries(split):
    wire = b": heartbeat\r\n\r\n" + _wire(_event()) + _wire(_event(terminal=True)) + b"data: [DONE]\n\n"
    assert await _collect([wire[i : i + split] for i in range(0, len(wire), split)]) == [
        _event(),
        _event(terminal=True),
    ]


async def test_multiline_data_and_cr_delimiters():
    wire = b'event: message\rdata: {"meta_info":\rdata: {"finish_reason":{"type":"stop"}}}\r\rdata: [DONE]\r\r'
    # A final LF also completes a split CRLF.
    assert len(await _collect([wire, b"\n"])) == 1
    assert len(await _collect([wire])) == 1


@pytest.mark.parametrize(
    "wire",
    [
        b"data: [DONE]\n\n",
        b"data: broken\n\n",
        b"data: []\n\n",
        b'data: {"error":"bad"}\n\n',
        b"event: error\ndata: rejected\n\n",
        b"data: \xff\n\n",
        _wire(_event()),
        _wire(_event(terminal=True)),
        _wire(_event(terminal=True)) + _wire(_event()),
        b'data: {"meta_info":{}}',
    ],
)
async def test_protocol_failures(wire):
    with pytest.raises(GenerateProtocolError):
        await _collect([wire])


async def test_bounded_event():
    for wire in [b"data: " + b"x" * 100, b"data:x\n" * 100]:
        with pytest.raises(GenerateProtocolError, match="exceeds"):
            await _collect([wire], max_event_bytes=32)


async def test_http_errors_and_content_type():
    with pytest.raises(httpx.HTTPStatusError):
        await _collect([], status=503)
    with pytest.raises(GenerateProtocolError, match="text/event-stream"):
        await _collect([], content_type="application/json")


async def test_cancellation_timeout_and_early_exit_close_response():
    for mode in ("cancel", "timeout", "break"):
        stream = _Stream([_wire(_event())], block=True)
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda request, stream=stream: httpx.Response(
                    200, headers={"content-type": "text/event-stream"}, stream=stream
                )
            )
        ) as client:

            async def consume(mode=mode):
                async with stream_generate(
                    client, url="http://frontend/generate", payload={}, timeout_seconds=0.05
                ) as events:
                    async for _event_value in events:
                        if mode == "break":
                            break

            if mode == "break":
                await consume()
            elif mode == "timeout":
                with pytest.raises(TimeoutError):
                    await consume()
            else:
                task = asyncio.create_task(consume())
                await stream.waiting.wait()
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
        assert stream.closed


@pytest.mark.parametrize(
    "options",
    [
        {"timeout_seconds": 0},
        {"timeout_seconds": float("inf")},
        {"max_event_bytes": 0},
        {"payload": {"stream": False}},
    ],
)
async def test_invalid_options(options):
    async with httpx.AsyncClient() as client:
        with pytest.raises(ValueError):
            async with stream_generate(
                client, **{"url": "http://unused/generate", "payload": {}, "timeout_seconds": 1, **options}
            ):
                pytest.fail("invalid options must fail before opening a request")


async def test_real_http_chunked_response():
    received = []
    finished = asyncio.Event()

    async def serve(reader, writer):
        try:
            request = await reader.readuntil(b"\r\n\r\n")
            received.append(request)
            writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n")
            wire = _wire(_event(terminal=True)) + b"data: [DONE]\n\n"
            for byte in wire:
                writer.write(b"1\r\n" + bytes([byte]) + b"\r\n")
                await writer.drain()
                await asyncio.sleep(0)
            writer.write(b"0\r\n\r\n")
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()
            finished.set()

    async with await asyncio.start_server(serve, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        async with httpx.AsyncClient() as client:
            async with stream_generate(
                client, url=f"http://127.0.0.1:{port}/generate", payload={"input_ids": [1]}, timeout_seconds=2
            ) as events:
                assert [value async for value in events] == [_event(terminal=True)]
        await asyncio.wait_for(finished.wait(), timeout=2)
    assert len(received) == 1 and received[0].startswith(b"POST /generate ")
