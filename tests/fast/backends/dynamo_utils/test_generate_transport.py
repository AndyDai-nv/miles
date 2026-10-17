import json

import pytest

from miles.backends.dynamo_utils.generate_transport import GenerateProtocolError, _generate_events

pytestmark = pytest.mark.asyncio


def _event(*, terminal=False):
    return {"text": "你好", "output_ids": [12], "meta_info": {"finish_reason": {"type": "stop"} if terminal else None}}


def _wire(value):
    return b"data: " + json.dumps(value, ensure_ascii=False).encode() + b"\n\n"


async def _collect(chunks, *, max_event_bytes=8 * 1024 * 1024):
    async def source():
        for chunk in chunks:
            yield chunk

    return [event async for event in _generate_events(source(), max_event_bytes=max_event_bytes)]


@pytest.mark.parametrize("split", [1, 2, 7, 4096])
async def test_chunk_boundaries(split):
    wire = b": heartbeat\r\n\r\n" + _wire(_event()) + _wire(_event(terminal=True)) + b"data: [DONE]\n\n"
    assert await _collect([wire[i : i + split] for i in range(0, len(wire), split)]) == [
        _event(),
        _event(terminal=True),
    ]


async def test_multiline_data_and_cr_delimiters():
    wire = b'event: message\rdata: {"meta_info":\rdata: {"finish_reason":{"type":"stop"}}}\r\rdata: [DONE]\r\r'
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
