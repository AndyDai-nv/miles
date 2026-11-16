import json
import re
from collections.abc import AsyncIterable, AsyncIterator
from typing import Any

_NEWLINE = re.compile(b"\r\n|\r|\n")


class GenerateProtocolError(ValueError):
    pass


async def _generate_events(chunks: AsyncIterable[bytes], *, max_event_bytes: int) -> AsyncIterator[dict[str, Any]]:
    terminal = False
    async for event, data in _sse_events(chunks, max_event_bytes=max_event_bytes):
        if event == "error":
            raise GenerateProtocolError("Dynamo returned an SSE error event")
        if data == "[DONE]":
            if not terminal:
                raise GenerateProtocolError("stream ended without a terminal response")
            return
        if terminal:
            raise GenerateProtocolError("received data after a terminal response")
        try:
            value = json.loads(data)
        except ValueError as error:
            raise GenerateProtocolError("invalid SSE JSON") from error
        if not isinstance(value, dict) or "error" in value:
            raise GenerateProtocolError("expected a native generation response, not an error")
        meta = value.get("meta_info")
        if not isinstance(meta, dict):
            raise GenerateProtocolError("response is missing meta_info")
        terminal = meta.get("finish_reason") is not None
        yield value
    raise GenerateProtocolError("stream closed before [DONE]")


async def _sse_events(chunks: AsyncIterable[bytes], *, max_event_bytes: int) -> AsyncIterator[tuple[str, str]]:
    buffer = b""
    data: list[bytes] = []
    event = b""
    size = 0
    async for chunk in chunks:
        buffer += chunk
        while (match := _NEWLINE.search(buffer)) is not None:
            # A CR at the chunk boundary may be the first half of CRLF.
            if match[0] == b"\r" and match.end() == len(buffer):
                break
            line, buffer = buffer[: match.start()], buffer[match.end() :]
            size += len(line) + len(match[0])
            if size > max_event_bytes:
                raise GenerateProtocolError("SSE event exceeds max_event_bytes")
            if not line:
                if data:
                    try:
                        yield event.decode("utf-8"), b"\n".join(data).decode("utf-8")
                    except UnicodeDecodeError as error:
                        raise GenerateProtocolError("invalid SSE UTF-8") from error
                data, event, size = [], b"", 0
            else:
                field, separator, value = line.partition(b":")
                value = value.removeprefix(b" ") if separator else b""
                if field == b"data":
                    data.append(value)
                elif field == b"event":
                    event = value
            if size + len(buffer) > max_event_bytes and _NEWLINE.search(buffer) is None:
                raise GenerateProtocolError("SSE event exceeds max_event_bytes")
        if size + len(buffer) > max_event_bytes:
            raise GenerateProtocolError("SSE event exceeds max_event_bytes")
    if buffer == b"\r":
        if data:
            try:
                yield event.decode("utf-8"), b"\n".join(data).decode("utf-8")
            except UnicodeDecodeError as error:
                raise GenerateProtocolError("invalid SSE UTF-8") from error
        buffer, data = b"", []
    if buffer or data:
        raise GenerateProtocolError("incomplete SSE event")
