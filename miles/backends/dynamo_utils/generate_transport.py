import asyncio
import json
import math
import re
from collections.abc import AsyncIterable, AsyncIterator, Mapping
from contextlib import aclosing, asynccontextmanager
from typing import Any

import httpx

_NEWLINE = re.compile(b"\r\n|\r|\n")


class GenerateProtocolError(ValueError):
    pass


@asynccontextmanager
async def stream_generate(
    client: httpx.AsyncClient,
    *,
    url: str,
    payload: Mapping[str, Any],
    timeout_seconds: float,
    headers: Mapping[str, str] | None = None,
    max_event_bytes: int = 8 * 1024 * 1024,
) -> AsyncIterator[AsyncIterator[dict[str, Any]]]:
    """Consume inside async with; closing HTTP is not proof that the engine drained."""
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be finite and positive")
    if type(max_event_bytes) is not int or max_event_bytes <= 0:
        raise ValueError("max_event_bytes must be a positive integer")
    if "stream" in payload and payload["stream"] is not True:
        raise ValueError("Dynamo native generate requires stream=true")
    # Streaming is a transport constraint, not a caller-overridable sampling option.
    body = {**payload, "stream": True}
    async with asyncio.timeout(timeout_seconds):
        async with client.stream("POST", url, json=body, headers=headers, timeout=timeout_seconds) as response:
            response.raise_for_status()
            if response.headers.get("content-type", "").split(";")[0].strip().lower() != "text/event-stream":
                raise GenerateProtocolError("expected text/event-stream")
            async with aclosing(_generate_events(response.aiter_bytes(), max_event_bytes=max_event_bytes)) as events:
                yield events


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
