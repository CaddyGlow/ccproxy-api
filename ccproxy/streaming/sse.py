"""Shared helpers for serializing JSON streams into SSE messages."""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator, AsyncIterator
from typing import Any

from ccproxy.core.logging import get_logger
from ccproxy.llms.streaming import AnthropicSSEFormatter


logger = get_logger(__name__)


async def serialize_json_to_sse_stream(
    json_stream: AsyncIterator[Any],
    *,
    include_done: bool = True,
    request_context: Any | None = None,
) -> AsyncGenerator[bytes, None]:
    """Serialize JSON-like stream items into SSE-compliant bytes.

    This matches the behaviour previously implemented in
    ``DeferredStreaming._serialize_json_to_sse_stream`` and is shared by
    SDK and HTTP-based providers alike.

    Args:
        json_stream: Async iterator yielding dict-like SSE payloads (or
            objects with ``model_dump``/``dict``).
        include_done: Whether to append the ``data: [DONE]`` sentinel at
            the end of the stream.
        request_context: Optional request context for logging (expects a
            ``request_id`` attribute when present).
    """

    formatter = AnthropicSSEFormatter()
    request_id = None
    if request_context and hasattr(request_context, "request_id"):
        request_id = request_context.request_id

    chunk_count = 0
    anthropic_chunks = 0
    openai_chunks = 0

    async for json_obj in json_stream:
        chunk_count += 1

        # Normalise the payload to a dict
        if hasattr(json_obj, "model_dump") and callable(json_obj.model_dump):
            json_obj = json_obj.model_dump()
        elif hasattr(json_obj, "dict") and callable(json_obj.dict):
            json_obj = json_obj.dict()

        if not isinstance(json_obj, dict):
            # Skip unsupported payloads
            logger.debug(
                "sse_serialization_skipped_non_dict",
                chunk_number=chunk_count,
                payload_type=type(json_obj).__name__,
                request_id=request_id,
                category="sse_format",
            )
            continue

        event_type = json_obj.get("type")
        if isinstance(event_type, str) and event_type:
            anthropic_chunks += 1
            if event_type == "ping":
                sse_event = formatter.format_ping()
            else:
                sse_event = formatter.format_event(event_type, json_obj)

            logger.trace(
                "sse_serialization_anthropic_format",
                event_type=event_type,
                chunk_number=chunk_count,
                request_id=request_id,
                category="sse_format",
            )
        else:
            openai_chunks += 1
            json_str = json.dumps(json_obj, ensure_ascii=False)
            sse_event = f"data: {json_str}\n\n"

            logger.trace(
                "sse_serialization_openai_format",
                chunk_number=chunk_count,
                has_choices=bool(json_obj.get("choices")),
                request_id=request_id,
                category="sse_format",
            )

        yield sse_event.encode("utf-8")

    logger.debug(
        "sse_serialization_complete",
        total_chunks=chunk_count,
        anthropic_chunks=anthropic_chunks,
        openai_chunks=openai_chunks,
        request_id=request_id,
        category="sse_format",
    )

    if include_done:
        yield b"data: [DONE]\n\n"


__all__ = ["serialize_json_to_sse_stream"]
