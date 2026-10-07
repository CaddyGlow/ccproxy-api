"""Tests for streaming error payload normalization."""

from __future__ import annotations

import httpx
import pytest

from ccproxy.core.constants import FORMAT_ANTHROPIC_MESSAGES, FORMAT_OPENAI_CHAT
from ccproxy.core.logging import get_logger
from ccproxy.core.request_context import RequestContext
from ccproxy.streaming.deferred import DeferredStreaming


@pytest.mark.anyio
async def test_anthropic_error_wrapped_with_type() -> None:
    ctx = RequestContext(
        request_id="req_1",
        start_time=0.0,
        logger=get_logger(__name__),
    )
    ctx.format_chain = [FORMAT_ANTHROPIC_MESSAGES]

    client = httpx.AsyncClient()
    stream = DeferredStreaming(
        method="POST",
        url="http://example.test/v1/messages",
        headers={},
        body=b"{}",
        client=client,
        request_context=ctx,
    )

    error_obj = {"error": {"type": "timeout_error", "message": "Request timeout"}}
    formatted = stream._format_stream_error(error_obj)

    assert formatted["type"] == "error"
    assert formatted["error"]["type"] == "timeout_error"

    await client.aclose()


@pytest.mark.anyio
async def test_non_anthropic_error_left_unchanged() -> None:
    ctx = RequestContext(
        request_id="req_2",
        start_time=0.0,
        logger=get_logger(__name__),
    )
    ctx.format_chain = [FORMAT_OPENAI_CHAT]

    client = httpx.AsyncClient()
    stream = DeferredStreaming(
        method="POST",
        url="http://example.test/v1/chat/completions",
        headers={},
        body=b"{}",
        client=client,
        request_context=ctx,
    )

    error_obj = {"error": {"type": "timeout_error", "message": "Request timeout"}}
    formatted = stream._format_stream_error(error_obj)

    assert formatted == error_obj

    await client.aclose()
