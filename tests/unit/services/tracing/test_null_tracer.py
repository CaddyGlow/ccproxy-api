"""Tests for the null request tracer implementation."""

import pytest

from ccproxy.services.tracing.null_tracer import NullRequestTracer


@pytest.mark.asyncio
async def test_null_tracer_methods_do_nothing():
    tracer = NullRequestTracer()

    await tracer.trace_request("id", "GET", "https://example", {}, None)
    await tracer.trace_response("id", 200, {}, b"body")
    await tracer.trace_stream_start("id", {})
    await tracer.trace_stream_chunk("id", b"chunk", 1)
    await tracer.trace_stream_complete("id", 3, 123)
