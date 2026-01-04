"""Tests for the simple streaming adapter."""

import pytest

from ccproxy.streaming.simple_adapter import SimpleStreamingAdapter


@pytest.mark.asyncio
async def test_simple_adapter_passes_through_data():
    adapter = SimpleStreamingAdapter(name="test")

    request = {"prompt": "hi"}
    response = {"completion": "hello"}
    error = {"error": "oops"}

    assert await adapter.adapt_request(request) is request
    assert await adapter.adapt_response(response) is response
    assert await adapter.adapt_error(error) is error

    async def source():
        for i in range(3):
            yield {"chunk": i}

    chunks = []
    async for chunk in adapter.adapt_stream(source()):
        chunks.append(chunk)

    assert chunks == [{"chunk": 0}, {"chunk": 1}, {"chunk": 2}]
