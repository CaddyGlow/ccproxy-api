"""Tests for the mock response handler."""

import asyncio

import pytest

from ccproxy.core.request_context import RequestContext
from ccproxy.services.mocking.mock_handler import MockResponseHandler


class DummyGenerator:
    def generate_tool_use_response(self, model=None):
        return {"content": [{"text": "tool"}]}

    def generate_long_response(self, model=None):
        return {"content": [{"text": "long response"}]}

    def generate_medium_response(self, model=None):
        return {"content": [{"text": "medium"}]}

    def generate_short_response(self, model=None):
        return {"content": [{"text": "short"}]}


@pytest.mark.parametrize(
    "body,expected",
    [
        (b"", "short"),
        (b'{"tools": []}', "tool_use"),
        (('{"messages": [{"content": "' + "x" * 300 + '"}]}').encode(), "medium"),
        (('{"messages": [{"content": "' + "x" * 1200 + '"}]}').encode(), "long"),
    ],
)
def test_extract_message_type(body, expected):
    handler = MockResponseHandler(DummyGenerator())
    assert handler.extract_message_type(body) == expected


@pytest.mark.asyncio
async def test_generate_standard_response_success(monkeypatch):
    handler = MockResponseHandler(DummyGenerator(), error_rate=0.0)
    monkeypatch.setattr(handler, "should_simulate_error", lambda: False)
    monkeypatch.setattr("random.uniform", lambda *args, **kwargs: 0)

    async def fast_sleep(_):
        return None

    monkeypatch.setattr(asyncio, "sleep", fast_sleep)

    ctx = RequestContext(request_id="req", start_time=0, logger=None)
    status, headers, body = await handler.generate_standard_response(
        model="m1", is_openai_format=False, ctx=ctx, message_type="short"
    )

    assert status == 200
    assert headers["content-type"] == "application/json"
    assert b"short" in body
    assert ctx.metrics["mock_response_type"] == "short"


@pytest.mark.asyncio
async def test_generate_standard_response_error(monkeypatch):
    handler = MockResponseHandler(DummyGenerator(), error_rate=1.0)
    monkeypatch.setattr(handler, "should_simulate_error", lambda: True)

    async def fast_sleep(_):
        return None

    monkeypatch.setattr(asyncio, "sleep", fast_sleep)

    status, headers, body = await handler.generate_standard_response(
        model="m1", is_openai_format=True, ctx=None, message_type="short"
    )

    assert status == 429
    assert b"error" in body


@pytest.mark.asyncio
async def test_generate_streaming_response(monkeypatch):
    handler = MockResponseHandler(DummyGenerator(), error_rate=0.0)
    ctx = RequestContext(request_id="req", start_time=0, logger=None)

    stream = await handler.generate_streaming_response(
        model="m1", is_openai_format=True, ctx=ctx
    )

    chunks = []
    async for chunk in stream.body_iterator:
        chunks.append(chunk)

    assert any(b"[DONE]" in chunk for chunk in chunks)
