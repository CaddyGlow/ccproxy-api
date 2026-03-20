"""Tests for the mock response handler."""

import asyncio
import json
from collections.abc import Sequence
from unittest.mock import MagicMock

import pytest
from pydantic import TypeAdapter

from ccproxy.core.constants import (
    FORMAT_ANTHROPIC_MESSAGES,
    FORMAT_OPENAI_CHAT,
    FORMAT_OPENAI_RESPONSES,
)
from ccproxy.core.request_context import RequestContext
from ccproxy.llms.models import openai as openai_models
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


def _parse_sse_events(
    chunks: Sequence[bytes | str | memoryview],
) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    for chunk in chunks:
        if isinstance(chunk, memoryview):
            decoded = chunk.tobytes().decode()
        elif isinstance(chunk, bytes):
            decoded = chunk.decode()
        else:
            decoded = chunk
        for line in decoded.splitlines():
            if not line.startswith("data: "):
                continue
            payload = line[6:].strip()
            if not payload or payload == "[DONE]":
                continue
            event = json.loads(payload)
            if isinstance(event, dict):
                events.append(event)
    return events


@pytest.mark.parametrize(
    "body,expected",
    [
        (b"", "short"),
        (b'{"tools": []}', "tool_use"),
        (('{"messages": [{"content": "' + "x" * 300 + '"}]}').encode(), "medium"),
        (('{"messages": [{"content": "' + "x" * 1200 + '"}]}').encode(), "long"),
    ],
)
def test_extract_message_type(body: bytes, expected: str) -> None:
    handler = MockResponseHandler(DummyGenerator())  # type: ignore[arg-type]
    assert handler.extract_message_type(body) == expected


def test_extract_prompt_text_collects_nested_values() -> None:
    handler = MockResponseHandler(DummyGenerator())  # type: ignore[arg-type]
    body = json.dumps(
        {
            "instructions": "Top level instructions",
            "input": [
                {
                    "content": [
                        {"text": "First prompt"},
                        {"text": "Second prompt"},
                    ]
                }
            ],
        }
    ).encode()

    assert handler.extract_prompt_text(body) == (
        "Top level instructions\nFirst prompt\nSecond prompt"
    )


def test_extract_prompt_text_limits_deep_nesting() -> None:
    handler = MockResponseHandler(DummyGenerator())  # type: ignore[arg-type]
    nested: dict[str, object] = {"text": "too deep"}
    for _ in range(12):
        nested = {"input": [nested]}

    body = json.dumps(nested).encode()

    assert handler.extract_prompt_text(body) == ""


@pytest.mark.asyncio
async def test_generate_standard_response_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handler = MockResponseHandler(DummyGenerator(), error_rate=0.0)  # type: ignore[arg-type]
    monkeypatch.setattr(handler, "should_simulate_error", lambda: False)
    monkeypatch.setattr("random.uniform", lambda *args, **kwargs: 0)

    async def fast_sleep(_: float) -> None:
        return None

    monkeypatch.setattr(asyncio, "sleep", fast_sleep)

    mock_logger = MagicMock()
    ctx = RequestContext(request_id="req", start_time=0, logger=mock_logger)  # type: ignore[arg-type]
    status, headers, body = await handler.generate_standard_response(
        model="m1",
        target_format=FORMAT_ANTHROPIC_MESSAGES,
        ctx=ctx,
        message_type="short",
    )

    assert status == 200
    assert headers["content-type"] == "application/json"
    assert b"short" in body
    assert ctx.metrics["mock_response_type"] == "short"


@pytest.mark.asyncio
async def test_generate_standard_response_does_not_special_case_login_prompts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handler = MockResponseHandler(DummyGenerator(), error_rate=0.0)  # type: ignore[arg-type]
    monkeypatch.setattr(handler, "should_simulate_error", lambda: False)
    monkeypatch.setattr("random.uniform", lambda *args, **kwargs: 0)

    async def fast_sleep(_: float) -> None:
        return None

    monkeypatch.setattr(asyncio, "sleep", fast_sleep)

    mock_logger = MagicMock()
    ctx = RequestContext(request_id="req", start_time=0, logger=mock_logger)  # type: ignore[arg-type]
    status, _, body = await handler.generate_standard_response(
        model="m1",
        target_format=FORMAT_ANTHROPIC_MESSAGES,
        ctx=ctx,
        message_type="short",
        prompt_text="Write requirements for a login form.",
    )

    assert status == 200
    assert b"short" in body
    assert b"Login Form Requirements" not in body


@pytest.mark.asyncio
async def test_generate_standard_response_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handler = MockResponseHandler(DummyGenerator(), error_rate=1.0)  # type: ignore[arg-type]
    monkeypatch.setattr(handler, "should_simulate_error", lambda: True)

    async def fast_sleep(_: float) -> None:
        return None

    monkeypatch.setattr(asyncio, "sleep", fast_sleep)

    mock_logger = MagicMock()
    mock_ctx = RequestContext(request_id="req", start_time=0, logger=mock_logger)  # type: ignore[arg-type]
    status, headers, body = await handler.generate_standard_response(
        model="m1",
        target_format=FORMAT_OPENAI_CHAT,
        ctx=mock_ctx,
        message_type="short",
    )

    assert status == 429
    assert b"error" in body


@pytest.mark.asyncio
async def test_generate_streaming_response(monkeypatch: pytest.MonkeyPatch) -> None:
    handler = MockResponseHandler(DummyGenerator(), error_rate=0.0)  # type: ignore[arg-type]
    mock_logger = MagicMock()
    ctx = RequestContext(request_id="req", start_time=0, logger=mock_logger)  # type: ignore[arg-type]

    stream = await handler.generate_streaming_response(
        model="m1", target_format=FORMAT_OPENAI_CHAT, ctx=ctx
    )

    chunks = []
    async for chunk in stream.body_iterator:
        chunks.append(chunk)

    assert any(b"[DONE]" in chunk for chunk in chunks)


@pytest.mark.asyncio
async def test_generate_responses_streaming_response_emits_valid_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handler = MockResponseHandler(DummyGenerator(), error_rate=0.0)  # type: ignore[arg-type]
    mock_logger = MagicMock()
    ctx = RequestContext(request_id="req", start_time=0, logger=mock_logger)  # type: ignore[arg-type]

    async def fast_sleep(_: float) -> None:
        return None

    monkeypatch.setattr(asyncio, "sleep", fast_sleep)

    stream = await handler.generate_streaming_response(
        model="m1",
        target_format=FORMAT_OPENAI_RESPONSES,
        ctx=ctx,
    )

    chunks = []
    async for chunk in stream.body_iterator:
        chunks.append(chunk)

    events = _parse_sse_events(chunks)
    validator = TypeAdapter(openai_models.AnyStreamEvent)

    assert [event["type"] for event in events] == [
        "response.created",
        "response.output_item.added",
        "response.content_part.added",
        "response.output_text.delta",
        "response.output_text.done",
        "response.content_part.done",
        "response.output_item.done",
        "response.completed",
    ]
    for event in events:
        validator.validate_python(event)
