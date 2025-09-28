"""Integration coverage for streaming formatters using recorded samples."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest
from pydantic import TypeAdapter, ValidationError

from ccproxy.llms.formatters.anthropic_to_openai import (
    convert__anthropic_message_to_openai_responses__stream,
)
from ccproxy.llms.formatters.context import register_request
from ccproxy.llms.formatters.openai_to_anthropic import (
    convert__openai_chat_to_anthropic_messages__stream,
)
from ccproxy.llms.models import anthropic as anthropic_models
from ccproxy.llms.models import openai as openai_models
from tests.helpers.sample_loader import load_sample


async def _iter_events(events: list[Any]) -> AsyncIterator[Any]:
    for event in events:
        yield event


@pytest.mark.integration
@pytest.mark.asyncio
async def test_claude_stream_to_openai_responses_sample() -> None:
    """Ensure Anthropic streaming sample converts to OpenAI Responses events."""

    sample = load_sample("claude_messages_tools_stream")

    request_payload = sample["request"].get("payload", {})
    request_model = anthropic_models.CreateMessageRequest.model_validate(
        request_payload
    )
    instructions = request_model.system
    if isinstance(instructions, list):
        instructions_text = "\n".join(
            part.get("text", "") for part in instructions if isinstance(part, dict)
        )
    else:
        instructions_text = instructions or ""

    register_request(request_model, instructions_text)

    adapter = TypeAdapter(anthropic_models.MessageStreamEvent)
    events: list[Any] = []
    for raw_event in sample["response"].get("events", []):
        payload = raw_event.get("json")
        if not payload:
            continue
        try:
            events.append(adapter.validate_python(payload))
        except ValidationError:
            events.append(payload)

    streamed: list[openai_models.StreamEventType] = []
    async for evt in convert__anthropic_message_to_openai_responses__stream(
        _iter_events(events)
    ):
        streamed.append(evt)

    assert streamed, "expected streamed OpenAI events"
    event_types = [getattr(evt, "type", None) for evt in streamed]
    assert "response.function_call_arguments.delta" in event_types
    assert event_types[-1] == "response.completed"

    completed = streamed[-1]
    assert isinstance(completed, openai_models.ResponseCompletedEvent)
    response = completed.response
    assert response.usage is not None
    if instructions_text:
        assert response.instructions == instructions_text

    message_output = response.output[0]
    tool_blocks = [
        block
        for block in message_output.content  # type: ignore[assignment]
        if isinstance(block, dict) and block.get("type") == "tool_use"
    ]
    assert tool_blocks, "expected tool_use block in final response"
    tool_args = tool_blocks[0].get("arguments")
    assert isinstance(tool_args, dict)
    assert tool_args, "tool arguments should not be empty"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_openai_chat_stream_to_anthropic_sample() -> None:
    """Ensure OpenAI chat streaming sample converts to Anthropic events."""

    sample = load_sample("copilot_chat_completions_tools_stream")

    events: list[dict[str, Any]] = [
        raw_event.get("json", {})
        for raw_event in sample["response"].get("events", [])
        if raw_event.get("json")
    ]

    streamed = [
        evt
        async for evt in convert__openai_chat_to_anthropic_messages__stream(
            _iter_events(events)
        )
    ]

    assert streamed, "expected Anthropic events"
    event_types = [getattr(evt, "type", None) for evt in streamed]
    assert event_types[0] == "message_start"
    assert event_types[-1] == "message_stop"

    tool_event = next(
        evt
        for evt in streamed
        if isinstance(evt, anthropic_models.ContentBlockStartEvent)
        and getattr(evt.content_block, "type", None) == "tool_use"
    )
    assert tool_event.content_block.input, "tool input should be populated"

    message_delta = next(
        evt for evt in streamed if isinstance(evt, anthropic_models.MessageDeltaEvent)
    )
    assert message_delta.delta.stop_reason == "tool_use"
