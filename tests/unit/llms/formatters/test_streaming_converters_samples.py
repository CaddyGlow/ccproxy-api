"""Sample-driven streaming converter checks for llm formatters."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest
from pydantic import TypeAdapter, ValidationError
from tests.helpers.sample_loader import load_sample

from ccproxy.llms.formatters.anthropic_to_openai import (
    convert__anthropic_message_to_openai_responses__stream,
)
from ccproxy.llms.formatters.context import register_request
from ccproxy.llms.formatters.openai_to_anthropic import (
    convert__openai_chat_to_anthropic_messages__stream,
    convert__openai_responses_to_anthropic_messages__stream,
)
from ccproxy.llms.models import anthropic as anthropic_models
from ccproxy.llms.models import openai as openai_models


async def _iter_events(events: list[Any]) -> AsyncIterator[Any]:
    for event in events:
        yield event


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
            getattr(part, "text", "")
            if hasattr(part, "text")
            else (part.get("text", "") if isinstance(part, dict) else "")
            for part in instructions
        )
    else:
        instructions_text = instructions or ""

    register_request(request_model, instructions_text)

    adapter: TypeAdapter[anthropic_models.MessageStreamEvent] = TypeAdapter(
        anthropic_models.MessageStreamEvent
    )
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
    content = getattr(message_output, "content", [])
    tool_blocks = [
        block
        for block in content
        if isinstance(block, dict) and block.get("type") == "tool_use"
    ]
    assert tool_blocks, "expected tool_use block in final response"
    tool_args = tool_blocks[0].get("arguments")
    assert isinstance(tool_args, dict)
    assert tool_args, "tool arguments should not be empty"

    register_request(None)


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
    # Per Anthropic streaming spec, tool_use.input is empty at start and
    # streamed via input_json_delta events.
    assert getattr(tool_event.content_block, "input", None) == {}
    input_delta = next(
        evt
        for evt in streamed
        if isinstance(evt, anthropic_models.ContentBlockDeltaEvent)
        and evt.index == tool_event.index
        and getattr(evt.delta, "type", None) == "input_json_delta"
    )
    assert getattr(input_delta.delta, "partial_json", ""), (
        "tool input_json_delta should carry the arguments JSON"
    )

    message_delta = next(
        evt for evt in streamed if isinstance(evt, anthropic_models.MessageDeltaEvent)
    )
    assert message_delta.delta.stop_reason == "tool_use"

    register_request(None)


@pytest.mark.asyncio
async def test_openai_responses_stream_emits_text_delta_type() -> None:
    """OpenAI Responses -> Anthropic streaming must emit ``text_delta`` on the wire.

    Regression for issue #51 follow-up: Claude Code CLI pointed at ccproxy's
    /codex endpoint received 200 OK with well-structured SSE chunks but the
    text never rendered, because the converter was emitting
    ``ContentBlockDeltaEvent(delta=TextBlock(type="text"))`` instead of
    ``TextDelta(type="text_delta")``. The Pydantic model accepts both, but the
    real Anthropic wire protocol (and the CLI's parser) requires ``text_delta``.
    """

    events: list[dict[str, Any]] = [
        {
            "type": "response.created",
            "sequence_number": 1,
            "response": {
                "id": "resp_1",
                "object": "response",
                "model": "gpt-5-codex",
                "created_at": 0,
                "status": "in_progress",
                "parallel_tool_calls": False,
                "output": [],
            },
        },
        {
            "type": "response.output_item.added",
            "sequence_number": 2,
            "output_index": 0,
            "item": {
                "type": "message",
                "id": "msg_1",
                "status": "in_progress",
                "role": "assistant",
                "content": [],
            },
        },
        {
            "type": "response.output_text.delta",
            "sequence_number": 3,
            "item_id": "msg_1",
            "output_index": 0,
            "content_index": 0,
            "delta": "Hello",
        },
        {
            "type": "response.output_text.delta",
            "sequence_number": 4,
            "item_id": "msg_1",
            "output_index": 0,
            "content_index": 0,
            "delta": "!",
        },
        {
            "type": "response.output_text.done",
            "sequence_number": 5,
            "item_id": "msg_1",
            "output_index": 0,
            "content_index": 0,
            "text": "Hello!",
        },
        {
            "type": "response.completed",
            "sequence_number": 6,
            "response": {
                "id": "resp_1",
                "object": "response",
                "model": "gpt-5-codex",
                "created_at": 0,
                "status": "completed",
                "parallel_tool_calls": False,
                "output": [
                    {
                        "type": "message",
                        "id": "msg_1",
                        "status": "completed",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "Hello!"}],
                    }
                ],
            },
        },
    ]

    streamed: list[anthropic_models.MessageStreamEvent] = []
    async for evt in convert__openai_responses_to_anthropic_messages__stream(
        _iter_events(events)
    ):
        streamed.append(evt)

    deltas = [
        evt
        for evt in streamed
        if isinstance(evt, anthropic_models.ContentBlockDeltaEvent)
    ]
    assert len(deltas) == 2, "expected two content_block_delta events"
    for delta_evt in deltas:
        assert delta_evt.delta.type == "text_delta", (
            f"delta.type must be 'text_delta' on the wire, got {delta_evt.delta.type!r}"
        )
        dumped = delta_evt.model_dump(mode="json", by_alias=True)
        assert dumped["delta"]["type"] == "text_delta"

    combined = "".join(
        getattr(evt.delta, "text", "") for evt in deltas if hasattr(evt.delta, "text")
    )
    assert combined == "Hello!"


@pytest.mark.asyncio
async def test_openai_chat_stream_emits_text_delta_type() -> None:
    """OpenAI Chat -> Anthropic streaming must also emit ``text_delta``."""

    events: list[dict[str, Any]] = [
        {
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "gpt-4",
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": "Hi"},
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "gpt-4",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": " there"},
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "gpt-4",
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                }
            ],
        },
    ]

    streamed: list[anthropic_models.MessageStreamEvent] = []
    async for evt in convert__openai_chat_to_anthropic_messages__stream(
        _iter_events(events)
    ):
        streamed.append(evt)

    deltas = [
        evt
        for evt in streamed
        if isinstance(evt, anthropic_models.ContentBlockDeltaEvent)
    ]
    assert deltas, "expected at least one content_block_delta"
    for delta_evt in deltas:
        assert delta_evt.delta.type == "text_delta"
        dumped = delta_evt.model_dump(mode="json", by_alias=True)
        assert dumped["delta"]["type"] == "text_delta"
