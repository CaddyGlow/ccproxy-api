import json
from collections.abc import AsyncIterator
from typing import Any

import pytest

from ccproxy.llms.formatters.anthropic_to_openai import (
    convert__anthropic_message_to_openai_chat__response,
    convert__anthropic_message_to_openai_responses__request,
    convert__anthropic_message_to_openai_responses__stream,
)
from ccproxy.llms.formatters.context import register_request
from ccproxy.llms.models import anthropic as anthropic_models
from ccproxy.llms.models import openai as openai_models


@pytest.mark.asyncio
async def test_convert__anthropic_message_to_openai_chat__response_basic() -> None:
    resp = anthropic_models.MessageResponse(
        id="msg_1",
        type="message",
        role="assistant",
        model="claude-3",
        content=[anthropic_models.TextBlock(type="text", text="Hello")],
        stop_reason="end_turn",
        stop_sequence=None,
        usage=anthropic_models.Usage(input_tokens=1, output_tokens=2),
    )

    out = convert__anthropic_message_to_openai_chat__response(resp)
    assert isinstance(out, openai_models.ChatCompletionResponse)
    assert out.object == "chat.completion"
    assert out.choices and out.choices[0].message.content == "Hello"
    assert out.choices[0].finish_reason == "stop"
    assert out.usage is not None
    assert out.usage.total_tokens == 3


@pytest.mark.asyncio
async def test_convert__anthropic_message_to_openai_responses__stream_minimal() -> None:
    register_request(
        anthropic_models.CreateMessageRequest(
            model="claude-3",
            system="system instructions",
            max_tokens=32,
            messages=[anthropic_models.Message(role="user", content="Hi")],
        ),
        "system instructions",
    )

    async def gen():
        yield anthropic_models.MessageStartEvent(
            type="message_start",
            message=anthropic_models.MessageResponse(
                id="m1",
                type="message",
                role="assistant",
                model="claude-3",
                content=[],
                stop_reason=None,
                stop_sequence=None,
                usage=anthropic_models.Usage(input_tokens=0, output_tokens=0),
            ),
        )
        yield anthropic_models.ContentBlockStartEvent(
            type="content_block_start",
            index=0,
            content_block=anthropic_models.TextBlock(type="text", text=""),
        )
        yield anthropic_models.ContentBlockDeltaEvent(
            type="content_block_delta",
            delta=anthropic_models.TextBlock(type="text", text="Hi"),
            index=0,
        )
        yield anthropic_models.ContentBlockStopEvent(type="content_block_stop", index=0)
        yield anthropic_models.MessageDeltaEvent(
            type="message_delta",
            delta=anthropic_models.MessageDelta(stop_reason="end_turn"),
            usage=anthropic_models.Usage(input_tokens=1, output_tokens=2),
        )
        yield anthropic_models.MessageStopEvent(type="message_stop")

    chunks = []
    async for evt in convert__anthropic_message_to_openai_responses__stream(gen()):
        chunks.append(evt)

    # Expect the expanded Responses streaming lifecycle ordering
    types = [getattr(e, "type", None) for e in chunks]
    assert types == [
        "response.created",
        "response.in_progress",
        "response.output_item.added",
        "response.content_part.added",
        "response.output_text.delta",
        "response.in_progress",
        "response.output_text.done",
        "response.content_part.done",
        "response.output_item.done",
        "response.completed",
    ]

    text_deltas = [
        getattr(evt, "delta", "")
        for evt in chunks
        if getattr(evt, "type", "") == "response.output_text.delta"
    ]
    assert text_deltas == ["Hi"]

    done_event = next(
        evt for evt in chunks if getattr(evt, "type", "") == "response.output_text.done"
    )
    assert getattr(done_event, "text", "") == "Hi"

    completed = chunks[-1]
    assert getattr(completed, "type", "") == "response.completed"
    completed_response = completed.response  # type: ignore[union-attr]
    assert completed_response.output
    message = completed_response.output[0]
    content = getattr(message, "content", None)
    assert content and getattr(content[0], "text", "") == "Hi"
    assert completed_response.usage is not None
    assert completed_response.instructions == "system instructions"

    created = chunks[0]
    created_response = created.response  # type: ignore[union-attr]
    assert getattr(created_response, "background", None) is None
    assert created_response.parallel_tool_calls is True


@pytest.mark.asyncio
async def test_convert__anthropic_message_to_openai_responses__stream_with_thinking_and_tool() -> (
    None
):
    register_request(
        anthropic_models.CreateMessageRequest(
            model="claude-3-opus",
            system="assistant system",
            max_tokens=128,
            messages=[anthropic_models.Message(role="user", content="lookup weather")],
        ),
        "assistant system",
    )

    async def gen() -> AsyncIterator[Any]:
        yield anthropic_models.MessageStartEvent(
            type="message_start",
            message=anthropic_models.MessageResponse(
                id="m-thinking-tool",
                type="message",
                role="assistant",
                model="claude-3-opus",
                content=[],
                stop_reason=None,
                stop_sequence=None,
                usage=anthropic_models.Usage(input_tokens=0, output_tokens=0),
            ),
        )
        yield anthropic_models.ContentBlockStartEvent(
            type="content_block_start",
            index=0,
            content_block=anthropic_models.ThinkingBlock(
                thinking="", signature="sig-123"
            ),
        )
        yield anthropic_models.ContentBlockDeltaEvent(
            type="content_block_delta",
            index=0,
            delta=anthropic_models.ThinkingDelta(
                type="thinking_delta",
                thinking="Analyzing request",
            ),
        )
        yield anthropic_models.ContentBlockStopEvent(type="content_block_stop", index=0)
        yield anthropic_models.ContentBlockStartEvent(
            type="content_block_start",
            index=1,
            content_block=anthropic_models.ToolUseBlock(
                id="tool_1",
                name="get_weather",
                input={},
            ),
        )
        yield anthropic_models.ContentBlockDeltaEvent(
            type="content_block_delta",
            index=1,
            delta=anthropic_models.InputJsonDelta(
                type="input_json_delta",
                partial_json='{"location":"seattle',
            ),
        )
        yield anthropic_models.ContentBlockDeltaEvent(
            type="content_block_delta",
            index=1,
            delta=anthropic_models.InputJsonDelta(
                type="input_json_delta",
                partial_json='","units":"metric"}',
            ),
        )
        yield anthropic_models.ContentBlockStopEvent(type="content_block_stop", index=1)
        yield anthropic_models.MessageDeltaEvent(
            type="message_delta",
            delta=anthropic_models.MessageDelta(stop_reason="tool_use"),
            usage=anthropic_models.Usage(input_tokens=11, output_tokens=7),
        )
        yield anthropic_models.MessageStopEvent(type="message_stop")

    events: list[openai_models.StreamEventType] = []
    async for evt in convert__anthropic_message_to_openai_responses__stream(gen()):
        events.append(evt)

    event_types = [getattr(evt, "type", None) for evt in events]
    assert event_types.count("response.output_item.added") == 2
    assert event_types.count("response.in_progress") >= 1
    assert "response.reasoning_summary_text.delta" in event_types
    assert "response.reasoning_summary_text.done" in event_types
    assert "response.function_call_arguments.delta" in event_types
    assert "response.function_call_arguments.done" in event_types

    reasoning_deltas = [
        getattr(evt, "delta", "")
        for evt in events
        if getattr(evt, "type", "") == "response.reasoning_summary_text.delta"
    ]
    assert reasoning_deltas == ["Analyzing request"]

    complete_event = next(
        evt for evt in events if getattr(evt, "type", "") == "response.completed"
    )
    complete_response = complete_event.response  # type: ignore[union-attr]
    assert complete_response.usage is not None
    assert complete_response.usage.input_tokens == 11
    assert complete_response.usage.output_tokens == 7
    assert complete_response.instructions == "assistant system"

    reasoning_output = next(
        out
        for out in complete_response.output
        if getattr(out, "type", "") == "reasoning"
    )
    summary = getattr(reasoning_output, "summary", [])
    assert summary and summary[0]["text"] == "Analyzing request"  # type: ignore[comparison-overlap]
    assert summary[0]["signature"] == "sig-123"  # type: ignore[comparison-overlap]

    function_output = next(
        out
        for out in complete_response.output
        if getattr(out, "type", "") == "function_call"
    )
    assert getattr(function_output, "name", "") == "get_weather"
    assert (
        getattr(function_output, "arguments", "")
        == '{"location":"seattle","units":"metric"}'
    )

    tool_calls = getattr(complete_response, "tool_calls", []) or []
    assert tool_calls
    parsed_arguments = json.loads(tool_calls[0]["function"]["arguments"])
    assert parsed_arguments == {"location": "seattle", "units": "metric"}

    message_outputs = [
        out
        for out in getattr(complete_response, "output", [])
        if getattr(out, "type", "") == "message"
    ]
    if message_outputs:
        assert not getattr(message_outputs[0], "content", None)


@pytest.mark.asyncio
async def test_convert__anthropic_message_to_openai_responses__request_basic() -> None:
    req = anthropic_models.CreateMessageRequest(
        model="claude-3",
        system="sys",
        messages=[anthropic_models.Message(role="user", content="Hi")],
        max_tokens=256,
        stream=True,
    )

    out = convert__anthropic_message_to_openai_responses__request(req)
    resp_req = openai_models.ResponseRequest.model_validate(out)
    assert resp_req.model == "claude-3"
    assert resp_req.max_output_tokens == 256
    assert resp_req.stream is True
    assert resp_req.instructions == "sys"
    assert isinstance(resp_req.input, list) and resp_req.input
