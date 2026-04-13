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
async def test_convert__anthropic_message_to_openai_chat__response_tool_use() -> None:
    resp = anthropic_models.MessageResponse(
        id="msg_tool_1",
        type="message",
        role="assistant",
        model="claude-3",
        content=[
            anthropic_models.ToolUseBlock(
                type="tool_use",
                id="tool_123",
                name="get_weather",
                input={"location": "Boston", "units": "metric"},
            )
        ],
        stop_reason="tool_use",
        stop_sequence=None,
        usage=anthropic_models.Usage(input_tokens=3, output_tokens=4),
    )

    out = convert__anthropic_message_to_openai_chat__response(resp)
    assert isinstance(out, openai_models.ChatCompletionResponse)
    assert out.choices[0].finish_reason == "tool_calls"
    assert out.choices[0].message.content is None

    tool_calls = out.choices[0].message.tool_calls
    assert tool_calls and len(tool_calls) == 1
    tool_call = tool_calls[0]
    assert tool_call.id == "tool_123"
    assert tool_call.function.name == "get_weather"
    assert json.loads(tool_call.function.arguments) == {
        "location": "Boston",
        "units": "metric",
    }


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


@pytest.mark.asyncio
async def test_convert__anthropic_message_to_openai_responses__request_tool_cycle() -> (
    None
):
    req = anthropic_models.CreateMessageRequest(
        model="claude-3",
        messages=[
            anthropic_models.Message(role="user", content="list files"),
            anthropic_models.Message(
                role="assistant",
                content=[
                    anthropic_models.TextBlock(type="text", text="running ls"),
                    anthropic_models.ToolUseBlock(
                        type="tool_use",
                        id="call_1",
                        name="shell",
                        input={"command": ["ls"]},
                    ),
                ],
            ),
            anthropic_models.Message(
                role="user",
                content=[
                    anthropic_models.ToolResultBlock(
                        type="tool_result",
                        tool_use_id="call_1",
                        content="a.txt\nb.txt",
                    )
                ],
            ),
        ],
        max_tokens=64,
    )

    out = convert__anthropic_message_to_openai_responses__request(req)

    assert isinstance(out.input, list)
    types = [item.get("type") for item in out.input]
    assert types == ["message", "message", "function_call", "function_call_output"]

    user_msg, assistant_msg, fn_call, fn_out = out.input
    assert user_msg["role"] == "user"
    assert user_msg["content"][0]["type"] == "input_text"
    assert user_msg["content"][0]["text"] == "list files"

    assert assistant_msg["role"] == "assistant"
    assert assistant_msg["content"][0]["type"] == "output_text"
    assert assistant_msg["content"][0]["text"] == "running ls"

    assert fn_call["call_id"] == "call_1"
    assert fn_call["name"] == "shell"
    assert json.loads(fn_call["arguments"]) == {"command": ["ls"]}

    assert fn_out["call_id"] == "call_1"
    assert fn_out["output"] == "a.txt\nb.txt"


@pytest.mark.asyncio
async def test_convert__anthropic_message_to_openai_responses__request_long_call_id() -> (
    None
):
    long_id = "fc_" + ("0123456789abcdef" * 8)  # 131 chars
    assert len(long_id) > 64

    req = anthropic_models.CreateMessageRequest(
        model="claude-3",
        messages=[
            anthropic_models.Message(role="user", content="run"),
            anthropic_models.Message(
                role="assistant",
                content=[
                    anthropic_models.ToolUseBlock(
                        type="tool_use",
                        id=long_id,
                        name="shell",
                        input={},
                    ),
                ],
            ),
            anthropic_models.Message(
                role="user",
                content=[
                    anthropic_models.ToolResultBlock(
                        type="tool_result",
                        tool_use_id=long_id,
                        content="done",
                    )
                ],
            ),
        ],
        max_tokens=64,
    )

    out = convert__anthropic_message_to_openai_responses__request(req)
    assert isinstance(out.input, list)
    fn_call = next(item for item in out.input if item.get("type") == "function_call")
    fn_out = next(
        item for item in out.input if item.get("type") == "function_call_output"
    )
    assert fn_call["call_id"] == fn_out["call_id"]
    assert len(fn_call["call_id"]) <= 64
    assert fn_call["call_id"] != long_id


@pytest.mark.asyncio
async def test_convert__anthropic_message_to_openai_responses__request_legacy_custom_tools() -> (
    None
):
    req = anthropic_models.CreateMessageRequest(
        model="claude-3",
        messages=[anthropic_models.Message(role="user", content="run ls")],
        max_tokens=64,
        tools=[
            anthropic_models.LegacyCustomTool(
                type="custom",
                name="Bash",
                description="Run a shell command",
                input_schema={
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"],
                },
            )
        ],
    )

    out = convert__anthropic_message_to_openai_responses__request(req)
    assert out.tools is not None
    assert len(out.tools) == 1
    assert out.tools[0]["type"] == "function"
    assert out.tools[0]["name"] == "Bash"


@pytest.mark.asyncio
async def test_convert__anthropic_message_to_openai_responses__request_tool_result_mixed_content() -> (
    None
):
    """tool_result with a list of text + image parts should stringify."""
    req = anthropic_models.CreateMessageRequest(
        model="claude-3",
        messages=[
            anthropic_models.Message(role="user", content="screenshot"),
            anthropic_models.Message(
                role="assistant",
                content=[
                    anthropic_models.ToolUseBlock(
                        type="tool_use",
                        id="call_img",
                        name="screenshot",
                        input={},
                    ),
                ],
            ),
            anthropic_models.Message(
                role="user",
                content=[
                    anthropic_models.ToolResultBlock(
                        type="tool_result",
                        tool_use_id="call_img",
                        content=[
                            anthropic_models.TextBlock(type="text", text="here: "),
                            anthropic_models.ImageBlock(
                                type="image",
                                source=anthropic_models.ImageSource(
                                    type="base64",
                                    media_type="image/png",
                                    data="AAAA",
                                ),
                            ),
                            anthropic_models.TextBlock(type="text", text=" done"),
                        ],
                    )
                ],
            ),
        ],
        max_tokens=64,
    )

    out = convert__anthropic_message_to_openai_responses__request(req)
    assert isinstance(out.input, list)
    fn_out = next(
        item for item in out.input if item.get("type") == "function_call_output"
    )
    assert isinstance(fn_out["output"], str)
    assert fn_out["output"].startswith("here: ")
    assert fn_out["output"].endswith(" done")
    # The image part must be serialized (as JSON) rather than dropped.
    assert "image" in fn_out["output"]


@pytest.mark.asyncio
async def test_convert__anthropic_message_to_openai_responses__request_pending_text_after_tool_result() -> (
    None
):
    """Text following a tool_result in the same user message is flushed."""
    req = anthropic_models.CreateMessageRequest(
        model="claude-3",
        messages=[
            anthropic_models.Message(role="user", content="run"),
            anthropic_models.Message(
                role="assistant",
                content=[
                    anthropic_models.ToolUseBlock(
                        type="tool_use",
                        id="call_1",
                        name="shell",
                        input={"cmd": "ls"},
                    ),
                ],
            ),
            anthropic_models.Message(
                role="user",
                content=[
                    anthropic_models.ToolResultBlock(
                        type="tool_result",
                        tool_use_id="call_1",
                        content="ok",
                    ),
                    anthropic_models.TextBlock(type="text", text="now list again"),
                ],
            ),
        ],
        max_tokens=64,
    )

    out = convert__anthropic_message_to_openai_responses__request(req)
    assert isinstance(out.input, list)
    types = [item.get("type") for item in out.input]
    # user, function_call, function_call_output, user (post-result text)
    assert types == ["message", "function_call", "function_call_output", "message"]
    trailing = out.input[-1]
    assert trailing["role"] == "user"
    assert trailing["content"][0]["text"] == "now list again"


@pytest.mark.asyncio
async def test_convert__anthropic_message_to_openai_responses__request_assistant_interleaved_ordering() -> (
    None
):
    """Assistant text/tool_use interleave must preserve original order."""
    req = anthropic_models.CreateMessageRequest(
        model="claude-3",
        messages=[
            anthropic_models.Message(role="user", content="do it"),
            anthropic_models.Message(
                role="assistant",
                content=[
                    anthropic_models.TextBlock(type="text", text="first, "),
                    anthropic_models.ToolUseBlock(
                        type="tool_use",
                        id="call_a",
                        name="shell",
                        input={"cmd": "ls"},
                    ),
                    anthropic_models.TextBlock(type="text", text="then, "),
                    anthropic_models.ToolUseBlock(
                        type="tool_use",
                        id="call_b",
                        name="shell",
                        input={"cmd": "pwd"},
                    ),
                    anthropic_models.TextBlock(type="text", text="done."),
                ],
            ),
        ],
        max_tokens=64,
    )

    out = convert__anthropic_message_to_openai_responses__request(req)
    assert isinstance(out.input, list)
    types = [item.get("type") for item in out.input]
    assert types == [
        "message",  # user: do it
        "message",  # assistant: first,
        "function_call",  # call_a
        "message",  # assistant: then,
        "function_call",  # call_b
        "message",  # assistant: done.
    ]

    assistant_texts = [
        item["content"][0]["text"]
        for item in out.input
        if item.get("type") == "message" and item.get("role") == "assistant"
    ]
    assert assistant_texts == ["first, ", "then, ", "done."]

    fn_calls = [item for item in out.input if item.get("type") == "function_call"]
    assert [fc["call_id"] for fc in fn_calls] == ["call_a", "call_b"]
    assert [fc["name"] for fc in fn_calls] == ["shell", "shell"]
    assert [json.loads(fc["arguments"]) for fc in fn_calls] == [
        {"cmd": "ls"},
        {"cmd": "pwd"},
    ]
