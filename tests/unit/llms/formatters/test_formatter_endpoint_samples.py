from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest
from pydantic import TypeAdapter, ValidationError
from tests.helpers.sample_loader import load_sample

from ccproxy.llms.formatters.anthropic_to_openai import (
    convert__anthropic_message_to_openai_chat__request,
    convert__anthropic_message_to_openai_chat__response,
    convert__anthropic_message_to_openai_chat__stream,
    convert__anthropic_message_to_openai_responses__request,
    convert__anthropic_message_to_openai_responses__response,
    convert__anthropic_message_to_openai_responses__stream,
)
from ccproxy.llms.formatters.openai_to_anthropic import (
    convert__openai_chat_to_anthropic_message__request,
    convert__openai_chat_to_anthropic_messages__response,
    convert__openai_chat_to_anthropic_messages__stream,
    convert__openai_responses_to_anthropic_message__request,
    convert__openai_responses_to_anthropic_message__response,
    convert__openai_responses_to_anthropic_messages__stream,
)
from ccproxy.llms.models import anthropic as anthropic_models
from ccproxy.llms.models import openai as openai_models


async def _iterate(items: list[Any]) -> AsyncIterator[Any]:
    for item in items:
        yield item


def _anthropic_text(blocks: list[anthropic_models.ResponseContentBlock]) -> str:
    for block in blocks:
        if getattr(block, "type", None) == "text":
            return getattr(block, "text", "")
    return ""


# ---------------------------------------------------------------------------
# OpenAI ChatCompletion -> Anthropic Message conversions
# ---------------------------------------------------------------------------

OPENAI_CHAT_RESPONSE_CASES = [
    (
        "copilot_chat_completions",
        {
            "block_types": ["text"],
            "stop_reason": "end_turn",
            "text_snippet": "Hello",
        },
    ),
    (
        "copilot_chat_completions_structured",
        {
            "block_types": ["text"],
            "stop_reason": "end_turn",
            "text_startswith": "{",
        },
    ),
    (
        "copilot_chat_completions_thinking",
        {
            "block_types": ["text"],
            "stop_reason": "end_turn",
            "text_snippet": "factorial",
        },
    ),
    (
        "copilot_chat_completions_tools",
        {
            "block_types": ["tool_use", "tool_use"],
            "stop_reason": "tool_use",
            "tool_names": {"get_weather", "calculate_distance"},
        },
    ),
]


@pytest.mark.unit
@pytest.mark.parametrize(("sample_name", "expect"), OPENAI_CHAT_RESPONSE_CASES)
def test_openai_chat_response_to_anthropic(
    sample_name: str, expect: dict[str, Any]
) -> None:
    sample = load_sample(sample_name)
    payload = sample["response"]["body"]

    openai_response = openai_models.ChatCompletionResponse.model_validate(payload)
    converted = convert__openai_chat_to_anthropic_messages__response(openai_response)

    block_types = [block.type for block in converted.content]
    assert block_types == expect["block_types"]
    assert converted.stop_reason == expect["stop_reason"]

    text = _anthropic_text(converted.content)
    if expect.get("text_snippet"):
        assert expect["text_snippet"] in text
    if expect.get("text_startswith"):
        assert text.startswith(expect["text_startswith"])

    if expect.get("tool_names"):
        tool_names = {
            block.name for block in converted.content if block.type == "tool_use"
        }
        assert tool_names == expect["tool_names"]

    if openai_response.usage:
        assert converted.usage.input_tokens == openai_response.usage.prompt_tokens
        assert converted.usage.output_tokens == openai_response.usage.completion_tokens


@pytest.mark.unit
@pytest.mark.asyncio
async def test_openai_chat_request_to_anthropic_includes_custom_tools() -> None:
    request = openai_models.ChatCompletionRequest(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hello"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                        "required": ["location"],
                    },
                },
            }
        ],
        tool_choice="auto",
        parallel_tool_calls=False,
    )

    converted = await convert__openai_chat_to_anthropic_message__request(request)

    assert bool(converted.stream) == bool(request.stream)
    assert converted.tool_choice is not None
    assert getattr(converted.tool_choice, "type", None) == "auto"
    assert getattr(converted.tool_choice, "disable_parallel_tool_use", None) is True

    assert converted.tools
    first_tool = converted.tools[0]
    assert getattr(first_tool, "type", None) == "custom"
    assert getattr(first_tool, "name", None) == "get_weather"
    assert getattr(first_tool, "description", None) == "Get weather"
    assert first_tool.input_schema["required"] == ["location"]

    assert converted.messages[0].role == "user"
    assert converted.messages[0].content == "Hello"


OPENAI_CHAT_STREAM_CASES = [
    (
        "copilot_chat_completions_stream",
        {
            "expect_text": True,
            "text_snippet": "Hello",
            "expect_tool_blocks": False,
        },
    ),
    (
        "copilot_chat_completions_structured_stream",
        {
            "expect_text": True,
            "text_startswith": "{",
            "expect_tool_blocks": False,
        },
    ),
    (
        "copilot_chat_completions_thinking_stream",
        {
            "expect_text": True,
            "text_snippet": "factorial",
            "expect_tool_blocks": False,
        },
    ),
    (
        "copilot_chat_completions_tools_stream",
        {
            "expect_text": False,
            "expect_tool_blocks": True,
            "expected_tool_names": {"get_weather", "calculate_distance"},
        },
    ),
]


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize(("sample_name", "expect"), OPENAI_CHAT_STREAM_CASES)
async def test_openai_chat_stream_to_anthropic(
    sample_name: str, expect: dict[str, Any]
) -> None:
    sample = load_sample(sample_name)
    events = [
        evt.get("json")
        for evt in sample["response"].get("events", [])
        if evt.get("json")
    ]

    converted: list[anthropic_models.MessageStreamEvent] = []
    async for event in convert__openai_chat_to_anthropic_messages__stream(
        _iterate(events)
    ):
        converted.append(event)

    assert converted, "expected converted stream events"
    assert converted[0].type == "message_start"
    assert converted[-1].type == "message_stop"

    text = "".join(
        getattr(evt.delta, "text", "")
        for evt in converted
        if evt.type == "content_block_delta"
    )
    if expect.get("expect_text"):
        assert text
        if expect.get("text_snippet"):
            assert expect["text_snippet"] in text
        if expect.get("text_startswith"):
            assert text.startswith(expect["text_startswith"])
    else:
        assert not text

    tool_names = {
        getattr(evt.content_block, "name", None)
        for evt in converted
        if evt.type == "content_block_start"
        and getattr(evt.content_block, "type", None) == "tool_use"
    }
    has_tool_blocks = bool(tool_names)
    assert has_tool_blocks is expect["expect_tool_blocks"]
    if expect.get("expected_tool_names"):
        assert tool_names == expect["expected_tool_names"]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_openai_chat_stream_request_to_anthropic() -> None:
    request = openai_models.ChatCompletionRequest(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hi"}],
        stream=True,
    )

    converted = await convert__openai_chat_to_anthropic_message__request(request)

    assert converted.stream is True
    assert converted.max_tokens is not None
    assert converted.messages[0].content == "Hi"


# ---------------------------------------------------------------------------
# OpenAI Responses -> Anthropic Message conversions
# ---------------------------------------------------------------------------

OPENAI_RESPONSES_CASES = [
    (
        "codex_responses",
        {
            "block_types": ["text"],
            "text_snippet": "Hey!",
        },
    ),
    (
        "codex_responses_structured",
        {
            "block_types": ["text"],
            "text_startswith": "{",
        },
    ),
    (
        "codex_responses_thinking",
        {
            "block_types": ["text"],
            "text_snippet": "factorial",
        },
    ),
    (
        "codex_responses_tools",
        {
            "block_types": ["tool_use", "tool_use"],
            "tool_names": {"get_weather", "calculate_distance"},
        },
    ),
]


@pytest.mark.unit
@pytest.mark.parametrize(("sample_name", "expect"), OPENAI_RESPONSES_CASES)
def test_openai_responses_to_anthropic(
    sample_name: str, expect: dict[str, Any]
) -> None:
    sample = load_sample(sample_name)
    payload = sample["response"]["body"]

    openai_response = openai_models.ResponseObject.model_validate(payload)
    converted = convert__openai_responses_to_anthropic_message__response(
        openai_response
    )

    block_types = [block.type for block in converted.content]
    assert block_types == expect["block_types"]

    text = _anthropic_text(converted.content)
    if expect.get("text_snippet"):
        assert expect["text_snippet"] in text
    if expect.get("text_startswith"):
        assert text.startswith(expect["text_startswith"])

    if expect.get("tool_names"):
        tool_names = {
            block.name for block in converted.content if block.type == "tool_use"
        }
        assert tool_names == expect["tool_names"]

    if openai_response.usage:
        assert converted.usage.input_tokens == openai_response.usage.input_tokens
        assert converted.usage.output_tokens == openai_response.usage.output_tokens


@pytest.mark.unit
def test_openai_responses_request_to_anthropic_includes_tools() -> None:
    request = openai_models.ResponseRequest(
        model="gpt-4o",
        input=[
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "hi"}],
            }
        ],
        tools=[
            {
                "type": "function",
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"],
                },
            }
        ],
        tool_choice={"type": "function", "function": {"name": "get_weather"}},
    )

    converted = convert__openai_responses_to_anthropic_message__request(request)

    assert converted.tools
    tool = converted.tools[0]
    assert getattr(tool, "type", None) == "custom"
    assert getattr(tool, "name", None) == "get_weather"
    assert tool.input_schema["required"] == ["location"]

    assert converted.tool_choice is not None
    assert getattr(converted.tool_choice, "type", None) == "tool"
    assert getattr(converted.tool_choice, "name", None) == "get_weather"


OPENAI_RESPONSES_STREAM_CASES = [
    (
        "codex_responses_stream",
        {"expect_function_events": False, "expect_text_delta": True},
    ),
    (
        "codex_responses_structured_stream",
        {"expect_function_events": False, "expect_text_delta": True},
    ),
    (
        "codex_responses_thinking_stream",
        {"expect_function_events": False, "expect_text_delta": True},
    ),
    (
        "codex_responses_tools_stream",
        {"expect_function_events": True, "expect_text_delta": False},
    ),
]


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize(("sample_name", "expect"), OPENAI_RESPONSES_STREAM_CASES)
async def test_openai_responses_stream_to_anthropic(
    sample_name: str, expect: dict[str, bool]
) -> None:
    sample = load_sample(sample_name)
    payloads = [
        evt.get("json")
        for evt in sample["response"].get("events", [])
        if evt.get("json")
    ]

    converted: list[anthropic_models.MessageStreamEvent] = []
    async for event in convert__openai_responses_to_anthropic_messages__stream(
        _iterate(payloads)
    ):
        converted.append(event)

    assert converted, "expected converted responses stream"
    assert converted[0].type == "message_start"
    assert converted[-1].type == "message_stop"

    has_text_delta = any(evt.type == "content_block_delta" for evt in converted)
    if expect["expect_text_delta"]:
        assert has_text_delta
    else:
        assert not has_text_delta


# ---------------------------------------------------------------------------
# Anthropic -> OpenAI Chat conversions
# ---------------------------------------------------------------------------

ANTHROPIC_CHAT_CASES = [
    (
        "claude_messages",
        {
            "expected_finish": "stop",
            "text_snippet": "Hello!",
        },
    ),
    (
        "claude_messages_tools",
        {
            "expected_finish": "tool_calls",
            "text_snippet": "weather information",
        },
    ),
]


@pytest.mark.unit
@pytest.mark.parametrize(("sample_name", "expect"), ANTHROPIC_CHAT_CASES)
def test_anthropic_message_to_openai_chat(
    sample_name: str, expect: dict[str, Any]
) -> None:
    sample = load_sample(sample_name)
    payload = sample["response"]["body"]

    message = anthropic_models.MessageResponse.model_validate(payload)
    converted = convert__anthropic_message_to_openai_chat__response(message)

    choice = converted.choices[0]
    assert choice.finish_reason == expect["expected_finish"]

    content = choice.message.content or ""
    if isinstance(content, str) and expect.get("text_snippet"):
        assert expect["text_snippet"] in content


@pytest.mark.unit
def test_anthropic_message_request_to_openai_chat_handles_tools() -> None:
    request = anthropic_models.CreateMessageRequest.model_validate(
        {
            "model": "claude-sonnet",
            "max_tokens": 512,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What's the weather?"},
                    ],
                }
            ],
            "tools": [
                {
                    "type": "tool",
                    "name": "get_weather",
                    "description": "Get weather",
                    "input_schema": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                        "required": ["location"],
                    },
                }
            ],
        }
    )

    converted = convert__anthropic_message_to_openai_chat__request(request)

    assert converted.model == request.model
    assert converted.messages[-1].role == "user"
    assert "weather" in converted.messages[-1].content

    assert converted.tools
    tool = converted.tools[0]
    assert tool.type == "function"
    assert tool.function.name == "get_weather"
    assert tool.function.parameters["required"] == ["location"]


ANTHROPIC_CHAT_STREAM_CASES = [
    ("claude_messages_stream", {"expect_tool_calls": False}),
    ("claude_messages_tools_stream", {"expect_tool_calls": False}),
]


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize(("sample_name", "expect"), ANTHROPIC_CHAT_STREAM_CASES)
async def test_anthropic_stream_to_openai_chat(
    sample_name: str, expect: dict[str, bool]
) -> None:
    sample = load_sample(sample_name)
    adapter = TypeAdapter(anthropic_models.MessageStreamEvent)

    events: list[Any] = []
    for evt in sample["response"].get("events", []):
        body = evt.get("json")
        if not body:
            continue
        try:
            events.append(adapter.validate_python(body))
        except ValidationError:
            events.append(body)

    converted: list[openai_models.ChatCompletionChunk] = []
    async for chunk in convert__anthropic_message_to_openai_chat__stream(
        _iterate(events)
    ):
        converted.append(chunk)

    assert converted, "expected converted openai chunks"
    has_tool_call = any(
        choice.delta.tool_calls
        for chunk in converted
        for choice in chunk.choices
        if choice.delta
    )
    assert has_tool_call is expect["expect_tool_calls"]


# ---------------------------------------------------------------------------
# Anthropic -> OpenAI Responses conversions
# ---------------------------------------------------------------------------

ANTHROPIC_RESPONSES_CASES = [
    (
        "claude_messages",
        {
            "text_snippet": "Hello!",
        },
    ),
    (
        "claude_messages_tools",
        {
            "text_snippet": "weather information",
        },
    ),
]


@pytest.mark.unit
@pytest.mark.parametrize(("sample_name", "expect"), ANTHROPIC_RESPONSES_CASES)
def test_anthropic_message_to_openai_responses(
    sample_name: str, expect: dict[str, Any]
) -> None:
    sample = load_sample(sample_name)
    payload = sample["response"]["body"]

    message = anthropic_models.MessageResponse.model_validate(payload)
    converted = convert__anthropic_message_to_openai_responses__response(message)

    assert converted.output
    message_output = converted.output[0]
    assert message_output.type == "message"

    text = ""
    for content in message_output.content:
        if isinstance(content, dict) and content.get("type") == "output_text":
            text = content.get("text", "")
            break
        if hasattr(content, "type") and getattr(content, "type", None) == "output_text":
            text = getattr(content, "text", "")
            break
    assert expect["text_snippet"] in text

    if message.usage:
        usage = converted.usage
        assert usage is not None
        assert usage.input_tokens == message.usage.input_tokens
        assert usage.output_tokens == message.usage.output_tokens


@pytest.mark.unit
def test_anthropic_message_request_to_openai_responses_includes_tools() -> None:
    request = anthropic_models.CreateMessageRequest.model_validate(
        {
            "model": "claude-sonnet",
            "max_tokens": 256,
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "calc 1+1"}],
                }
            ],
            "tools": [
                {
                    "type": "tool",
                    "name": "calculate",
                    "description": "Do math",
                    "input_schema": {
                        "type": "object",
                        "properties": {"expression": {"type": "string"}},
                        "required": ["expression"],
                    },
                }
            ],
        }
    )

    converted = convert__anthropic_message_to_openai_responses__request(request)

    assert converted.model == request.model
    assert converted.input
    item = converted.input[0]
    assert item["type"] == "message"

    assert converted.tools
    tool = converted.tools[0]
    assert tool["type"] == "function"
    assert tool["name"] == "calculate"


ANTHROPIC_RESPONSES_STREAM_CASES = [
    ("claude_messages_stream", {"expect_function_events": False}),
    ("claude_messages_tools_stream", {"expect_function_events": True}),
]


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize(("sample_name", "expect"), ANTHROPIC_RESPONSES_STREAM_CASES)
async def test_anthropic_stream_to_openai_responses(
    sample_name: str, expect: dict[str, bool]
) -> None:
    sample = load_sample(sample_name)
    adapter = TypeAdapter(anthropic_models.MessageStreamEvent)

    events: list[Any] = []
    for evt in sample["response"].get("events", []):
        body = evt.get("json")
        if not body:
            continue
        try:
            events.append(adapter.validate_python(body))
        except ValidationError:
            events.append(body)

    converted: list[Any] = []
    async for event in convert__anthropic_message_to_openai_responses__stream(
        _iterate(events)
    ):
        converted.append(event)

    assert converted, "expected openai responses stream events"

    has_function_event = any(
        getattr(event, "type", "").startswith("response.function_call")
        for event in converted
    )
    assert has_function_event is expect["expect_function_events"]

    assert converted[0].type == "response.created"
