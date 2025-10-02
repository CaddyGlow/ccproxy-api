from __future__ import annotations

from typing import Any

import pytest
from pydantic import TypeAdapter, ValidationError
from tests.helpers.sample_loader import load_sample

from ccproxy.llms.models import anthropic as anthropic_models
from ccproxy.llms.models import openai as openai_models


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                text = block.get("text")
            else:
                text = getattr(block, "text", None)
            if isinstance(text, str) and text:
                parts.append(text)
        return " ".join(parts)
    return ""


CHAT_RESPONSE_CASES = [
    (
        "copilot_chat_completions",
        {
            "finish_reason": "stop",
            "tool_calls": 0,
            "text_contains": "Hello",
        },
    ),
    (
        "copilot_chat_completions_structured",
        {
            "finish_reason": "stop",
            "tool_calls": 0,
            "text_startswith": "{",
        },
    ),
    (
        "copilot_chat_completions_thinking",
        {
            "finish_reason": "stop",
            "tool_calls": 0,
            "text_contains": "factorial",
        },
    ),
    (
        "copilot_chat_completions_tools",
        {
            "finish_reason": "tool_calls",
            "tool_calls": 2,
            "text_empty": True,
        },
    ),
]


@pytest.mark.unit
@pytest.mark.parametrize(("sample_name", "expect"), CHAT_RESPONSE_CASES)
def test_openai_chat_samples_validate(sample_name: str, expect: dict[str, Any]) -> None:
    sample = load_sample(sample_name)
    payload = sample["response"]["body"]

    response = openai_models.ChatCompletionResponse.model_validate(payload)
    assert response.id
    assert response.choices

    choice = response.choices[0]
    tool_calls = choice.message.tool_calls or []
    assert len(tool_calls) == expect["tool_calls"]
    assert choice.finish_reason == expect["finish_reason"]

    content = choice.message.content
    text = _message_text(content)
    if expect.get("text_empty"):
        assert not text
        assert content is None
    if expect.get("text_contains"):
        assert expect["text_contains"] in text
    if expect.get("text_startswith"):
        assert text.startswith(expect["text_startswith"])


CHAT_STREAM_CASES = [
    (
        "copilot_chat_completions_stream",
        {"expect_tool_chunk": False, "expect_text_chunk": True},
    ),
    (
        "copilot_chat_completions_structured_stream",
        {"expect_tool_chunk": False, "expect_text_chunk": True},
    ),
    (
        "copilot_chat_completions_thinking_stream",
        {"expect_tool_chunk": False, "expect_text_chunk": True},
    ),
    (
        "copilot_chat_completions_tools_stream",
        {"expect_tool_chunk": True, "expect_text_chunk": False},
    ),
]


@pytest.mark.unit
@pytest.mark.parametrize(("sample_name", "expect"), CHAT_STREAM_CASES)
def test_openai_chat_stream_samples_validate(
    sample_name: str, expect: dict[str, bool]
) -> None:
    sample = load_sample(sample_name)
    events = sample["response"].get("events", [])

    parsed: list[openai_models.ChatCompletionChunk] = []
    for event in events:
        body = event.get("json")
        if not body:
            continue
        try:
            chunk = openai_models.ChatCompletionChunk.model_validate(body)
        except ValidationError:
            continue
        parsed.append(chunk)

    assert parsed, "expected at least one parsed chunk"

    has_text_delta = any(
        isinstance(choice.delta.content, str) and choice.delta.content
        for chunk in parsed
        for choice in chunk.choices
    )
    assert has_text_delta is expect["expect_text_chunk"]

    has_tool_call = any(
        choice.delta.tool_calls for chunk in parsed for choice in chunk.choices
    )
    assert has_tool_call is expect["expect_tool_chunk"]


RESPONSES_CASES = [
    (
        "codex_responses",
        {
            "expect_reasoning": True,
            "expect_function_calls": 0,
            "text_contains": "Hey!",
        },
    ),
    (
        "codex_responses_structured",
        {
            "expect_reasoning": True,
            "expect_function_calls": 0,
            "text_startswith": "{",
        },
    ),
    (
        "codex_responses_thinking",
        {
            "expect_reasoning": True,
            "expect_function_calls": 0,
            "text_contains": "factorial",
        },
    ),
    (
        "codex_responses_tools",
        {
            "expect_reasoning": True,
            "expect_function_calls": 2,
            "text_empty": True,
        },
    ),
]


@pytest.mark.unit
@pytest.mark.parametrize(("sample_name", "expect"), RESPONSES_CASES)
def test_openai_responses_samples_validate(
    sample_name: str, expect: dict[str, Any]
) -> None:
    sample = load_sample(sample_name)
    payload = sample["response"]["body"]

    response = openai_models.ResponseObject.model_validate(payload)
    assert response.id
    assert response.model
    assert response.output

    reasoning_count = sum(
        1 for item in response.output if getattr(item, "type", None) == "reasoning"
    )
    assert bool(reasoning_count) is expect["expect_reasoning"]

    function_calls = [
        item
        for item in response.output
        if getattr(item, "type", None) == "function_call"
    ]
    assert len(function_calls) == expect["expect_function_calls"]

    message_items = [
        item for item in response.output if getattr(item, "type", None) == "message"
    ]
    if expect.get("text_empty"):
        assert not message_items
    else:
        assert message_items
        text = _message_text(getattr(message_items[0], "content", ""))
        if expect.get("text_contains"):
            assert expect["text_contains"] in text
        if expect.get("text_startswith"):
            assert text.startswith(expect["text_startswith"])


RESPONSES_STREAM_CASES = [
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
@pytest.mark.parametrize(("sample_name", "expect"), RESPONSES_STREAM_CASES)
def test_openai_responses_stream_samples_validate(
    sample_name: str, expect: dict[str, bool]
) -> None:
    sample = load_sample(sample_name)
    events = sample["response"].get("events", [])

    adapter = TypeAdapter(openai_models.AnyStreamEvent)
    parsed: list[openai_models.AnyStreamEvent] = []
    for event in events:
        body = event.get("json")
        if not body:
            continue
        try:
            parsed.append(adapter.validate_python(body))
        except ValidationError:
            continue

    assert parsed, "expected at least one parsed stream event"

    has_text_delta = any(
        getattr(evt.root, "type", "") == "response.output_text.delta" for evt in parsed
    )
    if expect["expect_text_delta"]:
        assert has_text_delta
    else:
        assert not has_text_delta

    has_function_event = any(
        getattr(evt.root, "type", "").startswith("response.function_call")
        for evt in parsed
    )
    assert has_function_event is expect["expect_function_events"]


ANTHROPIC_MESSAGE_CASES = [
    (
        "claude_messages",
        {
            "stop_reason": "end_turn",
            "tool_blocks": 0,
            "text_contains": "Hello!",
        },
    ),
    (
        "claude_messages_tools",
        {
            "stop_reason": "tool_use",
            "tool_blocks": 2,
            "text_contains": "I'll help you",
        },
    ),
]


@pytest.mark.unit
@pytest.mark.parametrize(("sample_name", "expect"), ANTHROPIC_MESSAGE_CASES)
def test_anthropic_message_samples_validate(
    sample_name: str, expect: dict[str, Any]
) -> None:
    sample = load_sample(sample_name)
    payload = sample["response"]["body"]

    message = anthropic_models.MessageResponse.model_validate(payload)
    assert message.id
    assert message.model
    assert message.content
    assert message.stop_reason == expect["stop_reason"]

    tool_blocks = [block for block in message.content if block.type == "tool_use"]
    assert len(tool_blocks) == expect["tool_blocks"]

    text_blocks = [block for block in message.content if block.type == "text"]
    if text_blocks:
        assert expect["text_contains"] in text_blocks[0].text


ANTHROPIC_STREAM_CASES = [
    ("claude_messages_stream", {"expect_tool_start": False}),
    ("claude_messages_tools_stream", {"expect_tool_start": True}),
]


@pytest.mark.unit
@pytest.mark.parametrize(("sample_name", "expect"), ANTHROPIC_STREAM_CASES)
def test_anthropic_stream_samples_validate(
    sample_name: str, expect: dict[str, bool]
) -> None:
    sample = load_sample(sample_name)
    events = sample["response"].get("events", [])

    adapter: TypeAdapter[anthropic_models.MessageStreamEvent] = TypeAdapter(
        anthropic_models.MessageStreamEvent
    )
    parsed: list[Any] = []
    for event in events:
        body = event.get("json")
        if not body:
            continue
        try:
            parsed.append(adapter.validate_python(body))
        except ValidationError:
            parsed.append(body)

    assert parsed, "expected at least one parsed Anthropic event"

    has_tool_start = any(
        evt.type == "content_block_start"
        and getattr(evt, "content_block", None)
        and getattr(evt.content_block, "type", None) == "tool_use"
        for evt in parsed
    )
    assert has_tool_start is expect["expect_tool_start"]
