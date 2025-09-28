from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from ccproxy.llms.formatters.context import register_request
from ccproxy.llms.formatters.openai_to_openai import (
    convert__openai_chat_to_openai_responses__request,
    convert__openai_chat_to_openai_responses__response,
    convert__openai_chat_to_openai_responses__stream,
    convert__openai_responses_to_openai_chat__response,
    convert__openai_responses_to_openai_chat__stream,
    convert__openai_responses_to_openaichat__request,
)
from ccproxy.llms.models import openai as openai_models


def _make_response_object_with_reasoning() -> openai_models.ResponseObject:
    reasoning_summary = [
        {"type": "summary_text", "text": "Thoughts", "signature": "sig"}
    ]
    return openai_models.ResponseObject(
        id="resp-1",
        object="response",
        created_at=0,
        status="completed",
        model="gpt-test",
        output=[
            openai_models.ReasoningOutput(
                type="reasoning",
                id="reasoning-1",
                status="completed",
                summary=reasoning_summary,
            ),
            openai_models.MessageOutput(
                type="message",
                role="assistant",
                id="msg-1",
                status="completed",
                content=[
                    openai_models.OutputTextContent(
                        type="output_text", text="Final answer"
                    )
                ],
            ),
        ],
        parallel_tool_calls=False,
    )


def test_responses_to_chat_serializes_thinking() -> None:
    response = _make_response_object_with_reasoning()

    chat = convert__openai_responses_to_openai_chat__response(response)

    choice = chat.choices[0]
    assert choice.message.content == (
        '<thinking signature="sig">Thoughts</thinking>Final answer'
    )


@pytest.mark.asyncio
async def test_chat_to_responses_extracts_thinking() -> None:
    chat_response = openai_models.ChatCompletionResponse(
        id="chat-1",
        created=0,
        model="gpt-test",
        object="chat.completion",
        choices=[
            openai_models.Choice(
                index=0,
                message=openai_models.ResponseMessage(
                    role="assistant",
                    content='<thinking signature="sig">Thoughts</thinking>Final answer',
                ),
                finish_reason="stop",
            )
        ],
    )

    response = await convert__openai_chat_to_openai_responses__response(chat_response)

    reasoning_items = [
        entry for entry in response.output if _get_type(entry) == "reasoning"
    ]
    assert len(reasoning_items) == 1
    summary = reasoning_items[0].summary  # type: ignore[attr-defined]
    assert summary is not None
    assert summary[0]["text"] == "Thoughts"
    assert summary[0]["signature"] == "sig"

    message_items = [
        entry for entry in response.output if _get_type(entry) == "message"
    ]
    assert len(message_items) == 1
    content = message_items[0].content[0].text  # type: ignore[attr-defined]
    assert content == "Final answer"

    assert response.reasoning is not None
    assert response.reasoning.summary  # type: ignore[union-attr]


def _get_type(entry: object) -> str | None:
    return getattr(entry, "type", None)


def test_responses_to_chat_handles_nested_summary_blocks() -> None:
    nested_summary = [
        {
            "type": "summary_group",
            "content": [
                {"type": "summary_text", "text": "First part. "},
                {"type": "summary_text", "text": "Second part."},
                {"type": "signature", "text": "sig-nested"},
            ],
        }
    ]

    response = openai_models.ResponseObject(
        id="resp-nested",
        object="response",
        created_at=0,
        status="completed",
        model="gpt-test",
        output=[
            openai_models.ReasoningOutput(
                type="reasoning",
                id="reasoning-nested",
                status="completed",
                summary=nested_summary,
            ),
        ],
        parallel_tool_calls=False,
    )

    chat = convert__openai_responses_to_openai_chat__response(response)
    choice = chat.choices[0]

    assert (
        choice.message.content
        == '<thinking signature="sig-nested">First part. Second part.</thinking>'
    )


def test_responses_to_chat_ignores_summary_mode_strings() -> None:
    response = openai_models.ResponseObject(
        id="resp-summary-mode",
        object="response",
        created_at=0,
        status="completed",
        model="gpt-test",
        output=[],
        parallel_tool_calls=False,
        reasoning=openai_models.Reasoning(summary="detailed"),
    )

    chat = convert__openai_responses_to_openai_chat__response(response)

    choice = chat.choices[0]
    assert choice.message.content == ""


@pytest.mark.asyncio
async def test_responses_request_to_chat_maps_reasoning_effort() -> None:
    request = openai_models.ResponseRequest(
        model="gpt-test",
        input="Hello",
        reasoning={"effort": "medium"},
    )

    chat_request = await convert__openai_responses_to_openaichat__request(request)

    assert chat_request.reasoning_effort == "medium"


@pytest.mark.asyncio
async def test_chat_request_to_responses_maps_reasoning_effort() -> None:
    chat_request = openai_models.ChatCompletionRequest(
        model="gpt-test",
        messages=[openai_models.ChatMessage(role="user", content="Hello")],
        reasoning_effort="high",
    )

    response_request = await convert__openai_chat_to_openai_responses__request(
        chat_request
    )

    assert response_request.reasoning == {"effort": "high", "summary": "auto"}


@pytest.mark.asyncio
async def test_chat_request_to_responses_defaults_reasoning(monkeypatch: Any) -> None:
    monkeypatch.delenv("LLM__OPENAI_THINKING_XML", raising=False)
    monkeypatch.delenv("OPENAI_STREAM_ENABLE_THINKING_SERIALIZATION", raising=False)

    chat_request = openai_models.ChatCompletionRequest(
        model="gpt-test",
        messages=[openai_models.ChatMessage(role="user", content="Hello")],
    )

    response_request = await convert__openai_chat_to_openai_responses__request(
        chat_request
    )

    assert response_request.reasoning == {"effort": "medium", "summary": "auto"}


@pytest.mark.asyncio
async def test_chat_request_to_responses_respects_disable_env(monkeypatch: Any) -> None:
    monkeypatch.setenv("LLM__OPENAI_THINKING_XML", "false")

    chat_request = openai_models.ChatCompletionRequest(
        model="gpt-test",
        messages=[openai_models.ChatMessage(role="user", content="Hello")],
    )

    response_request = await convert__openai_chat_to_openai_responses__request(
        chat_request
    )

    assert response_request.reasoning is None


@pytest.mark.asyncio
async def test_responses_request_promotes_developer_to_system() -> None:
    request = openai_models.ResponseRequest(
        model="gpt-test",
        input=[
            {
                "role": "developer",
                "content": "Developer instructions go here.\nFollow them exactly.",
            },
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "hello"},
                ],
            },
        ],
    )

    chat_request = await convert__openai_responses_to_openaichat__request(request)

    assert chat_request.messages[0].role == "system"
    assert "Developer instructions" in chat_request.messages[0].content
    assert chat_request.messages[1].role == "user"
    assert chat_request.messages[1].content == "hello"


@pytest.mark.asyncio
async def test_responses_request_preserves_plain_text_content() -> None:
    request = openai_models.ResponseRequest(
        model="gpt-test",
        input=[
            {
                "role": "user",
                "content": "plain text message",
            }
        ],
    )

    chat_request = await convert__openai_responses_to_openaichat__request(request)

    assert len(chat_request.messages) == 1
    assert chat_request.messages[0].role == "user"
    assert chat_request.messages[0].content == "plain text message"


@pytest.mark.asyncio
async def test_responses_fixture_request_converts_to_chat_messages() -> None:
    base_dir = Path(__file__).resolve().parents[3]
    json_path = base_dir / "request_debug_response_api.json"
    payload = json_path.read_text(encoding="utf-8")
    request = openai_models.ResponseRequest.model_validate_json(payload)

    chat_request = await convert__openai_responses_to_openaichat__request(request)

    assert len(chat_request.messages) == 27

    # The system instructions should be preserved and merged correctly.
    system_message = chat_request.messages[0]
    assert system_message.role == "system"
    assert isinstance(system_message.content, str)
    assert system_message.content.startswith(
        "You are a coding agent running in the opencode, a terminal-based"
    )
    assert "## Planning" in system_message.content

    # Spot-check early conversational turns.
    assert chat_request.messages[1].role == "user"
    assert chat_request.messages[1].content == "hello"
    assert chat_request.messages[2].role == "assistant"
    assert chat_request.messages[2].content == "Hi! How can I help you today?"

    # Repeated "(empty request)" placeholders indicate a regression.
    empty_markers = [
        message.content
        for message in chat_request.messages
        if isinstance(message.content, str) and message.content == "(empty request)"
    ]
    assert not empty_markers, "unexpected empty request placeholders present"

    # Tool interactions should be surfaced as assistant tool calls + tool outputs.
    assistant_with_tool = [
        message
        for message in chat_request.messages
        if message.role == "assistant" and message.tool_calls
    ]
    assert assistant_with_tool, "expected assistant tool call message missing"

    tool_messages = [m for m in chat_request.messages if m.role == "tool"]
    assert tool_messages, "expected tool output message missing"
    assert any(
        isinstance(msg.content, str) and "uid=1000" in msg.content
        for msg in tool_messages
    ), "tool output should include command result"


@pytest.mark.asyncio
async def test_chat_stream_to_responses_emits_full_lifecycle() -> None:
    register_request(
        openai_models.ChatCompletionRequest(
            model="gpt-4o-mini",
            messages=[
                openai_models.ChatMessage(role="system", content="Be helpful"),
                openai_models.ChatMessage(role="user", content="Hello"),
            ],
        ),
        "Be helpful",
    )

    async def source():
        yield {
            "id": "chunk-1",
            "model": "gpt-4o-mini",
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": "Hel"},
                    "finish_reason": None,
                }
            ],
        }
        yield {
            "id": "chunk-2",
            "model": "gpt-4o-mini",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": "lo"},
                    "finish_reason": None,
                }
            ],
            "usage": {
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "total_tokens": 2,
            },
        }
        yield {
            "id": "chunk-3",
            "model": "gpt-4o-mini",
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                }
            ],
        }

    events: list[openai_models.BaseStreamEvent] = []
    async for chunk in convert__openai_chat_to_openai_responses__stream(source()):
        events.append(chunk)

    types = [getattr(evt, "type", None) for evt in events]
    assert types[0] == "response.created"
    assert types[-1] == "response.completed"
    assert types.count("response.in_progress") >= 1
    assert "response.output_text.delta" in types
    assert "response.output_text.done" in types

    deltas = [
        evt.delta
        for evt in events
        if getattr(evt, "type", "") == "response.output_text.delta"
    ]
    assert deltas == ["Hel", "lo"]

    done = next(
        evt for evt in events if getattr(evt, "type", "") == "response.output_text.done"
    )
    assert done.text == "Hello"

    created = events[0]
    assert created.response.background is False
    assert created.response.instructions == "Be helpful"
    assert created.response.parallel_tool_calls is True
    assert created.response.temperature == pytest.approx(1.0)
    assert created.response.top_p == pytest.approx(1.0)
    assert created.response.text == {"format": {"type": "text"}, "verbosity": "low"}

    in_progress = events[1]
    assert in_progress.type == "response.in_progress"
    assert in_progress.response.parallel_tool_calls is True
    assert in_progress.response.instructions == "Be helpful"

    completed = events[-1]
    assert completed.type == "response.completed"
    assert completed.response.output
    final_message = completed.response.output[0]
    assert final_message.content and final_message.content[0].text == "Hello"
    assert completed.response.usage is not None
    assert completed.response.instructions == "Be helpful"


@pytest.mark.asyncio
async def test_chat_stream_to_responses_includes_usage_from_final_chunk() -> None:
    register_request(
        openai_models.ChatCompletionRequest(
            model="gpt-4o-mini",
            messages=[
                openai_models.ChatMessage(role="system", content="System note"),
                openai_models.ChatMessage(role="user", content="Hi"),
            ],
        ),
        "System note",
    )

    async def source():
        yield {
            "id": "chunk-1",
            "model": "gpt-4o-mini",
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": "Hello "},
                    "finish_reason": None,
                }
            ],
        }
        yield {
            "id": "chunk-2",
            "model": "gpt-4o-mini",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": "world"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 2,
                "completion_tokens": 2,
                "total_tokens": 4,
            },
        }

    events: list[openai_models.BaseStreamEvent] = []
    async for chunk in convert__openai_chat_to_openai_responses__stream(source()):
        events.append(chunk)

    types = [getattr(evt, "type", None) for evt in events]
    assert types[0] == "response.created"
    assert types[-1] == "response.completed"
    assert types.count("response.in_progress") >= 1
    assert types.count("response.output_text.delta") == 2
    assert "response.output_text.done" in types

    completed = events[-1]
    assert completed.response.usage is not None
    assert completed.response.usage.input_tokens == 2
    assert completed.response.usage.output_tokens == 2
    assert completed.response.instructions == "System note"

    created = events[0]
    assert created.response.text["verbosity"] == "low"


@pytest.mark.asyncio
async def test_chat_stream_tool_calls_emit_responses_events() -> None:
    register_request(None)

    async def source() -> AsyncIterator[dict[str, Any]]:
        yield {
            "id": "chunk-1",
            "model": "gpt-4o-mini",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_abc",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"city": "',
                                },
                            }
                        ]
                    },
                }
            ],
        }
        yield {
            "id": "chunk-2",
            "model": "gpt-4o-mini",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "function": {
                                    "arguments": "New York",
                                },
                            }
                        ]
                    },
                }
            ],
        }
        yield {
            "id": "chunk-3",
            "model": "gpt-4o-mini",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "function": {
                                    "arguments": '"}',
                                },
                            }
                        ]
                    },
                    "finish_reason": "tool_calls",
                }
            ],
        }

    events: list[openai_models.BaseStreamEvent] = []
    async for event in convert__openai_chat_to_openai_responses__stream(source()):
        events.append(event)

    types = [getattr(evt, "type", None) for evt in events]
    assert types == [
        "response.created",
        "response.in_progress",
        "response.output_item.added",
        "response.function_call_arguments.delta",
        "response.function_call_arguments.delta",
        "response.function_call_arguments.delta",
        "response.function_call_arguments.done",
        "response.output_item.done",
        "response.completed",
    ]

    deltas = [
        evt.delta
        for evt in events
        if getattr(evt, "type", "") == "response.function_call_arguments.delta"
    ]
    assert deltas == ['{"city": "', "New York", '"}']

    fn_added = next(
        evt
        for evt in events
        if getattr(evt, "type", "") == "response.output_item.added"
        and getattr(evt.item, "type", "") == "function_call"
    )
    assert fn_added.item.id == "call_abc"
    assert fn_added.item.name == "get_weather"
    assert fn_added.item.call_id == "call_abc"

    args_done = next(
        evt
        for evt in events
        if getattr(evt, "type", "") == "response.function_call_arguments.done"
    )
    assert args_done.arguments == '{"city": "New York"}'

    completed = events[-1]
    assert completed.type == "response.completed"
    assert completed.response.parallel_tool_calls is True
    assert len(completed.response.output) == 1
    fn_output = completed.response.output[0]
    assert fn_output.id == "call_abc"
    assert fn_output.name == "get_weather"
    assert fn_output.arguments == '{"city": "New York"}'
    tool_calls = getattr(completed.response, "tool_calls", []) or []
    assert tool_calls
    assert tool_calls[0]["function"]["arguments"] == '{"city": "New York"}'


@pytest.mark.asyncio
async def test_responses_stream_includes_thinking_xml() -> None:
    async def source():
        yield openai_models.ResponseCreatedEvent(
            type="response.created",
            sequence_number=1,
            response=_make_response_object_with_reasoning(),
        )
        yield openai_models.ReasoningSummaryPartAddedEvent(
            type="response.reasoning_summary_part.added",
            sequence_number=2,
            item_id="reasoning-1",
            output_index=0,
            summary_index=0,
            part=openai_models.ReasoningSummaryPart(type="signature", text="sig"),
        )
        yield openai_models.ReasoningSummaryTextDeltaEvent(
            type="response.reasoning_summary_text.delta",
            sequence_number=3,
            item_id="reasoning-1",
            output_index=0,
            summary_index=0,
            delta="Thoughts",
        )
        yield openai_models.ReasoningSummaryTextDoneEvent(
            type="response.reasoning_summary_text.done",
            sequence_number=4,
            item_id="reasoning-1",
            output_index=0,
            summary_index=0,
            text="Thoughts",
        )
        yield openai_models.ResponseOutputTextDeltaEvent(
            type="response.output_text.delta",
            sequence_number=5,
            item_id="msg_stream",
            output_index=0,
            content_index=0,
            delta="Final answer",
        )

    deltas: list[str] = []

    async for chunk in convert__openai_responses_to_openai_chat__stream(source()):
        if chunk.choices:
            delta_msg = chunk.choices[0].delta
            if delta_msg and delta_msg.content:
                deltas.append(delta_msg.content)

    assert deltas[0] == '<thinking signature="sig">Thoughts'
    assert deltas[1] == "</thinking>"
    assert deltas[2] == "Final answer"
