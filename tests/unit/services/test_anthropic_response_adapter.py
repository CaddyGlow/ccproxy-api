"""Unit tests for AnthropicResponseAPIAdapter.

Covers direct conversion between Anthropic Messages and OpenAI Response API:
- Request conversion (messages → input, system → instructions, fields passthrough)
- Response conversion from both nested `response.output` and `choices` styles
- Streaming conversion for response.output_text.delta and response.done
- Internal helper: messages → input adds required fields
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import pytest

from ccproxy.adapters.openai.anthropic_response_adapter import (
    AnthropicResponseAPIAdapter,
)


@pytest.fixture
def adapter() -> AnthropicResponseAPIAdapter:
    """Create adapter instance for tests."""
    return AnthropicResponseAPIAdapter()


@pytest.mark.asyncio
async def test_adapt_request_basic_mapping(
    adapter: AnthropicResponseAPIAdapter,
) -> None:
    """Maps core fields and normalizes for Response API usage."""
    request: dict[str, Any] = {
        "model": "claude-3-5-sonnet-20241022",
        "messages": [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}],
        "max_tokens": 128,
        "system": "You are helpful.",
        "tools": [
            {"name": "t", "description": "d", "input_schema": {"type": "object"}}
        ],
        "temperature": 0.2,
        "stream": True,
        "parallel_tool_calls": True,
    }

    out = await adapter.adapt_request(request)

    # Forced overrides
    assert out["model"] == "gpt-5"
    assert out["store"] is False

    # Field mappings and passthrough
    assert isinstance(out["input"], list)
    assert out["input"][0]["role"] == "user"
    assert out["input"][0]["type"] == "message"
    assert out["instructions"] == "You are helpful."
    assert out["tools"] == request["tools"]
    assert out["stream"] is True
    assert out["parallel_tool_calls"] is True
    assert out["temperature"] == 0.2

    # max_completion_tokens is created then removed by adapter
    assert "max_completion_tokens" not in out


@pytest.mark.asyncio
async def test_adapt_response_from_nested_output(
    adapter: AnthropicResponseAPIAdapter,
) -> None:
    """Converts nested Response API shape with output list of messages."""
    response: dict[str, Any] = {
        "id": "resp_1",
        "model": "gpt-5",
        "response": {
            "output": [
                {
                    "type": "message",
                    "content": [
                        {"type": "text", "text": "Hello world"},
                        {"type": "output_text", "text": "!"},
                    ],
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        },
    }

    out = await adapter.adapt_response(response)

    assert out["content"] == [
        {"type": "text", "text": "Hello world"},
        {"type": "text", "text": "!"},
    ]
    assert out["stop_reason"] == "end_turn"
    assert out["usage"] == {"input_tokens": 10, "output_tokens": 5}
    assert out["model"] == "gpt-5"
    assert out["id"] == "resp_1"


@pytest.mark.asyncio
async def test_adapt_response_from_choices_with_tool_calls(
    adapter: AnthropicResponseAPIAdapter,
) -> None:
    """Converts OpenAI-like choices with tool calls to Anthropic blocks."""
    response: dict[str, Any] = {
        "id": "resp_2",
        "model": "gpt-5",
        "choices": [
            {
                "message": {
                    "content": "Here is data",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "function": {
                                "name": "get_data",
                                "arguments": json.dumps({"q": 1}),
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
    }

    out = await adapter.adapt_response(response)
    assert {"type": "text", "text": "Here is data"} in out["content"]
    tool_blocks = [b for b in out["content"] if b.get("type") == "tool_use"]
    assert len(tool_blocks) == 1
    tool = tool_blocks[0]
    assert tool["id"] == "call_1"
    assert tool["name"] == "get_data"
    assert tool["input"] == {"q": 1}


@pytest.mark.asyncio
async def test_adapt_response_default_empty_content(
    adapter: AnthropicResponseAPIAdapter,
) -> None:
    """Produces a default empty text block when nothing found."""
    out = await adapter.adapt_response({"model": "gpt-5", "id": "x"})
    assert out["content"] == [{"type": "text", "text": ""}]
    assert out["stop_reason"] == "end_turn"


@pytest.mark.asyncio
async def test_adapt_stream_sequence(adapter: AnthropicResponseAPIAdapter) -> None:
    """Converts response.output_text.delta and response.done to Anthropic stream events."""

    async def mock_stream() -> AsyncIterator[dict[str, Any]]:
        yield {"type": "response.output_text.delta", "delta": "Hello"}
        yield {"type": "response.output_text.delta", "delta": " world"}
        yield {"type": "response.done"}

    results: list[dict[str, Any]] = []
    async for ev in adapter.adapt_stream(mock_stream()):
        results.append(ev)

    # First message_start and block_start, then two deltas, block_stop, message_delta, message_stop
    assert results[0]["type"] == "message_start"
    assert results[1]["type"] == "content_block_start"
    assert (
        results[2]["type"] == "content_block_delta"
        and results[2]["delta"]["text"] == "Hello"
    )
    assert (
        results[3]["type"] == "content_block_delta"
        and results[3]["delta"]["text"] == " world"
    )
    assert results[-3]["type"] == "content_block_stop"
    assert (
        results[-2]["type"] == "message_delta"
        and results[-2]["delta"]["stop_reason"] == "end_turn"
    )
    assert results[-1]["type"] == "message_stop"


def test_convert_messages_to_input(adapter: AnthropicResponseAPIAdapter) -> None:
    """Ensures helper adds type=message and preserves ids."""
    messages = [
        {"id": "m1", "role": "user", "content": [{"type": "text", "text": "Hi"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "Yo"}]},
    ]

    out = adapter._convert_messages_to_input(messages)
    assert out[0]["type"] == "message" and out[0]["id"] == "m1"
    assert out[1]["type"] == "message"
