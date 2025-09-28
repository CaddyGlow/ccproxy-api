from collections.abc import AsyncIterator
from typing import Any

import pytest

from ccproxy.llms.formatters.openai_to_anthropic import (
    convert__openai_chat_to_anthropic_messages__response,
    convert__openai_chat_to_anthropic_messages__stream,
)
from ccproxy.llms.models import anthropic as anthropic_models
from ccproxy.llms.models import openai as openai_models


@pytest.mark.unit
def test_stop_reason_mapping_stop() -> None:
    resp = openai_models.ChatCompletionResponse(
        id="r1",
        object="chat.completion",
        created=0,
        model="gpt-x",
        choices=[
            {
                "index": 0,
                "message": {"role": "assistant", "content": "ok"},
                "finish_reason": "stop",
            }
        ],
        usage=openai_models.CompletionUsage(
            prompt_tokens=1, completion_tokens=2, total_tokens=3
        ),
    )
    out = convert__openai_chat_to_anthropic_messages__response(resp)
    assert out.stop_reason == "end_turn"


@pytest.mark.unit
def test_stop_reason_mapping_length() -> None:
    resp = openai_models.ChatCompletionResponse(
        id="r1",
        object="chat.completion",
        created=0,
        model="gpt-x",
        choices=[
            {
                "index": 0,
                "message": {"role": "assistant", "content": "ok"},
                "finish_reason": "length",
            }
        ],
        usage=openai_models.CompletionUsage(
            prompt_tokens=1, completion_tokens=2, total_tokens=3
        ),
    )
    out = convert__openai_chat_to_anthropic_messages__response(resp)
    assert out.stop_reason == "max_tokens"


@pytest.mark.unit
def test_usage_mapping_cached_tokens() -> None:
    usage = openai_models.CompletionUsage(
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
        prompt_tokens_details=openai_models.PromptTokensDetails(
            cached_tokens=7, audio_tokens=0
        ),
        completion_tokens_details=openai_models.CompletionTokensDetails(
            reasoning_tokens=0,
            audio_tokens=0,
            accepted_prediction_tokens=0,
            rejected_prediction_tokens=0,
        ),
    )
    resp = openai_models.ChatCompletionResponse(
        id="r1",
        object="chat.completion",
        created=0,
        model="gpt-x",
        choices=[
            {
                "index": 0,
                "message": {"role": "assistant", "content": "ok"},
                "finish_reason": "stop",
            }
        ],
        usage=usage,
    )
    out = convert__openai_chat_to_anthropic_messages__response(resp)
    assert out.usage.input_tokens == 10
    assert out.usage.output_tokens == 5
    assert (out.usage.cache_read_input_tokens or 0) == 7


@pytest.mark.unit
def test_tool_calls_strict_arguments_json() -> None:
    msg: dict[str, Any] = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "tool_1",
                "type": "function",
                "function": {"name": "do", "arguments": '{"a":1}'},
            }
        ],
    }
    resp = openai_models.ChatCompletionResponse(
        id="r1",
        object="chat.completion",
        created=0,
        model="gpt-x",
        choices=[{"index": 0, "message": msg, "finish_reason": "tool_calls"}],
        usage=openai_models.CompletionUsage(
            prompt_tokens=1, completion_tokens=1, total_tokens=2
        ),
    )
    out = convert__openai_chat_to_anthropic_messages__response(resp)
    names: list[str] = [
        b.name for b in out.content if getattr(b, "type", None) == "tool_use"
    ]  # type: ignore[list-item]
    assert names == ["do"]


@pytest.mark.unit
def test_tool_calls_strict_arguments_invalid_raises() -> None:
    msg2: dict[str, Any] = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "tool_1",
                "type": "function",
                "function": {"name": "do", "arguments": "not-json"},
            }
        ],
    }
    resp = openai_models.ChatCompletionResponse(
        id="r1",
        object="chat.completion",
        created=0,
        model="gpt-x",
        choices=[{"index": 0, "message": msg2, "finish_reason": "tool_calls"}],
        usage=openai_models.CompletionUsage(
            prompt_tokens=1, completion_tokens=1, total_tokens=2
        ),
    )
    with pytest.raises(ValueError):
        _ = convert__openai_chat_to_anthropic_messages__response(resp)


@pytest.mark.unit
def test_response_usage_mapping() -> None:
    """Test that ResponseUsage model can be properly instantiated with minimal fields."""
    # Create a ResponseUsage instance with minimal fields
    usage = openai_models.ResponseUsage(
        input_tokens=10, output_tokens=5, total_tokens=15
    )

    # Verify default values were applied
    assert usage.input_tokens_details.cached_tokens == 0
    assert usage.output_tokens_details.reasoning_tokens == 0

    # Create a more complete ResponseUsage instance
    details_usage = openai_models.ResponseUsage(
        input_tokens=10,
        output_tokens=5,
        total_tokens=15,
        input_tokens_details=openai_models.InputTokensDetails(cached_tokens=3),
        output_tokens_details=openai_models.OutputTokensDetails(reasoning_tokens=2),
    )

    # Verify values were properly set
    assert details_usage.input_tokens_details.cached_tokens == 3
    assert details_usage.output_tokens_details.reasoning_tokens == 2


@pytest.mark.unit
def test_response_usage_from_dict() -> None:
    """Test that ResponseUsage model can be properly created from a dictionary."""
    from ccproxy.llms.formatters.utils import openai_response_usage_snapshot

    # Create a dictionary with minimal fields - this should work now
    usage_dict = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}

    # Test our utility function can handle dictionary input without crashing
    snapshot = openai_response_usage_snapshot(usage_dict)
    assert snapshot.input_tokens == 10
    assert snapshot.output_tokens == 5

    # Create a ResponseUsage model from dictionary
    usage_model = openai_models.ResponseUsage.model_validate(
        {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
    )

    # Ensure the model was created properly
    assert usage_model.input_tokens == 10
    assert usage_model.output_tokens == 5
    assert usage_model.total_tokens == 15
    assert usage_model.input_tokens_details is not None
    assert usage_model.output_tokens_details is not None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_openai_chat_stream_to_anthropic_events_with_tool_call() -> None:
    async def gen() -> AsyncIterator[dict[str, Any]]:
        yield {
            "id": "chunk-1",
            "model": "gpt-test",
            "choices": [{"delta": {"role": "assistant"}}],
        }
        yield {
            "id": "chunk-1",
            "model": "gpt-test",
            "choices": [
                {
                    "delta": {
                        "content": "Weather report:",
                    }
                }
            ],
        }
        yield {
            "id": "chunk-1",
            "model": "gpt-test",
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "id": "tool_1",
                                "index": 0,
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"city":"Sea',
                                },
                            }
                        ]
                    }
                }
            ],
        }
        yield {
            "id": "chunk-1",
            "model": "gpt-test",
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "id": "tool_1",
                                "index": 0,
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": 'ttle","units":"metric"}',
                                },
                            }
                        ]
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 6,
                "input_tokens": 10,
                "output_tokens": 6,
            },
        }

    events = [
        evt async for evt in convert__openai_chat_to_anthropic_messages__stream(gen())
    ]

    assert events, "expected streamed events"
    event_types = [getattr(evt, "type", None) for evt in events]
    assert event_types[0] == "message_start"
    assert "content_block_start" in event_types
    assert event_types[-1] == "message_stop"

    tool_start = next(
        evt
        for evt in events
        if isinstance(evt, anthropic_models.ContentBlockStartEvent)
        and getattr(evt.content_block, "type", None) == "tool_use"
    )
    assert tool_start.content_block.input == {
        "city": "Seattle",
        "units": "metric",
    }

    message_delta = next(
        evt for evt in events if isinstance(evt, anthropic_models.MessageDeltaEvent)
    )
    assert message_delta.delta.stop_reason == "tool_use"
    assert message_delta.usage.input_tokens == 10
    assert message_delta.usage.output_tokens == 6
