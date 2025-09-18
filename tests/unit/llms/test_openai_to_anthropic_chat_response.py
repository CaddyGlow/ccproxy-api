from typing import Any

import pytest

from ccproxy.llms.formatters.openai_to_anthropic.helpers import (
    convert__openai_chat_to_anthropic_messages__response,
)
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
