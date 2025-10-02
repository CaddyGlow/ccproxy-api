"""Tests for response handling utilities."""

from datetime import UTC, datetime
from typing import Any

import httpx

from ccproxy.testing.config import RequestScenario
from ccproxy.testing.response_handlers import MetricsExtractor, ResponseHandler


def make_scenario(**overrides: Any) -> RequestScenario:
    base: dict[str, Any] = {
        "model": "m",
        "message_type": "chat",
        "streaming": False,
        "response_type": "success",
        "timestamp": datetime.now(UTC),
        "api_format": "openai",
        "headers": {},
    }
    base.update(overrides)
    return RequestScenario(**base)


def test_process_standard_openai_response() -> None:
    handler = ResponseHandler()
    payload = {
        "usage": {"prompt_tokens": 5, "completion_tokens": 3},
        "choices": [{"message": {"content": "hello world"}}],
    }
    response = httpx.Response(200, json=payload)
    result = handler.process_response(response, make_scenario())

    assert result["tokens_input"] == 5
    assert result["tokens_output"] == 3
    assert result["content_preview"] == "hello world"


def test_process_standard_anthropic_response() -> None:
    handler = ResponseHandler()
    payload = {
        "usage": {"input_tokens": 2, "output_tokens": 4},
        "content": [{"type": "text", "text": "anthropic reply"}],
    }
    response = httpx.Response(200, json=payload)
    result = handler.process_response(response, make_scenario(api_format="anthropic"))

    assert result["tokens_input"] == 2
    assert result["content_preview"] == "anthropic reply"


def test_process_standard_response_json_error() -> None:
    handler = ResponseHandler()
    response = httpx.Response(200, content=b"not json")
    result = handler.process_response(response, make_scenario())

    assert "Failed to parse" in result["error"]


def test_process_streaming_openai_response() -> None:
    handler = ResponseHandler()
    content = (
        'data: {"choices": [{"delta": {"content": "Hello"}}]}\n\n'
        'data: {"choices": [{"delta": {"content": " World"}}]}\n\n'
        "data: [DONE]\n\n"
    )
    response = httpx.Response(
        200,
        content=content.encode(),
        headers={"content-type": "text/event-stream"},
    )

    result = handler.process_response(response, make_scenario(streaming=True))

    assert result["chunk_count"] == 2
    assert result["total_content"] == "Hello World"


def test_metrics_extractor_openai_and_anthropic() -> None:
    openai_usage = {"usage": {"prompt_tokens": 1, "completion_tokens": 2}}
    anthropic_usage = {
        "usage": {
            "input_tokens": 3,
            "output_tokens": 4,
            "cache_read_input_tokens": 5,
            "cache_creation_input_tokens": 6,
        }
    }

    openai_metrics = MetricsExtractor.extract_token_metrics(openai_usage, "openai")
    anthropic_metrics = MetricsExtractor.extract_token_metrics(
        anthropic_usage, "anthropic"
    )

    assert openai_metrics["input_tokens"] == 1
    assert anthropic_metrics["cache_write_tokens"] == 6

    content = MetricsExtractor.extract_content(
        {
            "choices": [{"message": {"content": "hi"}}],
            "content": [{"type": "text", "text": "anthropic"}],
        },
        "openai",
    )
    assert content == "hi"

    anthropic_content = MetricsExtractor.extract_content(
        {"content": [{"type": "text", "text": "anthropic"}]}, "anthropic"
    )
    assert anthropic_content == "anthropic"
