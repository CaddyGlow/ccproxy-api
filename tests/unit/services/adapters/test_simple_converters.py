"""Test the simplified dict-based conversion functions."""

import pytest

from ccproxy.services.adapters.simple_converters import (
    convert_anthropic_to_openai_response,
    convert_openai_to_anthropic_request,
)


@pytest.mark.asyncio
async def test_openai_to_anthropic_request_conversion():
    """Test OpenAI to Anthropic request conversion."""
    openai_request = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "Hello, world!"}],
        "max_tokens": 100,
    }

    # Should not raise an exception
    result = await convert_openai_to_anthropic_request(openai_request)

    # Basic validation that conversion happened
    assert isinstance(result, dict)
    assert "model" in result
    assert "messages" in result
    assert "max_tokens" in result


@pytest.mark.asyncio
async def test_anthropic_to_openai_response_conversion():
    """Test Anthropic to OpenAI response conversion."""
    anthropic_response = {
        "id": "msg_123",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "Hello! How can I help you today?"}],
        "model": "claude-3-sonnet-20240229",
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 10, "output_tokens": 20},
    }

    # Should not raise an exception
    result = await convert_anthropic_to_openai_response(anthropic_response)

    # Basic validation that conversion happened
    assert isinstance(result, dict)
    assert "id" in result
    assert "choices" in result
    assert "usage" in result
