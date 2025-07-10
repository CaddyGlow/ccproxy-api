"""Tests for OpenAI to Anthropic translation service."""

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from ccproxy.adapters.openai import OpenAIAdapter
from ccproxy.models.openai import (
    OpenAIChatCompletionRequest,
    OpenAIMessage,
    OpenAIResponseFormat,
    OpenAIStreamOptions,
)


@pytest.mark.unit
class TestOpenAIAdapter:
    """Test OpenAI to Anthropic translator."""

    def test_translate_basic_request(self):
        """Test basic request translation."""
        translator = OpenAIAdapter()

        openai_request = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 100,
        }

        result = translator.adapt_request(openai_request)

        assert result["model"] == "claude-3-5-sonnet-20241022"  # Mapped model
        assert result["messages"] == [{"role": "user", "content": "Hello"}]
        assert result["max_tokens"] == 100

    def test_translate_request_with_metadata(self):
        """Test request translation with metadata fields."""
        translator = OpenAIAdapter()

        openai_request = {
            "model": "claude-3-5-sonnet-20241022",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 100,
            "user": "user-123",
            "metadata": {"session_id": "abc", "request_type": "test"},
        }

        result = translator.adapt_request(openai_request)

        assert result["metadata"] == {
            "user_id": "user-123",
            "session_id": "abc",
            "request_type": "test",
        }

    def test_translate_request_with_response_format_json(self):
        """Test request translation with JSON response format."""
        translator = OpenAIAdapter()

        openai_request = {
            "model": "claude-3-5-sonnet-20241022",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Generate JSON"},
            ],
            "max_tokens": 100,
            "response_format": {"type": "json_object"},
        }

        result = translator.adapt_request(openai_request)

        # JSON mode should be added to system prompt
        assert "You must respond with valid JSON only." in result["system"]

    def test_translate_request_with_response_format_json_schema(self):
        """Test request translation with JSON schema response format."""
        translator = OpenAIAdapter()

        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        openai_request = {
            "model": "claude-3-5-sonnet-20241022",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Generate JSON"},
            ],
            "max_tokens": 100,
            "response_format": {"type": "json_schema", "json_schema": schema},
        }

        result = translator.adapt_request(openai_request)

        # JSON schema should be added to system prompt
        assert (
            "You must respond with valid JSON that conforms to this schema:"
            in result["system"]
        )
        assert str(schema) in result["system"]

    def test_translate_request_with_unsupported_fields(self):
        """Test request translation with fields not supported by Anthropic."""
        translator = OpenAIAdapter()

        openai_request = {
            "model": "claude-3-5-sonnet-20241022",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 100,
            "seed": 42,
            "logprobs": True,
            "top_logprobs": 5,
            "store": True,
        }

        # Should not raise error, just log warnings
        result = translator.adapt_request(openai_request)

        # These fields should not be in the result
        assert "seed" not in result
        assert "logprobs" not in result
        assert "top_logprobs" not in result
        assert "store" not in result

    def test_translate_response_with_thinking_blocks(self):
        """Test response translation with thinking blocks."""
        translator = OpenAIAdapter()

        anthropic_response = {
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "content": [
                {"type": "thinking", "text": "Let me think about this..."},
                {"type": "text", "text": "The answer is 42."},
            ],
            "model": "claude-3-5-sonnet-20241022",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 20},
        }

        result = translator.adapt_response(anthropic_response)

        # Check that thinking blocks are included with marker
        assert "[Thinking]" in result["choices"][0]["message"]["content"]
        assert (
            "Let me think about this..." in result["choices"][0]["message"]["content"]
        )
        assert "The answer is 42." in result["choices"][0]["message"]["content"]

        # Check model is preserved
        assert result["model"] == "claude-3-5-sonnet-20241022"

    def test_translate_response_with_new_stop_reasons(self):
        """Test response translation with new stop reasons."""
        translator = OpenAIAdapter()

        stop_reason_mappings = [
            ("pause_turn", "stop"),
            ("refusal", "content_filter"),
            ("end_turn", "stop"),
            ("max_tokens", "length"),
            ("tool_use", "tool_calls"),
            ("stop_sequence", "stop"),
        ]

        for anthropic_reason, expected_openai_reason in stop_reason_mappings:
            anthropic_response = {
                "id": "msg_123",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "Test"}],
                "model": "claude-3-5-sonnet-20241022",
                "stop_reason": anthropic_reason,
                "usage": {"input_tokens": 10, "output_tokens": 20},
            }

            result = translator.adapt_response(anthropic_response)

            assert result["choices"][0]["finish_reason"] == expected_openai_reason

    def test_translate_response_with_system_fingerprint(self):
        """Test that response includes system_fingerprint."""
        translator = OpenAIAdapter()

        anthropic_response = {
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "Hello"}],
            "model": "claude-3-5-sonnet-20241022",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 20},
        }

        result = translator.adapt_response(anthropic_response)

        # Should have system_fingerprint
        assert "system_fingerprint" in result
        assert result["system_fingerprint"] is not None
        assert result["system_fingerprint"].startswith("fp_")

    @pytest.mark.asyncio
    async def test_translate_streaming_with_thinking_blocks(self):
        """Test streaming translation with thinking blocks."""
        translator = OpenAIAdapter()

        async def mock_stream():
            yield {"type": "message_start"}
            yield {"type": "content_block_start", "content_block": {"type": "thinking"}}
            yield {
                "type": "content_block_delta",
                "delta": {"type": "thinking_delta", "thinking": "Processing..."},
            }
            yield {
                "type": "content_block_delta",
                "delta": {"type": "text_delta", "text": "The answer is 42."},
            }
            yield {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {"input_tokens": 10, "output_tokens": 20},
            }

        chunks = []
        async for chunk in translator.adapt_stream(mock_stream()):
            chunks.append(chunk)

        # Check that we have chunks
        assert len(chunks) > 0

        # Check thinking block marker
        thinking_chunk = next(
            c
            for c in chunks
            if "[Thinking]" in c["choices"][0]["delta"].get("content", "")
        )
        assert thinking_chunk is not None

        # Check thinking content
        thinking_content_chunk = next(
            c
            for c in chunks
            if "Processing..." in c["choices"][0]["delta"].get("content", "")
        )
        assert thinking_content_chunk is not None

        # Check regular text
        text_chunk = next(
            c
            for c in chunks
            if "The answer is 42." in c["choices"][0]["delta"].get("content", "")
        )
        assert text_chunk is not None

        # Check final chunk has usage
        final_chunk = chunks[-1]
        assert final_chunk["choices"][0]["finish_reason"] == "stop"
        assert final_chunk["usage"]["prompt_tokens"] == 10
        assert final_chunk["usage"]["completion_tokens"] == 20
        assert final_chunk["usage"]["total_tokens"] == 30

    @pytest.mark.asyncio
    async def test_translate_streaming_with_new_stop_reasons(self):
        """Test streaming translation with new stop reasons."""
        translator = OpenAIAdapter()

        stop_reason_mappings = [("pause_turn", "stop"), ("refusal", "content_filter")]

        for anthropic_reason, expected_openai_reason in stop_reason_mappings:

            async def mock_stream(reason=anthropic_reason):
                yield {"type": "message_start"}
                yield {
                    "type": "content_block_delta",
                    "delta": {"type": "text_delta", "text": "Content"},
                }
                yield {
                    "type": "message_delta",
                    "delta": {"stop_reason": reason},
                }

            chunks = []
            async for chunk in translator.adapt_stream(mock_stream()):
                chunks.append(chunk)

            # Find final chunk with finish_reason
            final_chunk = next(
                c for c in chunks if c["choices"][0].get("finish_reason") is not None
            )
            assert final_chunk["choices"][0]["finish_reason"] == expected_openai_reason


@pytest.mark.unit
class TestModelMapping:
    """Test model name mapping."""

    def test_openai_to_claude_model_mapping(self):
        """Test OpenAI model names map to Claude models."""
        from ccproxy.adapters.openai import map_openai_model_to_claude

        mappings = [
            ("gpt-4o-mini", "claude-3-5-haiku-20241022"),
            ("gpt-4o-mini-2024-07-18", "claude-3-5-haiku-20241022"),  # exact match
            ("o1-mini", "claude-3-5-haiku-20241022"),
            ("gpt-4o", "claude-3-5-sonnet-20241022"),
            ("gpt-4o-2024-11-20", "claude-3-5-sonnet-20241022"),  # exact match
        ]

        for openai_model, expected_claude_model in mappings:
            result = map_openai_model_to_claude(openai_model)
            assert result == expected_claude_model

    def test_claude_model_passthrough(self):
        """Test Claude models are mapped to defaults (new behavior)."""
        from ccproxy.adapters.openai import map_openai_model_to_claude

        claude_models = [
            "claude-3-5-sonnet-20241022",
            "claude-3-opus-20240229", 
            "claude-opus-4-20250514",
            "claude-sonnet-4-20250514",
        ]

        for model in claude_models:
            result = map_openai_model_to_claude(model)
            # New behavior: unknown models map to default
            assert result == "claude-3-5-sonnet-20241022"

    def test_unknown_model_passthrough(self):
        """Test unknown models are mapped to default (new behavior)."""
        from ccproxy.adapters.openai import map_openai_model_to_claude

        unknown_model = "some-unknown-model"
        result = map_openai_model_to_claude(unknown_model)
        # New behavior: unknown models map to default
        assert result == "claude-3-5-sonnet-20241022"
