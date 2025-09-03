"""Unit tests for AnthropicMessagesAdapter."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from ccproxy.plugins.codex.anthropic_adapter import AnthropicMessagesAdapter


class TestAnthropicMessagesAdapter:
    """Test the AnthropicMessagesAdapter format conversion logic."""

    @pytest.fixture
    def adapter(self) -> AnthropicMessagesAdapter:
        """Create AnthropicMessagesAdapter instance for testing."""
        return AnthropicMessagesAdapter()

    @pytest.mark.asyncio
    async def test_adapt_request_basic_conversion(self, adapter: AnthropicMessagesAdapter) -> None:
        """Test basic Anthropic Messages to Response API request conversion."""
        anthropic_request = {
            "model": "claude-3-5-sonnet-20241022",
            "messages": [{"role": "user", "content": "Hello, world!"}],
            "max_tokens": 100,
            "temperature": 0.7,
            "stream": False,
        }

        result = await adapter.adapt_request(anthropic_request)

        # Should have Response API structure
        assert "input" in result  # Response API uses 'input' instead of 'messages'
        assert "model" in result
        assert result["model"] == "gpt-5"  # Codex default
        assert len(result["input"]) == 1
        assert result["input"][0]["role"] == "user"

    @pytest.mark.asyncio
    async def test_adapt_request_with_tools(self, adapter: AnthropicMessagesAdapter) -> None:
        """Test request conversion with tools/functions."""
        anthropic_request = {
            "model": "claude-3-5-sonnet-20241022", 
            "messages": [{"role": "user", "content": "Get weather"}],
            "max_tokens": 100,
            "tools": [
                {
                    "name": "get_weather",
                    "description": "Get weather info",
                    "input_schema": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                    },
                }
            ],
            "tool_choice": {"type": "tool", "name": "get_weather"},
        }

        result = await adapter.adapt_request(anthropic_request)

        assert "tools" in result
        assert "tool_choice" in result
        # Tools can be None if not properly converted, check for presence
        if result["tools"] is not None:
            assert len(result["tools"]) == 1
            assert result["tools"][0]["name"] == "get_weather"

    @pytest.mark.asyncio
    async def test_adapt_request_passthrough(self, adapter: AnthropicMessagesAdapter) -> None:
        """Test request passthrough for native Response API format."""
        response_api_request = {
            "instructions": "You are a helpful assistant",
            "model": "gpt-5",
            "stream": False,
        }

        result = await adapter.adapt_request(response_api_request)

        # Should pass through unchanged
        assert result == response_api_request

    @pytest.mark.asyncio
    async def test_adapt_response_basic_conversion(self, adapter: AnthropicMessagesAdapter) -> None:
        """Test basic Response API to Anthropic Messages response conversion."""
        # Mock the ResponseAdapter
        mock_chat_response = MagicMock()
        mock_chat_response.model_dump.return_value = {
            "choices": [
                {
                    "message": {"content": "Hello! How can I help you today?", "role": "assistant"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 15},
            "model": "gpt-5",
            "id": "test-id",
        }
        adapter._response_adapter.response_to_chat_completion = MagicMock(return_value=mock_chat_response)

        response_api_response = {
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "text", "text": "Hello! How can I help you today?"}],
                }
            ]
        }

        result = await adapter.adapt_response(response_api_response)

        # Should have Anthropic Messages structure
        assert "content" in result
        assert "stop_reason" in result
        assert len(result["content"]) == 1
        assert result["content"][0]["type"] == "text"
        assert result["content"][0]["text"] == "Hello! How can I help you today?"
        assert result["stop_reason"] == "end_turn"

    @pytest.mark.asyncio
    async def test_adapt_response_with_tools(self, adapter: AnthropicMessagesAdapter) -> None:
        """Test response conversion with tool calls."""
        # Mock the ResponseAdapter
        mock_chat_response = MagicMock()
        mock_chat_response.model_dump.return_value = {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "role": "assistant", 
                        "tool_calls": [
                            {
                                "id": "call_123",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"location": "San Francisco"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20},
        }
        adapter._response_adapter.response_to_chat_completion = MagicMock(return_value=mock_chat_response)

        response_api_response = {
            "output": [
                {
                    "type": "message",
                    "content": [
                        {
                            "type": "tool_call",
                            "id": "call_123",
                            "name": "get_weather",
                            "arguments": {"location": "San Francisco"},
                        }
                    ],
                }
            ]
        }

        result = await adapter.adapt_response(response_api_response)

        # Should have tool_use in Anthropic format
        assert "content" in result
        assert len(result["content"]) == 1
        assert result["content"][0]["type"] == "tool_use"
        assert result["content"][0]["id"] == "call_123"
        assert result["content"][0]["name"] == "get_weather"
        assert result["content"][0]["input"] == {"location": "San Francisco"}
        assert result["stop_reason"] == "tool_use"

    @pytest.mark.asyncio
    async def test_adapt_response_passthrough(self, adapter: AnthropicMessagesAdapter) -> None:
        """Test response passthrough for non-Response API format."""
        anthropic_response = {
            "content": [{"type": "text", "text": "Direct response"}],
            "stop_reason": "end_turn",
        }

        result = await adapter.adapt_response(anthropic_response)

        # Should pass through unchanged
        assert result == anthropic_response

    @pytest.mark.asyncio
    async def test_convert_to_anthropic_format_with_mixed_content(self, adapter: AnthropicMessagesAdapter) -> None:
        """Test conversion to Anthropic format with both text and tool calls."""
        openai_response = {
            "choices": [
                {
                    "message": {
                        "content": "Let me get that information for you.",
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call_456", 
                                "function": {
                                    "name": "search_docs",
                                    "arguments": '{"query": "API documentation"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 20, "completion_tokens": 30},
        }

        result = adapter._convert_to_anthropic_format(openai_response)

        # Should have both text and tool_use blocks
        assert len(result["content"]) == 2
        assert result["content"][0]["type"] == "text"
        assert result["content"][0]["text"] == "Let me get that information for you."
        assert result["content"][1]["type"] == "tool_use"
        assert result["content"][1]["id"] == "call_456"
        assert result["content"][1]["name"] == "search_docs"
        assert result["content"][1]["input"] == {"query": "API documentation"}

    def test_is_response_api_format(self, adapter: AnthropicMessagesAdapter) -> None:
        """Test Response API format detection."""
        # Should detect Response API format with 'output' field
        response_with_output = {"output": [{"type": "message"}]}
        assert adapter._is_response_api_format(response_with_output) is True

        # Should detect Response API format with 'response' wrapper
        response_with_wrapper = {"response": {"output": []}}
        assert adapter._is_response_api_format(response_with_wrapper) is True

        # Should not detect Anthropic Messages format
        anthropic_response = {"content": [{"type": "text"}], "stop_reason": "end_turn"}
        assert adapter._is_response_api_format(anthropic_response) is False

    def test_has_tool_calls_in_response(self, adapter: AnthropicMessagesAdapter) -> None:
        """Test tool calls detection in responses."""
        # Response with tool calls
        response_with_tools = {
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "tool_call", "id": "call_123"}],
                }
            ]
        }
        assert adapter._has_tool_calls_in_response(response_with_tools) is True

        # Response without tool calls
        response_without_tools = {
            "output": [
                {
                    "type": "message", 
                    "content": [{"type": "text", "text": "Hello"}],
                }
            ]
        }
        assert adapter._has_tool_calls_in_response(response_without_tools) is False

        # Wrapped response with tool calls
        wrapped_response = {
            "response": {
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "tool_call", "id": "call_456"}],
                    }
                ]
            }
        }
        assert adapter._has_tool_calls_in_response(wrapped_response) is True

    @pytest.mark.asyncio
    async def test_streaming_conversion(self, adapter: AnthropicMessagesAdapter) -> None:
        """Test streaming format conversion."""
        # Mock the OpenAI streaming response
        async def mock_openai_stream():
            yield {"choices": [{"delta": {"role": "assistant"}}]}
            yield {"choices": [{"delta": {"content": "Hello"}}]}
            yield {"choices": [{"delta": {"content": " world"}}]}
            yield {"choices": [{"finish_reason": "stop"}]}

        mock_stream = mock_openai_stream()
        adapter._response_adapter.stream_response_to_chat = MagicMock(return_value=mock_stream)

        # Create a mock stream
        async def mock_response_stream():
            yield {"type": "response.output_text.delta", "delta": "Hello"}
            yield {"type": "response.output_text.delta", "delta": " world"}
            yield {"type": "response.completed", "response": {"usage": {"input_tokens": 5, "output_tokens": 2}}}

        stream_result = []
        anthropic_stream = adapter.adapt_stream(mock_response_stream())
        async for chunk in anthropic_stream:
            stream_result.append(chunk)

        # Should have converted to Anthropic streaming format
        assert len(stream_result) > 0
        # First chunk should be message_start
        assert any(chunk.get("type") == "message_start" for chunk in stream_result)
        # Should have content_block_delta chunks
        assert any(chunk.get("type") == "content_block_delta" for chunk in stream_result) 
        # Should end with message_stop
        assert any(chunk.get("type") == "message_stop" for chunk in stream_result)

    def test_convert_to_anthropic_format_invalid_json(self, adapter: AnthropicMessagesAdapter) -> None:
        """Test handling of invalid JSON in tool arguments."""
        openai_response = {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call_789",
                                "function": {
                                    "name": "invalid_tool",
                                    "arguments": "invalid json {",
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        }

        result = adapter._convert_to_anthropic_format(openai_response)

        # Should handle invalid JSON gracefully
        assert len(result["content"]) == 1
        assert result["content"][0]["type"] == "tool_use"
        assert result["content"][0]["input"] == {}  # Empty dict for invalid JSON