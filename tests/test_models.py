"""Tests for Pydantic models."""

from typing import Any

import pytest
from pydantic import ValidationError

from ccproxy.models.errors import (
    AnthropicError,
    ErrorDetail,
    InvalidRequestError,
    create_error_response,
)
from ccproxy.models.requests import (
    ImageContent,
    Message,
    TextContent,
    ToolDefinition,
    Usage,
)
from ccproxy.models.responses import (
    ChatCompletionResponse,
    StreamingChatCompletionResponse,
    ToolUse,
)


@pytest.mark.unit
class TestMessage:
    """Test Message model."""

    def test_user_message_with_text(self):
        """Test user message with text content."""
        message_data: dict[str, Any] = {
            "role": "user",
            "content": "Hello, how are you?",
        }

        message = Message(**message_data)

        assert message.role == "user"
        assert message.content == "Hello, how are you?"

    def test_user_message_with_content_blocks(self):
        """Test user message with content blocks."""
        message_data: dict[str, Any] = {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==",
                    },
                },
            ],
        }

        message = Message(**message_data)

        assert message.role == "user"
        assert isinstance(message.content, list)
        assert len(message.content) == 2
        assert isinstance(message.content[0], TextContent)
        assert isinstance(message.content[1], ImageContent)

    def test_assistant_message(self):
        """Test assistant message."""
        message_data: dict[str, Any] = {
            "role": "assistant",
            "content": "I'm doing well, thank you!",
        }

        message = Message(**message_data)

        assert message.role == "assistant"
        assert message.content == "I'm doing well, thank you!"

    def test_invalid_role(self):
        """Test invalid role validation."""
        message_data: dict[str, Any] = {"role": "invalid", "content": "Hello"}

        with pytest.raises(ValidationError):
            Message(**message_data)


@pytest.mark.unit
class TestErrorModels:
    """Test error models."""

    def test_anthropic_error(self):
        """Test AnthropicError model."""
        error_detail = ErrorDetail(
            type="invalid_request_error", message="Invalid request"
        )
        error = AnthropicError(error=error_detail)

        assert error.type == "error"
        assert error.error.type == "invalid_request_error"
        assert error.error.message == "Invalid request"

    def test_invalid_request_error(self):
        """Test InvalidRequestError model."""
        error = InvalidRequestError()

        assert error.type == "error"
        assert error.error.type == "invalid_request_error"
        assert error.error.message == "Invalid request"

    def test_create_error_response(self):
        """Test create_error_response function."""
        error_dict, status_code = create_error_response(
            "invalid_request_error", "Test error message", 400
        )

        assert status_code == 400
        assert error_dict["type"] == "error"
        assert error_dict["error"]["type"] == "invalid_request_error"
        assert error_dict["error"]["message"] == "Test error message"


@pytest.mark.unit
class TestChatCompletionResponse:
    """Test ChatCompletionResponse model."""

    def test_valid_response(self, sample_claude_response):
        """Test valid chat completion response."""
        response = ChatCompletionResponse(**sample_claude_response)

        assert response.id == "msg_test123"
        assert response.type == "message"
        assert response.role == "assistant"
        assert response.model == "claude-3-5-sonnet-20241022"
        assert response.stop_reason == "end_turn"
        assert response.usage.input_tokens == 10
        assert response.usage.output_tokens == 15
        assert response.usage.input_tokens == 10
        assert response.usage.output_tokens == 15


@pytest.mark.unit
class TestUsage:
    """Test Usage model."""

    def test_basic_usage(self):
        """Test basic usage with input and output tokens."""
        usage_data: dict[str, Any] = {
            "input_tokens": 10,
            "output_tokens": 15,
        }

        usage = Usage(**usage_data)

        assert usage.input_tokens == 10
        assert usage.output_tokens == 15
        assert usage.cache_creation_input_tokens is None
        assert usage.cache_read_input_tokens is None

    def test_usage_with_cache_tokens(self):
        """Test usage with cache-related tokens."""
        usage_data: dict[str, Any] = {
            "input_tokens": 10,
            "output_tokens": 15,
            "cache_creation_input_tokens": 5,
            "cache_read_input_tokens": 3,
        }

        usage = Usage(**usage_data)

        assert usage.input_tokens == 10
        assert usage.output_tokens == 15
        assert usage.cache_creation_input_tokens == 5
        assert usage.cache_read_input_tokens == 3

    def test_usage_defaults(self):
        """Test usage with default values."""
        usage = Usage(
            input_tokens=0,
            output_tokens=0,
            cache_creation_input_tokens=None,
            cache_read_input_tokens=None,
        )

        assert usage.input_tokens == 0
        assert usage.output_tokens == 0
        assert usage.cache_creation_input_tokens is None
        assert usage.cache_read_input_tokens is None


@pytest.mark.unit
class TestToolDefinition:
    """Test ToolDefinition model."""

    def test_valid_tool_definition(self):
        """Test valid tool definition."""
        tool_data: dict[str, Any] = {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "City name"}
                    },
                    "required": ["location"],
                },
            },
        }

        tool = ToolDefinition(**tool_data)

        assert tool.type == "function"
        assert tool.function.name == "get_weather"
        assert tool.function.description == "Get current weather"
        assert hasattr(tool.function, "parameters")

    def test_tool_definition_minimal(self):
        """Test minimal tool definition."""
        tool_data: dict[str, Any] = {
            "function": {
                "name": "simple_function",
                "description": "A simple function",
                "parameters": {"type": "object", "properties": {}, "required": []},
            }
        }

        tool = ToolDefinition(**tool_data)

        assert tool.type == "function"  # default value
        assert tool.function.name == "simple_function"


@pytest.mark.unit
class TestToolUse:
    """Test ToolUse model."""

    def test_valid_tool_use(self):
        """Test valid tool use."""
        tool_use_data: dict[str, Any] = {
            "type": "tool_use",
            "id": "tool_123",
            "name": "get_weather",
            "input": {"location": "New York"},
        }

        tool_use = ToolUse(**tool_use_data)

        assert tool_use.type == "tool_use"
        assert tool_use.id == "tool_123"
        assert tool_use.name == "get_weather"
        assert tool_use.input["location"] == "New York"

    def test_tool_use_default_type(self):
        """Test tool use with default type."""
        tool_use_data: dict[str, Any] = {
            "id": "tool_456",
            "name": "calculate",
            "input": {"x": 5, "y": 10},
        }

        tool_use = ToolUse(**tool_use_data)

        assert tool_use.type == "tool_use"  # default value
        assert tool_use.id == "tool_456"
        assert tool_use.name == "calculate"
        assert tool_use.input == {"x": 5, "y": 10}


@pytest.mark.unit
class TestStreamingChatCompletionResponse:
    """Test StreamingChatCompletionResponse model."""

    def test_message_start_event(self):
        """Test message start event."""
        event_data: dict[str, Any] = {
            "id": "msg_123",
            "type": "message_start",
            "message": {
                "id": "msg_123",
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": "claude-3-5-sonnet-20241022",
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 10, "output_tokens": 0},
            },
        }

        event = StreamingChatCompletionResponse(**event_data)

        assert event.id == "msg_123"
        assert event.type == "message_start"
        assert event.message is not None
        assert event.message["role"] == "assistant"
        assert event.index is None
        assert event.delta is None

    def test_content_block_delta_event(self):
        """Test content block delta event."""
        event_data: dict[str, Any] = {
            "id": "msg_123",
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "Hello"},
        }

        event = StreamingChatCompletionResponse(**event_data)

        assert event.id == "msg_123"
        assert event.type == "content_block_delta"
        assert event.index == 0
        assert event.delta is not None
        assert event.delta["type"] == "text_delta"
        assert event.delta["text"] == "Hello"
        assert event.message is None

    def test_message_delta_event(self):
        """Test message delta event."""
        event_data: dict[str, Any] = {
            "id": "msg_123",
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"output_tokens": 15},
        }

        event = StreamingChatCompletionResponse(**event_data)

        assert event.id == "msg_123"
        assert event.type == "message_delta"
        assert event.delta is not None
        assert event.delta["stop_reason"] == "end_turn"
        assert event.usage is not None
        assert event.usage.output_tokens == 15

    def test_ping_event(self):
        """Test ping event."""
        event_data: dict[str, Any] = {
            "id": "msg_123",
            "type": "ping",
        }

        event = StreamingChatCompletionResponse(**event_data)

        assert event.id == "msg_123"
        assert event.type == "ping"
        assert event.message is None
        assert event.delta is None
        assert event.index is None
        assert event.usage is None
