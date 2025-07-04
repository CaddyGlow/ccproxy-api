"""Tests for Pydantic models."""

from typing import Any

import pytest
from pydantic import ValidationError

from claude_code_proxy.models.errors import (
    AnthropicError,
    ErrorDetail,
    InvalidRequestError,
    create_error_response,
)
from claude_code_proxy.models.requests import (
    ChatCompletionRequest,
    ImageContent,
    Message,
    TextContent,
    ToolDefinition,
    Usage,
)
from claude_code_proxy.models.responses import (
    ChatCompletionResponse,
    StreamingChatCompletionResponse,
    ToolUse,
)


@pytest.mark.unit
class TestChatCompletionRequest:
    """Test ChatCompletionRequest model."""

    def test_valid_request(self):
        """Test valid chat completion request."""
        request_data: dict[str, Any] = {
            "model": "claude-3-5-sonnet-20241022",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 100,
        }

        request = ChatCompletionRequest(**request_data)

        assert request.model == "claude-3-5-sonnet-20241022"
        assert len(request.messages) == 1
        assert request.messages[0].role == "user"
        assert request.messages[0].content == "Hello"
        assert request.max_tokens == 100
        assert request.temperature is None  # default
        assert request.stream is False  # default

    def test_invalid_model(self):
        """Test invalid model validation."""
        request_data: dict[str, Any] = {
            "model": "invalid-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 100,
        }

        with pytest.raises(ValidationError) as exc_info:
            ChatCompletionRequest(**request_data)

        assert "claude-" in str(exc_info.value)

    def test_temperature_validation(self):
        """Test temperature validation."""
        # Valid temperature
        request_data: dict[str, Any] = {
            "model": "claude-3-5-sonnet-20241022",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 100,
            "temperature": 0.5,
        }
        request = ChatCompletionRequest(**request_data)
        assert request.temperature == 0.5

        # Invalid temperature (too low)
        request_data["temperature"] = -0.1
        with pytest.raises(ValidationError):
            ChatCompletionRequest(**request_data)

        # Invalid temperature (too high)
        request_data["temperature"] = 2.1
        with pytest.raises(ValidationError):
            ChatCompletionRequest(**request_data)

    def test_max_tokens_validation(self):
        """Test max_tokens validation."""
        request_data: dict[str, Any] = {
            "model": "claude-3-5-sonnet-20241022",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 0,  # Invalid
        }

        with pytest.raises(ValidationError):
            ChatCompletionRequest(**request_data)

    def test_stop_sequences_validation(self):
        """Test stop_sequences validation."""
        request_data: dict[str, Any] = {
            "model": "claude-3-5-sonnet-20241022",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 100,
            "stop_sequences": ["\\n", "END", "STOP", "DONE", "FINISH"],  # Too many
        }

        with pytest.raises(ValidationError) as exc_info:
            ChatCompletionRequest(**request_data)

        assert "at most 4" in str(exc_info.value)

    def test_tools_validation(self):
        """Test tools validation."""
        request_data: dict[str, Any] = {
            "model": "claude-3-5-sonnet-20241022",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 100,
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get current weather",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "location": {
                                    "type": "string",
                                    "description": "City name",
                                }
                            },
                            "required": ["location"],
                        },
                    },
                }
            ],
        }

        request = ChatCompletionRequest(**request_data)

        assert request.tools is not None
        assert len(request.tools) == 1
        assert request.tools[0].type == "function"
        assert request.tools[0].function.name == "get_weather"

    def test_tool_choice_validation(self):
        """Test tool_choice validation."""
        request_data: dict[str, Any] = {
            "model": "claude-3-5-sonnet-20241022",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 100,
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get current weather",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "location": {
                                    "type": "string",
                                    "description": "The city and state, e.g. San Francisco, CA",
                                }
                            },
                            "required": ["location"],
                        },
                    },
                }
            ],
            "tool_choice": {"type": "function", "function": {"name": "get_weather"}},
        }

        request = ChatCompletionRequest(**request_data)

        assert request.tool_choice is not None
        assert request.tool_choice["type"] == "function"
        assert request.tool_choice["function"]["name"] == "get_weather"

    def test_max_thinking_tokens_validation(self):
        """Test max_thinking_tokens validation."""
        request_data: dict[str, Any] = {
            "model": "claude-3-5-sonnet-20241022",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 100,
            "max_thinking_tokens": 1000,
        }

        request = ChatCompletionRequest(**request_data)

        assert request.max_thinking_tokens == 1000

    def test_system_prompt_validation(self):
        """Test system prompt validation."""
        request_data: dict[str, Any] = {
            "model": "claude-3-5-sonnet-20241022",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 100,
            "system": "You are a helpful assistant.",
        }

        request = ChatCompletionRequest(**request_data)

        assert request.system == "You are a helpful assistant."


@pytest.mark.unit
class TestChatCompletionRequestValidationEdgeCases:
    """Test ChatCompletionRequest validation edge cases for coverage."""

    def test_message_alternation_validation_first_not_user(self) -> None:
        """Test that first message must be from user."""
        request_data: dict[str, Any] = {
            "model": "claude-3-5-sonnet-20241022",
            "messages": [
                {
                    "role": "assistant",
                    "content": "Hello!",
                }  # First message not from user
            ],
            "max_tokens": 100,
        }

        with pytest.raises(ValidationError) as exc_info:
            ChatCompletionRequest(**request_data)

        assert "First message must be from user" in str(exc_info.value)

    def test_message_alternation_validation_consecutive_same_role(self) -> None:
        """Test that messages must alternate between user and assistant."""
        request_data: dict[str, Any] = {
            "model": "claude-3-5-sonnet-20241022",
            "messages": [
                {"role": "user", "content": "Hello"},
                {
                    "role": "user",
                    "content": "Hello again",
                },  # Two consecutive user messages
            ],
            "max_tokens": 100,
        }

        with pytest.raises(ValidationError) as exc_info:
            ChatCompletionRequest(**request_data)

        assert "Messages must alternate between user and assistant" in str(
            exc_info.value
        )

    def test_unsupported_model_validation(self) -> None:
        """Test that unsupported model names raise validation error."""
        request_data: dict[str, Any] = {
            "model": "gpt-4",  # Not a Claude model
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 100,
        }

        with pytest.raises(ValidationError) as exc_info:
            ChatCompletionRequest(**request_data)

        assert "String should match pattern" in str(exc_info.value)

    def test_stop_sequences_max_length_validation(self) -> None:
        """Test that stop sequences over 4 items raise validation error."""
        request_data: dict[str, Any] = {
            "model": "claude-3-5-sonnet-20241022",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 100,
            "stop_sequences": ["STOP", "END", "QUIT", "DONE", "FINISH"],  # 5 items
        }

        with pytest.raises(ValidationError) as exc_info:
            ChatCompletionRequest(**request_data)

        assert "List should have at most 4 items" in str(exc_info.value)

    def test_stop_sequences_length_validation(self) -> None:
        """Test that stop sequences over 100 characters raise validation error."""
        long_sequence = "x" * 101  # 101 characters
        request_data: dict[str, Any] = {
            "model": "claude-3-5-sonnet-20241022",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 100,
            "stop_sequences": [long_sequence],
        }

        with pytest.raises(ValidationError) as exc_info:
            ChatCompletionRequest(**request_data)

        assert "Stop sequences must be 100 characters or less" in str(exc_info.value)

    def test_stop_sequences_none_validation(self) -> None:
        """Test that None stop_sequences is allowed."""
        request_data: dict[str, Any] = {
            "model": "claude-3-5-sonnet-20241022",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 100,
            "stop_sequences": None,
        }

        request = ChatCompletionRequest(**request_data)
        assert request.stop_sequences is None

    def test_claude_model_pattern_forward_compatibility(self) -> None:
        """Test that any model starting with 'claude-' is accepted."""
        request_data: dict[str, Any] = {
            "model": "claude-4-future-model",  # Future model that starts with claude-
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 100,
        }

        request = ChatCompletionRequest(**request_data)
        assert request.model == "claude-4-future-model"


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
