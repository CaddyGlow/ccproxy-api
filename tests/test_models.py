"""Tests for Pydantic models."""

import pytest
from pydantic import ValidationError

from claude_proxy.models.errors import (
    AnthropicError,
    InvalidRequestError,
    create_error_response,
)
from claude_proxy.models.requests import (
    ChatCompletionRequest,
    ImageContent,
    Message,
    TextContent,
)
from claude_proxy.models.responses import ChatCompletionResponse


class TestChatCompletionRequest:
    """Test ChatCompletionRequest model."""

    def test_valid_request(self):
        """Test valid chat completion request."""
        request_data = {
            "model": "claude-3-5-sonnet-20241022",
            "messages": [
                {"role": "user", "content": "Hello"}
            ],
            "max_tokens": 100
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
        request_data = {
            "model": "invalid-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 100
        }

        with pytest.raises(ValidationError) as exc_info:
            ChatCompletionRequest(**request_data)

        assert "claude-" in str(exc_info.value)

    def test_temperature_validation(self):
        """Test temperature validation."""
        # Valid temperature
        request_data = {
            "model": "claude-3-5-sonnet-20241022",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 100,
            "temperature": 0.5
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
        request_data = {
            "model": "claude-3-5-sonnet-20241022",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 0  # Invalid
        }

        with pytest.raises(ValidationError):
            ChatCompletionRequest(**request_data)

    def test_stop_sequences_validation(self):
        """Test stop_sequences validation."""
        request_data = {
            "model": "claude-3-5-sonnet-20241022",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 100,
            "stop_sequences": ["\\n", "END", "STOP", "DONE", "FINISH"]  # Too many
        }

        with pytest.raises(ValidationError) as exc_info:
            ChatCompletionRequest(**request_data)

        assert "at most 4" in str(exc_info.value)


class TestMessage:
    """Test Message model."""

    def test_user_message_with_text(self):
        """Test user message with text content."""
        message_data = {
            "role": "user",
            "content": "Hello, how are you?"
        }

        message = Message(**message_data)

        assert message.role == "user"
        assert message.content == "Hello, how are you?"

    def test_user_message_with_content_blocks(self):
        """Test user message with content blocks."""
        message_data = {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
                    }
                }
            ]
        }

        message = Message(**message_data)

        assert message.role == "user"
        assert isinstance(message.content, list)
        assert len(message.content) == 2
        assert isinstance(message.content[0], TextContent)
        assert isinstance(message.content[1], ImageContent)

    def test_assistant_message(self):
        """Test assistant message."""
        message_data = {
            "role": "assistant",
            "content": "I'm doing well, thank you!"
        }

        message = Message(**message_data)

        assert message.role == "assistant"
        assert message.content == "I'm doing well, thank you!"

    def test_invalid_role(self):
        """Test invalid role validation."""
        message_data = {
            "role": "invalid",
            "content": "Hello"
        }

        with pytest.raises(ValidationError):
            Message(**message_data)


class TestErrorModels:
    """Test error models."""

    def test_anthropic_error(self):
        """Test AnthropicError model."""
        error = AnthropicError(
            error={
                "type": "invalid_request_error",
                "message": "Invalid request"
            }
        )

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
            "invalid_request_error",
            "Test error message",
            400
        )

        assert status_code == 400
        assert error_dict["type"] == "error"
        assert error_dict["error"]["type"] == "invalid_request_error"
        assert error_dict["error"]["message"] == "Test error message"


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
