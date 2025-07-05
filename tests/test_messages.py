"""Tests for message models."""

from typing import Any

import pytest
from pydantic import ValidationError

from claude_code_proxy.models.messages import (
    MessageRequest,
    MessageResponse,
    SystemMessage,
)
from claude_code_proxy.models.requests import Usage


class TestMessageRequest:
    """Test MessageRequest model validation."""

    def test_valid_message_request(self):
        """Test valid MessageRequest creation."""
        request_data = {
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 1000,
            "messages": [{"role": "user", "content": "Hello"}],
        }
        request = MessageRequest(**request_data)
        assert request.model == "claude-3-5-sonnet-20241022"
        assert request.max_tokens == 1000
        assert len(request.messages) == 1

    def test_max_tokens_validation(self):
        """Test max_tokens validation allows large values."""
        request_data: dict[str, Any] = {
            "model": "claude-3-5-sonnet-20241022",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 0,  # Invalid
        }

        with pytest.raises(ValidationError):
            MessageRequest(**request_data)

        # Test that large values like 64000 are accepted
        request_data["max_tokens"] = 64000
        request = MessageRequest(**request_data)
        assert request.max_tokens == 64000

        # Test maximum allowed value
        request_data["max_tokens"] = 200000
        request = MessageRequest(**request_data)
        assert request.max_tokens == 200000

        # Test exceeding maximum should fail
        request_data["max_tokens"] = 200001
        with pytest.raises(ValidationError):
            MessageRequest(**request_data)

    def test_model_validation(self):
        """Test model validation."""
        request_data: dict[str, Any] = {
            "model": "gpt-4",  # Invalid - not a Claude model
            "max_tokens": 1000,
            "messages": [{"role": "user", "content": "Hello"}],
        }

        with pytest.raises(ValidationError):
            MessageRequest(**request_data)

    def test_system_message_string(self):
        """Test system message as string."""
        request_data = {
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 1000,
            "messages": [{"role": "user", "content": "Hello"}],
            "system": "You are a helpful assistant.",
        }
        request = MessageRequest(**request_data)
        assert request.system == "You are a helpful assistant."

    def test_system_message_blocks(self):
        """Test system message as blocks."""
        system_blocks = [SystemMessage(text="You are a helpful assistant.")]
        request_data = {
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 1000,
            "messages": [{"role": "user", "content": "Hello"}],
            "system": system_blocks,
        }
        request = MessageRequest(**request_data)
        assert request.system == system_blocks

    def test_message_alternation_validation(self):
        """Test message alternation validation."""
        # First message must be from user
        request_data: dict[str, Any] = {
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 1000,
            "messages": [{"role": "assistant", "content": "Hello"}],
        }

        with pytest.raises(ValidationError):
            MessageRequest(**request_data)

        # Messages must alternate
        request_data["messages"] = [
            {"role": "user", "content": "Hello"},
            {"role": "user", "content": "Hello again"},
        ]

        with pytest.raises(ValidationError):
            MessageRequest(**request_data)


class TestMessageResponse:
    """Test MessageResponse model."""

    def test_valid_message_response(self):
        """Test valid MessageResponse creation."""
        response_data = {
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "Hello there!"}],
            "model": "claude-3-5-sonnet-20241022",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
        response = MessageResponse(**response_data)
        assert response.id == "msg_123"
        assert response.role == "assistant"
        assert len(response.content) == 1


class TestSystemMessage:
    """Test SystemMessage model."""

    def test_valid_system_message(self):
        """Test valid SystemMessage creation."""
        system_msg = SystemMessage(text="You are a helpful assistant.")
        assert system_msg.type == "text"
        assert system_msg.text == "You are a helpful assistant."
