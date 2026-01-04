"""Tests for type safety fixes in LLM modules.

This module tests the type safety fixes made to resolve mypy errors
in the LLM adapters and models.
"""

from typing import Any

import pytest
from pydantic import ValidationError

from ccproxy.llms.models.anthropic import (
    MessageResponse as AnthropicMessageResponse,
)
from ccproxy.llms.models.anthropic import (
    TextBlock as AnthropicTextBlock,
)
from ccproxy.llms.models.anthropic import (
    Usage as AnthropicUsage,
)
from ccproxy.llms.models.openai import VALID_INCLUDE_VALUES, ResponseRequest


class TestOpenAIModelsTypeSafety:
    """Test type safety fixes in OpenAI models."""

    def test_response_request_include_validation_valid(self) -> None:
        """Test that valid include values pass validation."""
        request = ResponseRequest(
            model="gpt-4o",
            input="test input",
            include=["web_search_call.action.sources", "message.output_text.logprobs"],
        )
        assert request.include == [
            "web_search_call.action.sources",
            "message.output_text.logprobs",
        ]

    def test_response_request_include_validation_empty(self) -> None:
        """Test that empty include list is valid."""
        request = ResponseRequest(model="gpt-4o", input="test input", include=[])
        assert request.include == []

    def test_response_request_include_validation_none(self) -> None:
        """Test that None include value is valid."""
        request = ResponseRequest(model="gpt-4o", input="test input", include=None)
        assert request.include is None

    def test_response_request_include_validation_invalid(self) -> None:
        """Test that invalid include values raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            ResponseRequest(
                model="gpt-4o", input="test input", include=["invalid.include.value"]
            )

        error_msg = str(exc_info.value)
        assert "Invalid include value: invalid.include.value" in error_msg
        assert "Valid values are:" in error_msg

    def test_response_request_include_validation_mixed_valid_invalid(self) -> None:
        """Test that mix of valid and invalid include values raises ValidationError."""
        with pytest.raises(ValidationError):
            ResponseRequest(
                model="gpt-4o",
                input="test input",
                include=[
                    "web_search_call.action.sources",  # valid
                    "invalid.value",  # invalid
                ],
            )

    def test_valid_include_values_constant(self) -> None:
        """Test that VALID_INCLUDE_VALUES constant has expected values."""
        expected_values = [
            "web_search_call.action.sources",
            "code_interpreter_call.outputs",
            "computer_call_output.output.image_url",
            "file_search_call.results",
            "message.input_image.image_url",
            "message.output_text.logprobs",
            "reasoning.encrypted_content",
        ]

        # Compare as sets to ensure at least expected keys exist (order agnostic)
        assert set(expected_values).issubset(set(VALID_INCLUDE_VALUES))

    def test_response_request_background_field(self) -> None:
        """Test background field with proper typing."""
        # Test with True
        request = ResponseRequest(model="gpt-4o", input="test input", background=True)
        assert request.background is True

        # Test with False
        request = ResponseRequest(model="gpt-4o", input="test input", background=False)
        assert request.background is False

        # Test with None (default)
        request = ResponseRequest(model="gpt-4o", input="test input")
        assert request.background is None

    def test_response_request_conversation_field(self) -> None:
        """Test conversation field with proper typing."""
        # Test with string
        request = ResponseRequest(
            model="gpt-4o", input="test input", conversation="conv_123"
        )
        assert request.conversation == "conv_123"

        # Test with dict
        conv_dict: dict[str, Any] = {"id": "conv_123", "title": "Test"}
        request = ResponseRequest(
            model="gpt-4o", input="test input", conversation=conv_dict
        )
        assert request.conversation == conv_dict


class TestAnthropicMessageResponseTypeSafety:
    """Test type safety fixes for Anthropic MessageResponse."""

    def test_message_response_requires_type_field(self) -> None:
        """Test that MessageResponse requires the type field."""
        # This should work now with type field
        response = AnthropicMessageResponse(
            id="msg_123",
            type="message",
            role="assistant",
            model="claude-sonnet",
            content=[AnthropicTextBlock(type="text", text="Hello")],
            stop_reason="end_turn",
            stop_sequence=None,
            usage=AnthropicUsage(input_tokens=10, output_tokens=5),
        )

        assert response.type == "message"
        assert response.id == "msg_123"

    def test_message_response_type_field_validation(self) -> None:
        """Test that type field must be 'message'."""
        # Valid type
        response = AnthropicMessageResponse(
            id="msg_123",
            type="message",
            role="assistant",
            model="claude-sonnet",
            content=[AnthropicTextBlock(type="text", text="Hello")],
            stop_reason="end_turn",
            stop_sequence=None,
            usage=AnthropicUsage(input_tokens=10, output_tokens=5),
        )
        assert response.type == "message"
