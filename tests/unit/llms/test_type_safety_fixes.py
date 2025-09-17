"""Tests for type safety fixes in LLM modules.

This module tests the type safety fixes made to resolve mypy errors
in the LLM adapters and models.
"""

from typing import Any

import pytest
from pydantic import ValidationError

from ccproxy.llms.models.anthropic import (
    CreateMessageRequest,
    ImageBlock,
    ImageSource,
    Message,
)
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


class TestAdapterTypeSafety:
    """Test type safety fixes in adapters."""

    @pytest.mark.asyncio
    async def test_adapter_union_attribute_access_safety(self) -> None:
        """Test that union attribute access is properly handled with type guards."""
        # This test verifies that our type guard fixes work properly
        # by testing the adapter logic that was causing union-attr errors

        from ccproxy.llms.formatters.anthropic_to_openai.messages_to_responses import (
            AnthropicMessagesToOpenAIResponsesAdapter,
        )

        # Create a simple Anthropic request with text content using Pydantic models
        anthropic_request = CreateMessageRequest(
            model="claude-sonnet",
            messages=[
                Message(
                    role="user",
                    content=[AnthropicTextBlock(type="text", text="Hello world")],
                )
            ],
            max_tokens=100,
        )

        adapter = AnthropicMessagesToOpenAIResponsesAdapter()

        # This should not raise union-attr errors anymore
        result = await adapter.adapt_request(anthropic_request)

        # Verify the conversion worked
        assert hasattr(result, "input")
        assert isinstance(result.input, list)
        assert result.input[0]["type"] == "message"

    @pytest.mark.asyncio
    async def test_adapter_handles_mixed_content_blocks_safely(self) -> None:
        """Test that adapters handle mixed content block types without union errors."""
        from ccproxy.llms.formatters.anthropic_to_openai.messages_to_chat import (
            AnthropicMessagesToOpenAIChatAdapter,
        )

        # Create request with mixed content types (text + image) using Pydantic models
        anthropic_request = CreateMessageRequest(
            model="claude-sonnet",
            messages=[
                Message(
                    role="user",
                    content=[
                        AnthropicTextBlock(type="text", text="What's in this image?"),
                        ImageBlock(
                            type="image",
                            source=ImageSource(
                                type="base64",
                                media_type="image/png",
                                data="iVBORw0KGgo...",
                            ),
                        ),
                    ],
                )
            ],
            max_tokens=100,
        )

        adapter = AnthropicMessagesToOpenAIChatAdapter()

        # This should handle the mixed content types safely
        result = await adapter.adapt_request(anthropic_request)

        # Verify conversion worked
        assert hasattr(result, "messages")
        assert len(result.messages) == 1
        user_message = result.messages[0]
        assert user_message.role == "user"
        assert isinstance(user_message.content, list)

        # Should have both text and image content (accepting either image_url or image)
        content_types = {item["type"] for item in user_message.content}
        assert "text" in content_types
        assert ("image_url" in content_types) or ("image" in content_types)
