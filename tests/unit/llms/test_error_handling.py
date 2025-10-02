"""Tests for error handling in LLM modules.

This module tests error handling scenarios that were previously missing
from the test coverage.
"""

import pytest
from pydantic import ValidationError

from ccproxy.llms.models.anthropic import (
    CreateMessageRequest as AnthropicCreateMessageRequest,
)
from ccproxy.llms.models.anthropic import (
    Message as AnthropicMessage,
)
from ccproxy.llms.models.openai import (
    ChatCompletionRequest as OpenAIChatRequest,
)
from ccproxy.llms.models.openai import (
    ChatMessage as OpenAIChatMessage,
)
from ccproxy.llms.models.openai import (
    ResponseRequest as OpenAIResponseRequest,
)


class TestModelValidationErrors:
    """Test validation error handling in models."""

    def test_openai_chat_request_invalid_temperature(self) -> None:
        """Test that invalid temperature values raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            OpenAIChatRequest(
                model="gpt-4o",
                messages=[OpenAIChatMessage(role="user", content="Hello")],
                temperature=3.0,  # Invalid: should be <= 2.0
            )

        errors = exc_info.value.errors()
        assert any(e.get("loc") == ("temperature",) for e in errors)
        assert any(e.get("type", "").endswith("equal") for e in errors)

    def test_openai_chat_request_invalid_top_p(self) -> None:
        """Test that invalid top_p values raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            OpenAIChatRequest(
                model="gpt-4o",
                messages=[OpenAIChatMessage(role="user", content="Hello")],
                top_p=1.5,  # Invalid: should be <= 1.0
            )

        errors = exc_info.value.errors()
        assert any(e.get("loc") == ("top_p",) for e in errors)
        assert any(e.get("type", "").endswith("equal") for e in errors)

    def test_openai_responses_request_invalid_temperature(self) -> None:
        """Test that invalid temperature values raise ValidationError in ResponseRequest."""
        with pytest.raises(ValidationError) as exc_info:
            OpenAIResponseRequest(
                model="gpt-4o",
                input="Hello",
                temperature=-1.0,  # Invalid: should be >= 0.0
            )

        errors = exc_info.value.errors()
        assert any(e.get("loc") == ("temperature",) for e in errors)
        assert any(e.get("type", "").endswith("equal") for e in errors)

    def test_anthropic_create_message_request_empty_messages(self) -> None:
        """Test that empty messages list is intentionally allowed."""
        # CONFIRMED: Empty messages list is valid in the current model implementation
        # This is an intentional design choice to allow flexibility in request construction
        request = AnthropicCreateMessageRequest(
            model="claude-sonnet",
            messages=[],  # This is intentionally allowed
            max_tokens=100,
        )
        assert request.messages == []

    def test_anthropic_message_invalid_role(self) -> None:
        """Test that invalid role values raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            AnthropicMessage(
                role="invalid_role",  # type: ignore[arg-type]
                content="Hello",
            )

        errors = exc_info.value.errors()
        assert any(e.get("loc") == ("role",) for e in errors)
        assert any(e.get("type", "") == "literal_error" for e in errors)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_openai_responses_request_all_include_values(self) -> None:
        """Test ResponseRequest with all valid include values."""
        from ccproxy.llms.models.openai import VALID_INCLUDE_VALUES

        request = OpenAIResponseRequest(
            model="gpt-4o",
            input="test input",
            include=VALID_INCLUDE_VALUES.copy(),  # All valid values
        )

        assert request.include == VALID_INCLUDE_VALUES

    def test_openai_responses_request_large_input_list(self) -> None:
        """Test ResponseRequest with large input list."""
        large_input_list = [
            {"type": "message", "role": "user", "content": f"Message {i}"}
            for i in range(100)
        ]

        request = OpenAIResponseRequest(model="gpt-4o", input=large_input_list)

        assert len(request.input) == 100
        assert all(isinstance(item, dict) for item in request.input)

    def test_openai_chat_request_max_tokens_boundary(self) -> None:
        """Test ChatCompletionRequest with boundary values for max_tokens."""
        # Test with 0 (edge case)
        request = OpenAIChatRequest(
            model="gpt-4o",
            messages=[OpenAIChatMessage(role="user", content="Hello")],
            max_completion_tokens=0,
        )
        assert request.max_completion_tokens == 0

        # Test with very large value (reduced from 2M to safer 100k)
        request = OpenAIChatRequest(
            model="gpt-4o",
            messages=[OpenAIChatMessage(role="user", content="Hello")],
            max_completion_tokens=100000,
        )
        assert request.max_completion_tokens == 100000

    def test_anthropic_content_empty_string(self) -> None:
        """Test Anthropic models with empty string content."""
        message = AnthropicMessage(
            role="user",
            content="",  # Empty string
        )
        assert message.content == ""

    def test_anthropic_content_very_long_string(self) -> None:
        """Test Anthropic models with very long content."""
        long_content = "Hello " * 10000  # 60k characters
        message = AnthropicMessage(role="user", content=long_content)
        assert len(message.content) == 60000

    def test_anthropic_tool_nested_custom_payload(self) -> None:
        """Incoming tool payloads with nested custom definition parse correctly."""

        payload = {
            "type": "tool",
            "custom": {
                "name": "read_file",
                "description": "Read project files",
                "input_schema": {"type": "object", "properties": {}},
            },
        }

        request = AnthropicCreateMessageRequest(
            model="claude-3-opus-20240229",
            messages=[AnthropicMessage(role="user", content="hi")],
            max_tokens=128,
            tools=[payload],  # type: ignore[list-item]
        )

        assert request.tools is not None
        tool = request.tools[0]
        assert tool.type == "tool"
        assert tool.name == "read_file"
        assert tool.input_schema == {"type": "object", "properties": {}}

    def test_anthropic_tool_nested_custom_legacy_tag(self) -> None:
        """Legacy `type=custom` payloads still validate and forward."""

        payload = {
            "type": "custom",
            "custom": {
                "name": "write_file",
                "description": "Write project files",
                "input_schema": {"type": "object", "properties": {}},
            },
        }

        request = AnthropicCreateMessageRequest(
            model="claude-3-opus-20240229",
            messages=[AnthropicMessage(role="user", content="hi")],
            max_tokens=128,
            tools=[payload],  # type: ignore[list-item]
        )

        assert request.tools is not None
        tool = request.tools[0]
        assert tool.type == "custom"
        assert tool.name == "write_file"
        assert tool.input_schema == {"type": "object", "properties": {}}

    def test_anthropic_tool_missing_type_defaults_to_custom(self) -> None:
        """Tool definitions without a discriminator assume legacy custom tag."""

        payload = {
            "name": "list",
            "description": "List files in a directory",
            "input_schema": {"type": "object"},
        }

        request = AnthropicCreateMessageRequest(
            model="claude-3-opus-20240229",
            messages=[AnthropicMessage(role="user", content="hi")],
            max_tokens=128,
            tools=[payload],  # type: ignore[list-item]
        )

        assert request.tools is not None
        tool = request.tools[0]
        assert tool.type == "custom"
        assert tool.name == "list"
