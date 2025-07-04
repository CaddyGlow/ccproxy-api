"""Tests for OpenAI model validation edge cases."""

import pytest
from pydantic import ValidationError

from claude_code_proxy.models.openai_models import (
    OpenAIChatCompletionRequest,
    OpenAIMessage,
    OpenAITool,
)


@pytest.mark.unit
class TestOpenAIRequestValidation:
    """Test OpenAI request validation edge cases."""

    def test_empty_messages_validation_error(self) -> None:
        """Test that empty messages list raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            OpenAIChatCompletionRequest.model_validate(
                {
                    "model": "claude-opus-4-20250514",
                    "messages": [],  # Empty messages should fail - covers line 193
                }
            )

        assert "List should have at least 1 item" in str(exc_info.value)

    def test_custom_messages_validator_empty_list(self) -> None:
        """Test the custom messages validator directly with empty list."""
        from claude_code_proxy.models.openai_models import OpenAIChatCompletionRequest

        # Call the validator method directly to test line 193
        with pytest.raises(ValueError) as exc_info:
            OpenAIChatCompletionRequest.validate_messages([])

        assert "At least one message is required" in str(exc_info.value)

    def test_stop_string_validation(self) -> None:
        """Test stop parameter validation with string."""
        # Test valid string stop parameter - covers line 201-202
        request = OpenAIChatCompletionRequest.model_validate(
            {
                "model": "claude-opus-4-20250514",
                "messages": [{"role": "user", "content": "Hello"}],
                "stop": "STOP",  # Single string
            }
        )
        assert request.stop == "STOP"

    def test_stop_list_validation(self) -> None:
        """Test stop parameter validation with list."""
        # Test valid list stop parameter - covers line 203-206
        request = OpenAIChatCompletionRequest.model_validate(
            {
                "model": "claude-opus-4-20250514",
                "messages": [{"role": "user", "content": "Hello"}],
                "stop": ["STOP", "END", "QUIT"],  # List of strings
            }
        )
        assert request.stop == ["STOP", "END", "QUIT"]

    def test_stop_list_too_many_validation_error(self) -> None:
        """Test that more than 4 stop sequences raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            OpenAIChatCompletionRequest.model_validate(
                {
                    "model": "claude-opus-4-20250514",
                    "messages": [{"role": "user", "content": "Hello"}],
                    "stop": [
                        "STOP",
                        "END",
                        "QUIT",
                        "EXIT",
                        "FINISH",
                    ],  # 5 items - covers line 204-205
                }
            )

        assert "Maximum 4 stop sequences allowed" in str(exc_info.value)

    def test_tools_too_many_validation_error(self) -> None:
        """Test that more than 128 tools raises validation error."""
        # Create 129 tools to exceed the limit
        tools = []
        for i in range(129):
            from claude_code_proxy.models.openai_models import OpenAIFunction

            tools.append(
                OpenAITool(
                    type="function",
                    function=OpenAIFunction(
                        name=f"tool_{i}",
                        description=f"Tool {i}",
                        parameters={"type": "object", "properties": {}},
                    ),
                )
            )

        with pytest.raises(ValidationError) as exc_info:
            OpenAIChatCompletionRequest.model_validate(
                {
                    "model": "claude-opus-4-20250514",
                    "messages": [{"role": "user", "content": "Hello"}],
                    "tools": [
                        tool.model_dump() for tool in tools
                    ],  # 129 tools - covers line 214
                }
            )

        assert "Maximum 128 tools allowed" in str(exc_info.value)

    def test_tools_exactly_128_allowed(self) -> None:
        """Test that exactly 128 tools is allowed."""
        # Create exactly 128 tools (at the limit)
        tools = []
        for i in range(128):
            from claude_code_proxy.models.openai_models import OpenAIFunction

            tools.append(
                OpenAITool(
                    type="function",
                    function=OpenAIFunction(
                        name=f"tool_{i}",
                        description=f"Tool {i}",
                        parameters={"type": "object", "properties": {}},
                    ),
                )
            )

        # This should not raise an error
        request = OpenAIChatCompletionRequest.model_validate(
            {
                "model": "claude-opus-4-20250514",
                "messages": [{"role": "user", "content": "Hello"}],
                "tools": [tool.model_dump() for tool in tools],  # Exactly 128 tools
            }
        )
        assert request.tools is not None and len(request.tools) == 128

    def test_stop_none_validation(self) -> None:
        """Test that None stop parameter is allowed."""
        request = OpenAIChatCompletionRequest.model_validate(
            {
                "model": "claude-opus-4-20250514",
                "messages": [{"role": "user", "content": "Hello"}],
                "stop": None,  # None should be allowed - covers line 207
            }
        )
        assert request.stop is None

    def test_tools_none_validation(self) -> None:
        """Test that None tools parameter is allowed."""
        request = OpenAIChatCompletionRequest.model_validate(
            {
                "model": "claude-opus-4-20250514",
                "messages": [{"role": "user", "content": "Hello"}],
                "tools": None,  # None should be allowed
            }
        )
        assert request.tools is None


@pytest.mark.unit
class TestOpenAIResponseGeneration:
    """Test OpenAI response generation edge cases."""

    def test_create_response_factory_method(self) -> None:
        """Test the create class method for generating responses."""
        from claude_code_proxy.models.openai_models import OpenAIChatCompletionResponse

        # Test the factory method - covers line 337
        response = OpenAIChatCompletionResponse.create(
            model="claude-opus-4-20250514",
            content="Hello, world!",
            prompt_tokens=10,
            completion_tokens=5,
            finish_reason="stop",
        )

        assert response.model == "claude-opus-4-20250514"
        assert len(response.choices) == 1
        assert response.choices[0].message.content == "Hello, world!"
        assert response.choices[0].finish_reason == "stop"
        assert response.usage.prompt_tokens == 10
        assert response.usage.completion_tokens == 5
        assert response.usage.total_tokens == 15
        assert response.object == "chat.completion"
        assert response.id.startswith("chatcmpl-")

    def test_create_response_with_tool_calls(self) -> None:
        """Test creating response with tool calls."""
        from claude_code_proxy.models.openai_models import (
            OpenAIChatCompletionResponse,
            OpenAIFunctionCall,
            OpenAIToolCall,
        )

        tool_calls = [
            OpenAIToolCall(
                id="call_123",
                type="function",
                function=OpenAIFunctionCall(
                    name="get_weather",
                    arguments='{"location": "New York"}',
                ),
            )
        ]

        response = OpenAIChatCompletionResponse.create(
            model="claude-opus-4-20250514",
            content="I'll check the weather for you.",
            prompt_tokens=15,
            completion_tokens=8,
            finish_reason="tool_calls",
            tool_calls=tool_calls,
        )

        assert response.choices[0].finish_reason == "tool_calls"
        assert response.choices[0].message.tool_calls is not None
        assert len(response.choices[0].message.tool_calls) == 1
        assert response.choices[0].message.tool_calls[0].function.name == "get_weather"
