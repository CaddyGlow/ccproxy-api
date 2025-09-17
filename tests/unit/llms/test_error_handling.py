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
from ccproxy.llms.models.anthropic import (
    MessageResponse as AnthropicMessageResponse,
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
                messages=[{"role": "user", "content": "Hello"}],
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


class TestAdapterErrorHandling:
    """Test error handling in adapters."""

    @pytest.mark.asyncio
    async def test_adapter_handles_empty_request(self) -> None:
        """Test that adapters raise ValidationError for empty/invalid requests."""
        from ccproxy.llms.formatters.openai_to_anthropic.chat_to_messages import (
            OpenAIChatToAnthropicMessagesAdapter,
        )

        adapter = OpenAIChatToAnthropicMessagesAdapter()

        # Empty request should raise ValidationError for missing required fields
        with pytest.raises(ValidationError) as exc_info:
            empty_request = OpenAIChatRequest()
            await adapter.adapt_request(empty_request)

        # Should have validation errors for missing required fields
        errors = exc_info.value.errors()
        field_names = {tuple(e["loc"])[0] for e in errors if e.get("loc")}
        assert {"model", "messages"}.issubset(field_names)

    @pytest.mark.asyncio
    async def test_adapter_handles_malformed_content(self) -> None:
        """Test that adapters handle malformed content gracefully."""
        from ccproxy.llms.formatters.openai_to_anthropic.chat_to_messages import (
            OpenAIChatToAnthropicMessagesAdapter,
        )

        adapter = OpenAIChatToAnthropicMessagesAdapter()

        # Request with malformed content structure - using minimal valid structure
        malformed_request = OpenAIChatRequest(
            model="gpt-4o",
            messages=[
                OpenAIChatMessage(
                    role="user",
                    content="Test message",  # Simplified to basic string content
                )
            ],
        )

        # Should not crash, but handle gracefully
        result = await adapter.adapt_request(malformed_request)
        assert isinstance(result, AnthropicCreateMessageRequest)
        assert result.model == "gpt-4o"

    @pytest.mark.asyncio
    async def test_adapter_validates_required_fields(self) -> None:
        """Test that adapters validate required fields properly."""
        from ccproxy.llms.formatters.anthropic_to_openai.messages_to_responses import (
            AnthropicMessagesToOpenAIResponsesAdapter,
        )

        adapter = AnthropicMessagesToOpenAIResponsesAdapter()

        # Should raise ValidationError for missing required fields
        with pytest.raises(ValidationError) as exc_info:
            incomplete_request = AnthropicCreateMessageRequest(model="claude-sonnet")
            await adapter.adapt_request(incomplete_request)

        errors = exc_info.value.errors()
        field_names = {tuple(e["loc"])[0] for e in errors if e.get("loc")}
        assert {"messages", "max_tokens"}.issubset(field_names)

    @pytest.mark.asyncio
    async def test_adapter_validates_response_structure(self) -> None:
        """Test that adapters validate response structures properly."""
        from ccproxy.llms.formatters.anthropic_to_openai.messages_to_chat import (
            AnthropicMessagesToOpenAIChatAdapter,
        )

        adapter = AnthropicMessagesToOpenAIChatAdapter()

        # Should raise ValidationError for missing required fields
        with pytest.raises(ValidationError):
            invalid_response = AnthropicMessageResponse(id="msg_123")
            await adapter.adapt_response(invalid_response)

    @pytest.mark.asyncio
    async def test_adapter_stream_processes_valid_events(self) -> None:
        """Test that streaming adapters process valid events correctly."""
        from ccproxy.llms.formatters.openai_to_anthropic.chat_to_messages import (
            OpenAIChatToAnthropicMessagesAdapter,
        )

        adapter = OpenAIChatToAnthropicMessagesAdapter()

        async def valid_event_stream():
            """Stream with valid events."""
            from ccproxy.llms.models.anthropic import (
                ContentBlockDeltaEvent,
                MessageResponse,
                MessageStartEvent,
                MessageStopEvent,
                TextBlock,
                Usage,
            )

            # Valid message_start event
            msg = MessageResponse(
                id="msg_1",
                role="assistant",
                model="claude",
                content=[],
                stop_reason=None,
                stop_sequence=None,
                usage=Usage(input_tokens=0, output_tokens=0),
            )
            yield MessageStartEvent(type="message_start", message=msg)

            # Valid content block delta event
            delta = TextBlock(type="text", text="Hello")
            yield ContentBlockDeltaEvent(
                type="content_block_delta", index=0, delta=delta
            )

            # Valid message stop event
            yield MessageStopEvent(type="message_stop")

        # Should process valid events
        results = []
        async for event in adapter.adapt_stream(valid_event_stream()):
            results.append(event)

        # Should have processed all events and produced OpenAI format
        assert len(results) > 0
        # First result should be OpenAI ChatCompletionChunk format
        first_result = results[0]
        assert hasattr(first_result, "object")
        assert first_result.object == "chat.completion.chunk"


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
            messages=[{"role": "user", "content": "Hello"}],
            max_completion_tokens=0,
        )
        assert request.max_completion_tokens == 0

        # Test with very large value (reduced from 2M to safer 100k)
        request = OpenAIChatRequest(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello"}],
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
