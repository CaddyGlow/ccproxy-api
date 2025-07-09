"""Tests for message models with typed fields."""

from typing import Any

import pytest
from pydantic import ValidationError

from ccproxy.models.messages import (
    MessageCreateParams,
    MetadataParams,
    SystemMessage,
    ThinkingConfig,
    ToolChoiceParams,
)
from ccproxy.models.requests import Message


@pytest.mark.unit
class TestMessageCreateParams:
    """Test MessageCreateParams model with typed fields."""

    def test_basic_message_create_params(self):
        """Test creating a basic MessageCreateParams."""
        params = MessageCreateParams(
            model="claude-3-5-sonnet-20241022",
            messages=[Message(role="user", content="Hello")],
            max_tokens=100,
        )

        assert params.model == "claude-3-5-sonnet-20241022"
        assert len(params.messages) == 1
        assert params.messages[0].role == "user"
        assert params.messages[0].content == "Hello"
        assert params.max_tokens == 100
        assert params.stream is False  # default value

    def test_message_create_params_with_metadata(self):
        """Test MessageCreateParams with MetadataParams."""
        metadata = MetadataParams(user_id="user-123")
        params = MessageCreateParams(
            model="claude-3-5-sonnet-20241022",
            messages=[Message(role="user", content="Hello")],
            max_tokens=100,
            metadata=metadata,
        )

        assert params.metadata is not None
        assert params.metadata.user_id == "user-123"

    def test_message_create_params_with_thinking(self):
        """Test MessageCreateParams with ThinkingConfig."""
        thinking = ThinkingConfig(type="enabled", budget_tokens=2048)
        params = MessageCreateParams(
            model="claude-3-5-sonnet-20241022",
            messages=[Message(role="user", content="Solve this problem")],
            max_tokens=1000,
            thinking=thinking,
        )

        assert params.thinking is not None
        assert params.thinking.type == "enabled"
        assert params.thinking.budget_tokens == 2048

    def test_message_create_params_with_tool_choice(self):
        """Test MessageCreateParams with ToolChoiceParams."""
        tool_choice = ToolChoiceParams(
            type="tool",
            name="calculator",
            disable_parallel_tool_use=True,
        )
        params = MessageCreateParams(
            model="claude-3-5-sonnet-20241022",
            messages=[Message(role="user", content="Calculate 2+2")],
            max_tokens=100,
            tool_choice=tool_choice,
        )

        assert params.tool_choice is not None
        assert params.tool_choice.type == "tool"
        assert params.tool_choice.name == "calculator"
        assert params.tool_choice.disable_parallel_tool_use is True

    def test_message_create_params_with_all_optional_fields(self):
        """Test MessageCreateParams with all optional fields."""
        params = MessageCreateParams(
            model="claude-3-5-sonnet-20241022",
            messages=[Message(role="user", content="Hello")],
            max_tokens=100,
            system="You are a helpful assistant",
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            stop_sequences=["END", "STOP"],
            stream=True,
            metadata=MetadataParams(user_id="user-456"),
            service_tier="standard_only",
            thinking=ThinkingConfig(type="enabled", budget_tokens=1024),
            tool_choice=ToolChoiceParams(type="auto"),
        )

        assert params.system == "You are a helpful assistant"
        assert params.temperature == 0.7
        assert params.top_p == 0.9
        assert params.top_k == 50
        assert params.stop_sequences == ["END", "STOP"]
        assert params.stream is True
        assert params.metadata.user_id == "user-456"  # type: ignore
        assert params.service_tier == "standard_only"
        assert params.thinking.budget_tokens == 1024  # type: ignore
        assert params.tool_choice.type == "auto"  # type: ignore

    def test_message_create_params_with_system_blocks(self):
        """Test MessageCreateParams with system message blocks."""
        system_blocks = [
            SystemMessage(type="text", text="You are a coding assistant."),
            SystemMessage(type="text", text="Always provide clear explanations."),
        ]
        params = MessageCreateParams(
            model="claude-3-5-sonnet-20241022",
            messages=[Message(role="user", content="Help me code")],
            max_tokens=200,
            system=system_blocks,
        )

        assert isinstance(params.system, list)
        assert len(params.system) == 2
        assert params.system[0].text == "You are a coding assistant."
        assert params.system[1].text == "Always provide clear explanations."

    def test_max_tokens_validation(self):
        """Test max_tokens validation."""
        # Test maximum allowed value
        params = MessageCreateParams(
            model="claude-3-5-sonnet-20241022",
            messages=[Message(role="user", content="Hello")],
            max_tokens=200000,
        )
        assert params.max_tokens == 200000

        # Test exceeding maximum
        with pytest.raises(ValidationError) as exc_info:
            MessageCreateParams(
                model="claude-3-5-sonnet-20241022",
                messages=[Message(role="user", content="Hello")],
                max_tokens=200001,
            )
        assert "less than or equal to 200000" in str(exc_info.value)

        # Test minimum value
        with pytest.raises(ValidationError) as exc_info:
            MessageCreateParams(
                model="claude-3-5-sonnet-20241022",
                messages=[Message(role="user", content="Hello")],
                max_tokens=0,
            )
        assert "greater than or equal to 1" in str(exc_info.value)

    def test_model_validation(self):
        """Test model validation."""
        # Test valid Claude models
        valid_models = [
            "claude-opus-4-20250514",
            "claude-sonnet-4-20250514",
            "claude-3-5-sonnet-20241022",
            "claude-3-opus-20240229",
            "claude-3-5-haiku",
        ]

        for model in valid_models:
            params = MessageCreateParams(
                model=model,
                messages=[Message(role="user", content="Hello")],
                max_tokens=100,
            )
            assert params.model == model

        # Test invalid model
        with pytest.raises(ValidationError) as exc_info:
            MessageCreateParams(
                model="gpt-4",
                messages=[Message(role="user", content="Hello")],
                max_tokens=100,
            )
        # Check that it's a validation error for model pattern
        assert "String should match pattern" in str(
            exc_info.value
        ) or "Model gpt-4 is not supported" in str(exc_info.value)

    def test_stop_sequences_validation(self):
        """Test stop_sequences validation."""
        # Test valid stop sequences
        params = MessageCreateParams(
            model="claude-3-5-sonnet-20241022",
            messages=[Message(role="user", content="Hello")],
            max_tokens=100,
            stop_sequences=["END", "STOP", "DONE", "FINISH"],
        )
        assert len(params.stop_sequences) == 4  # type: ignore

        # Test too many stop sequences
        with pytest.raises(ValidationError) as exc_info:
            MessageCreateParams(
                model="claude-3-5-sonnet-20241022",
                messages=[Message(role="user", content="Hello")],
                max_tokens=100,
                stop_sequences=["1", "2", "3", "4", "5"],
            )
        # Check for validation error about list length
        assert "should have at most 4 items" in str(
            exc_info.value
        ) or "Maximum 4 stop sequences allowed" in str(exc_info.value)

        # Test stop sequence too long
        with pytest.raises(ValidationError) as exc_info:
            MessageCreateParams(
                model="claude-3-5-sonnet-20241022",
                messages=[Message(role="user", content="Hello")],
                max_tokens=100,
                stop_sequences=["x" * 101],
            )
        assert "Stop sequences must be 100 characters or less" in str(exc_info.value)

    def test_message_alternation_validation(self):
        """Test message alternation validation."""
        # Test valid alternation
        params = MessageCreateParams(
            model="claude-3-5-sonnet-20241022",
            messages=[
                Message(role="user", content="Hello"),
                Message(role="assistant", content="Hi there!"),
                Message(role="user", content="How are you?"),
            ],
            max_tokens=100,
        )
        assert len(params.messages) == 3

        # Test first message not from user
        with pytest.raises(ValidationError) as exc_info:
            MessageCreateParams(
                model="claude-3-5-sonnet-20241022",
                messages=[Message(role="assistant", content="Hello")],
                max_tokens=100,
            )
        assert "First message must be from user" in str(exc_info.value)

        # Test consecutive same role
        with pytest.raises(ValidationError) as exc_info:
            MessageCreateParams(
                model="claude-3-5-sonnet-20241022",
                messages=[
                    Message(role="user", content="Hello"),
                    Message(role="user", content="Are you there?"),
                ],
                max_tokens=100,
            )
        assert "Messages must alternate between user and assistant" in str(
            exc_info.value
        )


@pytest.mark.unit
class TestThinkingConfig:
    """Test ThinkingConfig model."""

    def test_valid_thinking_config(self):
        """Test creating a valid ThinkingConfig."""
        config = ThinkingConfig(type="enabled", budget_tokens=2048)
        assert config.type == "enabled"
        assert config.budget_tokens == 2048

    def test_thinking_config_minimum_budget(self):
        """Test minimum budget_tokens validation."""
        # Test minimum allowed value
        config = ThinkingConfig(type="enabled", budget_tokens=1024)
        assert config.budget_tokens == 1024

        # Test below minimum
        with pytest.raises(ValidationError) as exc_info:
            ThinkingConfig(type="enabled", budget_tokens=1023)
        assert "greater than or equal to 1024" in str(exc_info.value)

    def test_thinking_config_defaults(self):
        """Test ThinkingConfig default values."""
        config = ThinkingConfig(budget_tokens=5000)
        assert config.type == "enabled"  # default value


@pytest.mark.unit
class TestMetadataParams:
    """Test MetadataParams model."""

    def test_valid_metadata(self):
        """Test creating valid MetadataParams."""
        metadata = MetadataParams(user_id="user-123-abc")
        assert metadata.user_id == "user-123-abc"

    def test_metadata_user_id_length(self):
        """Test user_id length validation."""
        # Test maximum length
        metadata = MetadataParams(user_id="x" * 256)
        assert len(metadata.user_id) == 256  # type: ignore

        # Test exceeding maximum length
        with pytest.raises(ValidationError) as exc_info:
            MetadataParams(user_id="x" * 257)
        assert "should have at most 256 characters" in str(exc_info.value)

    def test_metadata_extra_fields(self):
        """Test that metadata allows extra fields."""
        metadata = MetadataParams(
            user_id="user-123",
            custom_field="custom_value",  # type: ignore
            another_field=42,  # type: ignore
        )
        assert metadata.user_id == "user-123"
        # Extra fields should be preserved due to Config.extra = "allow"
        assert hasattr(metadata, "custom_field")
        assert hasattr(metadata, "another_field")

    def test_metadata_none_user_id(self):
        """Test metadata with None user_id."""
        metadata = MetadataParams(user_id=None)
        assert metadata.user_id is None

        # Also test creating without user_id
        metadata = MetadataParams()
        assert metadata.user_id is None


@pytest.mark.unit
class TestToolChoiceParams:
    """Test ToolChoiceParams model."""

    def test_valid_tool_choice_auto(self):
        """Test creating ToolChoiceParams with auto type."""
        choice = ToolChoiceParams(type="auto")
        assert choice.type == "auto"
        assert choice.name is None
        assert choice.disable_parallel_tool_use is False  # default

    def test_valid_tool_choice_specific_tool(self):
        """Test creating ToolChoiceParams for specific tool."""
        choice = ToolChoiceParams(
            type="tool",
            name="calculator",
            disable_parallel_tool_use=True,
        )
        assert choice.type == "tool"
        assert choice.name == "calculator"
        assert choice.disable_parallel_tool_use is True

    def test_tool_choice_any_type(self):
        """Test creating ToolChoiceParams with any type."""
        choice = ToolChoiceParams(type="any")
        assert choice.type == "any"
        assert choice.name is None

    def test_tool_choice_defaults(self):
        """Test ToolChoiceParams default values."""
        choice = ToolChoiceParams(type="auto")
        assert choice.disable_parallel_tool_use is False
        assert choice.name is None


@pytest.mark.unit
class TestSystemMessage:
    """Test SystemMessage model."""

    def test_valid_system_message(self):
        """Test creating a valid SystemMessage."""
        msg = SystemMessage(type="text", text="You are a helpful assistant.")
        assert msg.type == "text"
        assert msg.text == "You are a helpful assistant."

    def test_system_message_defaults(self):
        """Test SystemMessage default values."""
        msg = SystemMessage(text="Hello")
        assert msg.type == "text"  # default value


@pytest.mark.unit
class TestMessageResponse:
    """Test MessageResponse model."""

    def test_basic_message_response(self):
        """Test creating a basic MessageResponse."""
        from ccproxy.models.messages import MessageContentBlock, MessageResponse
        from ccproxy.models.requests import Usage

        response = MessageResponse(
            id="msg_123",
            type="message",
            role="assistant",
            content=[
                MessageContentBlock(
                    type="text",
                    text="Hello, how can I help you?",
                )
            ],
            model="claude-3-5-sonnet-20241022",
            stop_reason="end_turn",
            usage=Usage(input_tokens=10, output_tokens=15),
        )

        assert response.id == "msg_123"
        assert response.type == "message"
        assert response.role == "assistant"
        assert len(response.content) == 1
        assert response.content[0].type == "text"
        assert response.content[0].text == "Hello, how can I help you?"
        assert response.model == "claude-3-5-sonnet-20241022"
        assert response.stop_reason == "end_turn"
        assert response.stop_sequence is None
        assert response.usage.input_tokens == 10
        assert response.usage.output_tokens == 15

    def test_message_response_with_thinking_block(self):
        """Test MessageResponse with thinking content block."""
        from ccproxy.models.messages import MessageContentBlock, MessageResponse
        from ccproxy.models.requests import Usage

        response = MessageResponse(
            id="msg_456",
            type="message",
            role="assistant",
            content=[
                MessageContentBlock(
                    type="thinking",
                    text="Let me think about this...",
                ),
                MessageContentBlock(
                    type="text",
                    text="The answer is 42.",
                ),
            ],
            model="claude-opus-4-20250514",
            stop_reason="end_turn",
            usage=Usage(input_tokens=20, output_tokens=30),
        )

        assert len(response.content) == 2
        assert response.content[0].type == "thinking"
        assert response.content[0].text == "Let me think about this..."
        assert response.content[1].type == "text"
        assert response.content[1].text == "The answer is 42."

    def test_message_response_with_tool_use(self):
        """Test MessageResponse with tool_use content block."""
        from ccproxy.models.messages import MessageContentBlock, MessageResponse
        from ccproxy.models.requests import Usage

        response = MessageResponse(
            id="msg_789",
            type="message",
            role="assistant",
            content=[
                MessageContentBlock(
                    type="tool_use",
                    id="tool_use_1",
                    name="calculator",
                    input={"operation": "add", "a": 2, "b": 2},
                )
            ],
            model="claude-3-5-sonnet-20241022",
            stop_reason="tool_use",
            usage=Usage(input_tokens=15, output_tokens=10),
        )

        assert response.content[0].type == "tool_use"
        assert response.content[0].id == "tool_use_1"
        assert response.content[0].name == "calculator"
        assert response.content[0].input == {"operation": "add", "a": 2, "b": 2}
        assert response.stop_reason == "tool_use"

    def test_message_response_with_stop_sequence(self):
        """Test MessageResponse with stop_sequence."""
        from ccproxy.models.messages import MessageContentBlock, MessageResponse
        from ccproxy.models.requests import Usage

        response = MessageResponse(
            id="msg_stop",
            type="message",
            role="assistant",
            content=[
                MessageContentBlock(
                    type="text",
                    text="The story begins...",
                )
            ],
            model="claude-3-5-sonnet-20241022",
            stop_reason="stop_sequence",
            stop_sequence="END",
            usage=Usage(input_tokens=20, output_tokens=10),
        )

        assert response.stop_reason == "stop_sequence"
        assert response.stop_sequence == "END"

    def test_message_response_with_container(self):
        """Test MessageResponse with container field."""
        from ccproxy.models.messages import MessageContentBlock, MessageResponse
        from ccproxy.models.requests import Usage

        response = MessageResponse(
            id="msg_container",
            type="message",
            role="assistant",
            content=[
                MessageContentBlock(
                    type="text",
                    text="Response with container info",
                )
            ],
            model="claude-3-5-sonnet-20241022",
            stop_reason="end_turn",
            usage=Usage(input_tokens=10, output_tokens=15),
            container={"id": "container_123", "version": "v1"},
        )

        assert response.container is not None
        assert response.container["id"] == "container_123"
        assert response.container["version"] == "v1"

    def test_all_stop_reasons(self):
        """Test all valid stop_reason values."""
        from ccproxy.models.messages import MessageContentBlock, MessageResponse
        from ccproxy.models.requests import Usage
        from ccproxy.models.types import StopReason

        stop_reasons: list[StopReason] = [
            "end_turn",
            "max_tokens",
            "stop_sequence",
            "tool_use",
            "pause_turn",
            "refusal",
        ]

        for stop_reason in stop_reasons:
            response = MessageResponse(
                id=f"msg_{stop_reason}",
                type="message",
                role="assistant",
                content=[MessageContentBlock(type="text", text="Test")],
                model="claude-3-5-sonnet-20241022",
                stop_reason=stop_reason,
                usage=Usage(input_tokens=10, output_tokens=5),
                container=None,
            )
            assert response.stop_reason == stop_reason

    def test_message_response_validation(self):
        """Test MessageResponse validation."""
        from ccproxy.models.messages import MessageResponse
        from ccproxy.models.requests import Usage

        # Test invalid stop_reason
        with pytest.raises(ValidationError) as exc_info:
            MessageResponse(
                id="msg_invalid",
                type="message",
                role="assistant",
                content=[],
                model="claude-3-5-sonnet-20241022",
                stop_reason="invalid_reason",  # type: ignore
                usage=Usage(input_tokens=10, output_tokens=5),
            )
        assert "Input should be" in str(exc_info.value)


@pytest.mark.unit
class TestMessageContentBlock:
    """Test MessageContentBlock model."""

    def test_text_content_block(self):
        """Test creating a text content block."""
        from ccproxy.models.messages import MessageContentBlock

        block = MessageContentBlock(type="text", text="Hello world")
        assert block.type == "text"
        assert block.text == "Hello world"
        assert block.id is None
        assert block.name is None
        assert block.input is None

    def test_thinking_content_block(self):
        """Test creating a thinking content block."""
        from ccproxy.models.messages import MessageContentBlock

        block = MessageContentBlock(type="thinking", text="Processing the request...")
        assert block.type == "thinking"
        assert block.text == "Processing the request..."

    def test_tool_use_content_block(self):
        """Test creating a tool_use content block."""
        from ccproxy.models.messages import MessageContentBlock

        block = MessageContentBlock(
            type="tool_use",
            id="tool_123",
            name="web_search",
            input={"query": "weather today"},
        )
        assert block.type == "tool_use"
        assert block.id == "tool_123"
        assert block.name == "web_search"
        assert block.input == {"query": "weather today"}
        assert block.text is None
