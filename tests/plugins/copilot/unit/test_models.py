"""Unit tests for Copilot plugin models."""

from ccproxy.plugins.copilot.models import (
    CopilotChatRequest,
    CopilotMessage,
    CopilotModel,
    CopilotUsage,
)


class TestCopilotMessage:
    """Test cases for CopilotMessage."""

    def test_basic_initialization(self) -> None:
        """Test basic message initialization."""
        message = CopilotMessage(
            role="user",
            content="Hello, world!",
        )

        assert message.role == "user"
        assert message.content == "Hello, world!"
        assert message.name is None

    def test_initialization_with_name(self) -> None:
        """Test message initialization with name."""
        message = CopilotMessage(
            role="assistant",
            content="Hi there!",
            name="copilot-assistant",
        )

        assert message.role == "assistant"
        assert message.content == "Hi there!"
        assert message.name == "copilot-assistant"

    def test_role_validation(self) -> None:
        """Test that valid roles are accepted."""
        valid_roles = ["system", "user", "assistant"]

        for role in valid_roles:
            message = CopilotMessage(role=role, content="test content")
            assert message.role == role

    def test_serialization(self) -> None:
        """Test message serialization."""
        message = CopilotMessage(
            role="user",
            content="Test message",
            name="test_user",
        )

        data = message.model_dump()

        assert data["role"] == "user"
        assert data["content"] == "Test message"
        assert data["name"] == "test_user"

    def test_serialization_exclude_none(self) -> None:
        """Test message serialization excluding None values."""
        message = CopilotMessage(
            role="user",
            content="Test message",
            name=None,
        )

        data = message.model_dump(exclude_none=True)

        assert data["role"] == "user"
        assert data["content"] == "Test message"
        assert "name" not in data

    def test_deserialization(self) -> None:
        """Test message deserialization."""
        data = {
            "role": "assistant",
            "content": "Hello from assistant",
            "name": "helpful_assistant",
        }

        message = CopilotMessage(**data)

        assert message.role == "assistant"
        assert message.content == "Hello from assistant"
        assert message.name == "helpful_assistant"

    def test_empty_content_handling(self) -> None:
        """Test handling of empty content."""
        message = CopilotMessage(
            role="assistant",
            content="",
        )

        assert message.role == "assistant"
        assert message.content == ""

    def test_unicode_content(self) -> None:
        """Test handling of Unicode content."""
        message = CopilotMessage(
            role="user",
            content="Hello 世界! 🌍 emoji test",
        )

        assert message.content == "Hello 世界! 🌍 emoji test"


class TestCopilotChatRequest:
    """Test cases for CopilotChatRequest."""

    def test_basic_initialization(self) -> None:
        """Test basic chat request initialization."""
        messages = [
            CopilotMessage(role="user", content="Hello"),
        ]

        request = CopilotChatRequest(
            messages=messages,
            model="copilot-chat",
        )

        assert len(request.messages) == 1
        assert request.messages[0].content == "Hello"
        assert request.model == "copilot-chat"
        assert request.stream is False
        assert request.temperature is None
        assert request.max_tokens is None

    def test_initialization_with_all_parameters(self) -> None:
        """Test initialization with all parameters."""
        messages = [
            CopilotMessage(role="system", content="You are helpful"),
            CopilotMessage(role="user", content="Hello"),
        ]

        request = CopilotChatRequest(
            messages=messages,
            model="gpt-4",
            temperature=0.7,
            max_tokens=150,
            top_p=0.9,
            stream=True,
            stop=["\\n", "END"],
        )

        assert len(request.messages) == 2
        assert request.model == "gpt-4"
        assert request.temperature == 0.7
        assert request.max_tokens == 150
        assert request.top_p == 0.9
        assert request.stream is True
        assert request.stop == ["\\n", "END"]

    def test_default_values(self) -> None:
        """Test default parameter values."""
        messages = [CopilotMessage(role="user", content="Test")]

        request = CopilotChatRequest(
            messages=messages,
            model="test-model",
        )

        assert request.stream is False
        assert request.temperature is None
        assert request.max_tokens is None
        assert request.top_p is None
        assert request.stop is None

    def test_empty_messages_list(self) -> None:
        """Test handling of empty messages list."""
        request = CopilotChatRequest(
            messages=[],
            model="test-model",
        )

        assert len(request.messages) == 0
        assert request.model == "test-model"

    def test_serialization(self) -> None:
        """Test request serialization."""
        messages = [
            CopilotMessage(role="user", content="Hello"),
        ]

        request = CopilotChatRequest(
            messages=messages,
            model="gpt-4",
            temperature=0.8,
            stream=True,
        )

        data = request.model_dump()

        assert len(data["messages"]) == 1
        assert data["messages"][0]["role"] == "user"
        assert data["messages"][0]["content"] == "Hello"
        assert data["model"] == "gpt-4"
        assert data["temperature"] == 0.8
        assert data["stream"] is True

    def test_serialization_exclude_none(self) -> None:
        """Test request serialization excluding None values."""
        messages = [CopilotMessage(role="user", content="Hello")]

        request = CopilotChatRequest(
            messages=messages,
            model="gpt-4",
            temperature=0.7,
            max_tokens=None,  # Should be excluded
            stream=False,
        )

        data = request.model_dump(exclude_none=True)

        assert data["model"] == "gpt-4"
        assert data["temperature"] == 0.7
        assert data["stream"] is False  # False is included
        assert "max_tokens" not in data
        assert "top_p" not in data
        assert "stop" not in data

    def test_deserialization(self) -> None:
        """Test request deserialization."""
        data = {
            "messages": [
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "Hello", "name": "john"},
            ],
            "model": "gpt-4",
            "temperature": 0.9,
            "max_tokens": 200,
            "stream": True,
            "stop": ["\\n"],
        }

        request = CopilotChatRequest(**data)

        assert len(request.messages) == 2
        assert request.messages[0].role == "system"
        assert request.messages[0].content == "You are helpful"
        assert request.messages[1].role == "user"
        assert request.messages[1].content == "Hello"
        assert request.messages[1].name == "john"
        assert request.model == "gpt-4"
        assert request.temperature == 0.9
        assert request.max_tokens == 200
        assert request.stream is True
        assert request.stop == ["\\n"]

    def test_temperature_bounds(self) -> None:
        """Test temperature parameter bounds."""
        messages = [CopilotMessage(role="user", content="Test")]

        # Test valid temperatures
        for temp in [0.0, 0.5, 1.0, 1.5, 2.0]:
            request = CopilotChatRequest(
                messages=messages,
                model="test-model",
                temperature=temp,
            )
            assert request.temperature == temp

    def test_max_tokens_validation(self) -> None:
        """Test max_tokens parameter validation."""
        messages = [CopilotMessage(role="user", content="Test")]

        # Test valid max_tokens values
        for max_tokens in [1, 100, 1000, 4000]:
            request = CopilotChatRequest(
                messages=messages,
                model="test-model",
                max_tokens=max_tokens,
            )
            assert request.max_tokens == max_tokens

    def test_top_p_bounds(self) -> None:
        """Test top_p parameter bounds."""
        messages = [CopilotMessage(role="user", content="Test")]

        # Test valid top_p values
        for top_p in [0.1, 0.5, 0.9, 1.0]:
            request = CopilotChatRequest(
                messages=messages,
                model="test-model",
                top_p=top_p,
            )
            assert request.top_p == top_p

    def test_stop_sequences_types(self) -> None:
        """Test different stop sequence types."""
        messages = [CopilotMessage(role="user", content="Test")]

        # Test string stop sequence
        request1 = CopilotChatRequest(
            messages=messages,
            model="test-model",
            stop="\\n",
        )
        assert request1.stop == "\\n"

        # Test list of stop sequences
        request2 = CopilotChatRequest(
            messages=messages,
            model="test-model",
            stop=["\\n", "END", "STOP"],
        )
        assert request2.stop == ["\\n", "END", "STOP"]


class TestCopilotModel:
    """Test cases for CopilotModel."""

    def test_basic_initialization(self) -> None:
        """Test basic model initialization."""
        model = CopilotModel(
            id="copilot-chat",
            object="model",
            owned_by="github",
        )

        assert model.id == "copilot-chat"
        assert model.object == "model"
        assert model.owned_by == "github"

    def test_serialization(self) -> None:
        """Test model serialization."""
        model = CopilotModel(
            id="gpt-4-copilot",
            object="model",
            owned_by="github",
        )

        data = model.model_dump()

        assert data["id"] == "gpt-4-copilot"
        assert data["object"] == "model"
        assert data["owned_by"] == "github"

    def test_deserialization(self) -> None:
        """Test model deserialization."""
        data = {
            "id": "copilot-preview",
            "object": "model",
            "owned_by": "github",
        }

        model = CopilotModel(**data)

        assert model.id == "copilot-preview"
        assert model.object == "model"
        assert model.owned_by == "github"

    def test_model_id_variations(self) -> None:
        """Test various model ID formats."""
        model_ids = [
            "copilot-chat",
            "gpt-4-copilot",
            "copilot-preview",
            "github-copilot-v1",
        ]

        for model_id in model_ids:
            model = CopilotModel(
                id=model_id,
                object="model",
                owned_by="github",
            )
            assert model.id == model_id


class TestCopilotUsage:
    """Test cases for CopilotUsage."""

    def test_basic_initialization(self) -> None:
        """Test basic usage info initialization."""
        usage = CopilotUsage(
            prompt_tokens=10,
            completion_tokens=25,
            total_tokens=35,
        )

        assert usage.prompt_tokens == 10
        assert usage.completion_tokens == 25
        assert usage.total_tokens == 35

    def test_zero_values(self) -> None:
        """Test usage info with zero values."""
        usage = CopilotUsage(
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
        )

        assert usage.prompt_tokens == 0
        assert usage.completion_tokens == 0
        assert usage.total_tokens == 0

    def test_large_values(self) -> None:
        """Test usage info with large token counts."""
        usage = CopilotUsage(
            prompt_tokens=100000,
            completion_tokens=50000,
            total_tokens=150000,
        )

        assert usage.prompt_tokens == 100000
        assert usage.completion_tokens == 50000
        assert usage.total_tokens == 150000

    def test_serialization(self) -> None:
        """Test usage info serialization."""
        usage = CopilotUsage(
            prompt_tokens=15,
            completion_tokens=30,
            total_tokens=45,
        )

        data = usage.model_dump()

        assert data["prompt_tokens"] == 15
        assert data["completion_tokens"] == 30
        assert data["total_tokens"] == 45

    def test_deserialization(self) -> None:
        """Test usage info deserialization."""
        data = {
            "prompt_tokens": 20,
            "completion_tokens": 40,
            "total_tokens": 60,
        }

        usage = CopilotUsage(**data)

        assert usage.prompt_tokens == 20
        assert usage.completion_tokens == 40
        assert usage.total_tokens == 60

    def test_token_calculation_consistency(self) -> None:
        """Test that total tokens can be calculated correctly."""
        usage = CopilotUsage(
            prompt_tokens=100,
            completion_tokens=200,
            total_tokens=300,
        )

        # Total should match sum (though this is just data validation, not enforced)
        expected_total = usage.prompt_tokens + usage.completion_tokens
        assert usage.total_tokens == expected_total

    def test_optional_completion_tokens(self) -> None:
        """Test usage with optional completion tokens."""
        usage = CopilotUsage(
            prompt_tokens=10,
            completion_tokens=None,
            total_tokens=10,
        )

        assert usage.prompt_tokens == 10
        assert usage.completion_tokens is None
        assert usage.total_tokens == 10
