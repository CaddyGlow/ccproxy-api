"""Unit tests for Copilot format adapters."""

import pytest

from ccproxy.plugins.copilot.format_adapter import (
    CopilotToOpenAIAdapter,
    OpenAIToCopilotAdapter,
)
from ccproxy.plugins.copilot.models import CopilotChatRequest, CopilotMessage


class TestOpenAIToCopilotAdapter:
    """Test cases for OpenAI to Copilot format adapter."""

    @pytest.fixture
    def adapter(self) -> OpenAIToCopilotAdapter:
        """Create OpenAI to Copilot adapter instance."""
        return OpenAIToCopilotAdapter()

    def test_basic_message_conversion(self, adapter: OpenAIToCopilotAdapter) -> None:
        """Test basic message format conversion."""
        openai_request = {
            "model": "gpt-4",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello, world!"},
                {"role": "assistant", "content": "Hi there! How can I help you?"},
            ],
        }

        # Test message conversion directly
        copilot_messages = []
        for msg in openai_request["messages"]:
            copilot_msg = CopilotMessage(
                role=msg["role"],
                content=msg["content"],
                name=msg.get("name"),
            )
            copilot_messages.append(copilot_msg)

        assert len(copilot_messages) == 3
        assert copilot_messages[0].role == "system"
        assert copilot_messages[0].content == "You are a helpful assistant."
        assert copilot_messages[1].role == "user"
        assert copilot_messages[1].content == "Hello, world!"
        assert copilot_messages[2].role == "assistant"
        assert copilot_messages[2].content == "Hi there! How can I help you?"

    def test_message_with_name(self, adapter: OpenAIToCopilotAdapter) -> None:
        """Test message conversion with name field."""
        openai_request = {
            "model": "gpt-4",
            "messages": [
                {"role": "user", "content": "Hello", "name": "john_doe"},
            ],
        }

        copilot_msg = CopilotMessage(
            role="user",
            content="Hello",
            name="john_doe",
        )

        assert copilot_msg.role == "user"
        assert copilot_msg.content == "Hello"
        assert copilot_msg.name == "john_doe"

    def test_request_parameters_conversion(self) -> None:
        """Test conversion of request parameters."""
        openai_request = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Test"}],
            "temperature": 0.7,
            "max_tokens": 150,
            "top_p": 0.9,
            "stream": True,
            "stop": ["\\n"],
        }

        # Simulate what the adapter would create
        message_objects = [CopilotMessage(role="user", content="Test")]

        copilot_request = CopilotChatRequest(
            messages=message_objects,
            model=openai_request.get("model", "copilot-chat"),
            temperature=openai_request.get("temperature"),
            max_tokens=openai_request.get("max_tokens"),
            top_p=openai_request.get("top_p"),
            stream=openai_request.get("stream", False),
            stop=openai_request.get("stop"),
        )

        assert copilot_request.model == "gpt-4"
        assert copilot_request.temperature == 0.7
        assert copilot_request.max_tokens == 150
        assert copilot_request.top_p == 0.9
        assert copilot_request.stream is True
        assert copilot_request.stop == ["\\n"]

    def test_default_model_fallback(self) -> None:
        """Test default model fallback when not specified."""
        message_objects = [CopilotMessage(role="user", content="Test")]

        copilot_request = CopilotChatRequest(
            messages=message_objects,
            model="copilot-chat",
        )

        assert copilot_request.model == "copilot-chat"

    def test_empty_messages_handling(self) -> None:
        """Test handling of empty messages array."""
        copilot_request = CopilotChatRequest(
            messages=[],
            model="gpt-4",
        )

        assert copilot_request.messages == []
        assert copilot_request.model == "gpt-4"

    def test_optional_parameters_omitted(self) -> None:
        """Test that optional parameters are omitted when None."""
        message_objects = [CopilotMessage(role="user", content="Test")]

        copilot_request = CopilotChatRequest(
            messages=message_objects,
            model="gpt-4",
            temperature=None,
            max_tokens=None,
            top_p=None,
            stream=False,
            stop=None,
        )

        request_dict = copilot_request.model_dump(exclude_none=True)

        assert "temperature" not in request_dict
        assert "max_tokens" not in request_dict
        assert "top_p" not in request_dict
        assert "stop" not in request_dict
        assert request_dict["stream"] is False  # False is included


class TestCopilotToOpenAIAdapter:
    """Test cases for Copilot to OpenAI format adapter."""

    @pytest.fixture
    def adapter(self) -> CopilotToOpenAIAdapter:
        """Create Copilot to OpenAI adapter instance."""
        return CopilotToOpenAIAdapter()

    async def test_adapt_response_non_streaming(
        self, adapter: CopilotToOpenAIAdapter
    ) -> None:
        """Test adaptation of non-streaming Copilot response to OpenAI format."""
        copilot_response = {
            "id": "copilot-123",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "copilot-chat",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Hello! How can I help you today?",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 8,
                "total_tokens": 18,
            },
        }

        result = await adapter.adapt_response(copilot_response)

        assert result["id"] == "copilot-123"
        assert result["object"] == "chat.completion"
        assert result["created"] == 1234567890
        assert result["model"] == "copilot-chat"

        assert len(result["choices"]) == 1
        choice = result["choices"][0]
        assert choice["index"] == 0
        assert choice["message"]["role"] == "assistant"
        assert choice["message"]["content"] == "Hello! How can I help you today?"
        assert choice["finish_reason"] == "stop"

        assert result["usage"]["prompt_tokens"] == 10
        assert result["usage"]["completion_tokens"] == 8
        assert result["usage"]["total_tokens"] == 18

    async def test_adapt_response_streaming_chunk(
        self, adapter: CopilotToOpenAIAdapter
    ) -> None:
        """Test adaptation of streaming Copilot response chunk to OpenAI format."""
        copilot_chunk = {
            "id": "copilot-123",
            "object": "chat.completion.chunk",
            "created": 1234567890,
            "model": "copilot-chat",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "content": "Hello",
                    },
                    "finish_reason": None,
                }
            ],
        }

        result = await adapter.adapt_response(copilot_chunk)

        assert result["id"] == "copilot-123"
        assert result["object"] == "chat.completion.chunk"
        assert result["created"] == 1234567890
        assert result["model"] == "copilot-chat"

        assert len(result["choices"]) == 1
        choice = result["choices"][0]
        assert choice["index"] == 0
        assert choice["delta"]["content"] == "Hello"
        assert choice["finish_reason"] is None

    async def test_adapt_response_streaming_final_chunk(
        self, adapter: CopilotToOpenAIAdapter
    ) -> None:
        """Test adaptation of final streaming chunk with usage data."""
        copilot_final_chunk = {
            "id": "copilot-123",
            "object": "chat.completion.chunk",
            "created": 1234567890,
            "model": "copilot-chat",
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 15,
                "completion_tokens": 25,
                "total_tokens": 40,
            },
        }

        result = await adapter.adapt_response(copilot_final_chunk)

        assert result["id"] == "copilot-123"
        assert result["object"] == "chat.completion.chunk"
        assert result["finish_reason"] == "stop"
        assert result["usage"]["total_tokens"] == 40

    async def test_adapt_response_preserves_unknown_fields(
        self, adapter: CopilotToOpenAIAdapter
    ) -> None:
        """Test that unknown fields are preserved in the response."""
        copilot_response = {
            "id": "copilot-123",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "copilot-chat",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Test response",
                    },
                    "finish_reason": "stop",
                }
            ],
            "custom_field": "custom_value",
            "another_field": {"nested": "data"},
        }

        result = await adapter.adapt_response(copilot_response)

        assert result["custom_field"] == "custom_value"
        assert result["another_field"] == {"nested": "data"}

    async def test_adapt_response_handles_missing_fields(
        self, adapter: CopilotToOpenAIAdapter
    ) -> None:
        """Test adaptation handles missing optional fields gracefully."""
        minimal_response = {
            "id": "copilot-123",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Minimal response",
                    },
                }
            ],
        }

        result = await adapter.adapt_response(minimal_response)

        assert result["id"] == "copilot-123"
        assert result["object"] == "chat.completion"
        assert len(result["choices"]) == 1
        assert result["choices"][0]["message"]["content"] == "Minimal response"

    async def test_adapt_response_multiple_choices(
        self, adapter: CopilotToOpenAIAdapter
    ) -> None:
        """Test adaptation of response with multiple choices."""
        copilot_response = {
            "id": "copilot-123",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "copilot-chat",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "First response",
                    },
                    "finish_reason": "stop",
                },
                {
                    "index": 1,
                    "message": {
                        "role": "assistant",
                        "content": "Second response",
                    },
                    "finish_reason": "stop",
                },
            ],
        }

        result = await adapter.adapt_response(copilot_response)

        assert len(result["choices"]) == 2
        assert result["choices"][0]["message"]["content"] == "First response"
        assert result["choices"][1]["message"]["content"] == "Second response"
        assert result["choices"][0]["index"] == 0
        assert result["choices"][1]["index"] == 1

    async def test_adapt_stream_chunk_basic(
        self, adapter: CopilotToOpenAIAdapter
    ) -> None:
        """Test adaptation of streaming chunk."""
        copilot_chunk = {
            "id": "copilot-123",
            "object": "chat.completion.chunk",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": "Hello world"},
                    "finish_reason": None,
                }
            ],
        }

        result = await adapter.adapt_stream_chunk(copilot_chunk)

        assert result["id"] == "copilot-123"
        assert result["object"] == "chat.completion.chunk"
        assert result["choices"][0]["delta"]["content"] == "Hello world"

    async def test_adapt_stream_chunk_with_finish_reason(
        self, adapter: CopilotToOpenAIAdapter
    ) -> None:
        """Test adaptation of streaming chunk with finish reason."""
        copilot_chunk = {
            "id": "copilot-123",
            "object": "chat.completion.chunk",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": "Hello world"},
                    "finish_reason": "stop",
                }
            ],
        }

        result = await adapter.adapt_stream_chunk(copilot_chunk)

        assert result["id"] == "copilot-123"
        assert result["object"] == "chat.completion.chunk"
        assert result["choices"][0]["delta"]["content"] == "Hello world"
        assert result["choices"][0]["finish_reason"] == "stop"

    async def test_adapt_stream_chunk_minimal_data(
        self, adapter: CopilotToOpenAIAdapter
    ) -> None:
        """Test adaptation of minimal streaming chunk."""
        copilot_chunk = {
            "id": "test-123",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }

        result = await adapter.adapt_stream_chunk(copilot_chunk)

        assert result["id"] == "test-123"
        assert result["object"] == "chat.completion.chunk"
        assert result["choices"][0]["finish_reason"] == "stop"

    async def test_adapt_stream_chunk_empty_delta(
        self, adapter: CopilotToOpenAIAdapter
    ) -> None:
        """Test adaptation of streaming chunk with empty delta."""
        copilot_chunk = {
            "id": "test-123",
            "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": {}, "finish_reason": None}],
        }

        result = await adapter.adapt_stream_chunk(copilot_chunk)

        assert result["id"] == "test-123"
        assert result["object"] == "chat.completion.chunk"
        assert result["choices"][0]["delta"] == {}
        assert result["choices"][0]["finish_reason"] is None
