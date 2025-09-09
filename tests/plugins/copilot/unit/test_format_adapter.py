"""Unit tests for Copilot format adapters."""

import pytest

from ccproxy.plugins.copilot.format_adapter import (
    CopilotFormatAdapter,
    CopilotToOpenAIAdapter,
    OpenAIToCopilotAdapter,
)


class TestOpenAIToCopilotAdapter:
    """Test cases for OpenAI to Copilot format adapter."""

    @pytest.fixture
    def adapter(self) -> OpenAIToCopilotAdapter:
        """Create OpenAI to Copilot adapter instance."""
        return OpenAIToCopilotAdapter()

    @pytest.mark.asyncio
    async def test_basic_request_adaptation(
        self, adapter: OpenAIToCopilotAdapter
    ) -> None:
        """Test basic request format adaptation."""
        openai_request = {
            "model": "gpt-4",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello, world!"},
            ],
            "temperature": 0.7,
            "max_tokens": 150,
        }

        result = await adapter.adapt_request(openai_request)

        # Since Copilot uses OpenAI format, the result should be validated and cleaned
        assert result["model"] == "gpt-4"
        assert len(result["messages"]) == 2
        assert result["messages"][0]["role"] == "system"
        assert result["messages"][0]["content"] == "You are a helpful assistant."
        assert result["messages"][1]["role"] == "user"
        assert result["messages"][1]["content"] == "Hello, world!"
        assert result["temperature"] == 0.7
        assert result["max_tokens"] == 150

    @pytest.mark.asyncio
    async def test_minimal_request(self, adapter: OpenAIToCopilotAdapter) -> None:
        """Test minimal request with just required fields."""
        openai_request = {
            "messages": [
                {"role": "user", "content": "Hello"},
            ],
            "model": "gpt-4",  # Model is required by OpenAI spec
        }

        result = await adapter.adapt_request(openai_request)

        assert result["model"] == "gpt-4"
        assert len(result["messages"]) == 1
        assert result["messages"][0]["role"] == "user"
        assert result["messages"][0]["content"] == "Hello"

    @pytest.mark.asyncio
    async def test_response_adaptation(self, adapter: OpenAIToCopilotAdapter) -> None:
        """Test response format adaptation."""
        copilot_response = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1677652288,
            "model": "gpt-4",
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
            "usage": {"prompt_tokens": 9, "completion_tokens": 12, "total_tokens": 21},
        }

        result = await adapter.adapt_response(copilot_response)

        # Should pass through with minimal changes since formats are compatible
        assert result["id"] == "chatcmpl-123"
        assert result["object"] == "chat.completion"
        assert result["model"] == "gpt-4"
        assert len(result["choices"]) == 1
        assert (
            result["choices"][0]["message"]["content"]
            == "Hello! How can I help you today?"
        )

    @pytest.mark.asyncio
    async def test_stream_chunk_adaptation(
        self, adapter: OpenAIToCopilotAdapter
    ) -> None:
        """Test streaming chunk adaptation."""
        copilot_chunk = {
            "id": "chatcmpl-123",
            "object": "chat.completion.chunk",
            "created": 1677652288,
            "model": "gpt-4",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": "Hello"},
                    "finish_reason": None,
                }
            ],
        }

        result = await adapter.adapt_stream_chunk(copilot_chunk)

        assert result["id"] == "chatcmpl-123"
        assert result["object"] == "chat.completion.chunk"
        assert result["model"] == "gpt-4"
        assert result["choices"][0]["delta"]["content"] == "Hello"


class TestCopilotToOpenAIAdapter:
    """Test cases for Copilot to OpenAI format adapter."""

    @pytest.fixture
    def adapter(self) -> CopilotToOpenAIAdapter:
        """Create Copilot to OpenAI adapter instance."""
        return CopilotToOpenAIAdapter()

    @pytest.mark.asyncio
    async def test_request_adaptation(self, adapter: CopilotToOpenAIAdapter) -> None:
        """Test Copilot request to OpenAI format adaptation."""
        copilot_request = {
            "messages": [{"role": "user", "content": "Hello"}],
            "model": "gpt-4",
            "temperature": 0.5,
            "stream": True,
        }

        result = await adapter.adapt_request(copilot_request)

        # Should be essentially pass-through since formats are compatible
        assert result["messages"] == [{"role": "user", "content": "Hello"}]
        assert result["model"] == "gpt-4"
        assert result["temperature"] == 0.5
        assert result["stream"] is True

    @pytest.mark.asyncio
    async def test_response_adaptation(self, adapter: CopilotToOpenAIAdapter) -> None:
        """Test response adaptation from OpenAI to Copilot format."""
        openai_response = {
            "id": "chatcmpl-456",
            "object": "chat.completion",
            "created": 1677652288,
            "model": "gpt-4",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hi there!"},
                    "finish_reason": "stop",
                }
            ],
        }

        result = await adapter.adapt_response(openai_response)

        # Should preserve all original fields
        assert result["id"] == "chatcmpl-456"
        assert result["object"] == "chat.completion"
        assert result["model"] == "gpt-4"
        assert result["choices"][0]["message"]["content"] == "Hi there!"


class TestCopilotFormatAdapter:
    """Test cases for bidirectional format adapter."""

    @pytest.fixture
    def adapter(self) -> CopilotFormatAdapter:
        """Create bidirectional format adapter instance."""
        return CopilotFormatAdapter()

    @pytest.mark.asyncio
    async def test_no_conversion_needed(self, adapter: CopilotFormatAdapter) -> None:
        """Test that no conversion occurs when formats are the same."""
        request = {"messages": [{"role": "user", "content": "Hello"}]}

        result = await adapter.adapt_request(request, "openai", "openai")
        assert result == request

        result = await adapter.adapt_request(request, "copilot", "copilot")
        assert result == request

    @pytest.mark.asyncio
    async def test_openai_to_copilot_conversion(
        self, adapter: CopilotFormatAdapter
    ) -> None:
        """Test OpenAI to Copilot format conversion."""
        openai_request = {
            "messages": [{"role": "user", "content": "Hello"}],
            "model": "gpt-4",
        }

        result = await adapter.adapt_request(openai_request, "openai", "copilot")

        assert result["messages"][0]["role"] == "user"
        assert result["messages"][0]["content"] == "Hello"
        assert result["model"] == "gpt-4"

    @pytest.mark.asyncio
    async def test_copilot_to_openai_conversion(
        self, adapter: CopilotFormatAdapter
    ) -> None:
        """Test Copilot to OpenAI format conversion."""
        copilot_request = {
            "messages": [{"role": "user", "content": "Hello"}],
            "model": "gpt-4",
        }

        result = await adapter.adapt_request(copilot_request, "copilot", "openai")

        assert result["messages"][0]["role"] == "user"
        assert result["messages"][0]["content"] == "Hello"
        assert result["model"] == "gpt-4"

    @pytest.mark.asyncio
    async def test_unsupported_format_conversion(
        self, adapter: CopilotFormatAdapter
    ) -> None:
        """Test handling of unsupported format conversions."""
        request = {"messages": [{"role": "user", "content": "Hello"}]}

        # Should return original request unchanged for unsupported conversions
        result = await adapter.adapt_request(request, "unknown", "openai")
        assert result == request

        result = await adapter.adapt_response(request, "unknown", "copilot")
        assert result == request

    @pytest.mark.asyncio
    async def test_stream_processing(self, adapter: CopilotFormatAdapter) -> None:
        """Test stream processing with format conversion."""

        async def mock_stream():
            chunks = [
                {
                    "id": "1",
                    "object": "chat.completion.chunk",
                    "choices": [{"delta": {"content": "Hello"}}],
                },
                {
                    "id": "1",
                    "object": "chat.completion.chunk",
                    "choices": [{"delta": {"content": " world"}}],
                },
                {
                    "id": "1",
                    "object": "chat.completion.chunk",
                    "choices": [{"finish_reason": "stop"}],
                },
            ]
            for chunk in chunks:
                yield chunk

        result_chunks = []
        async for chunk in adapter.process_stream(mock_stream(), "copilot", "openai"):
            result_chunks.append(chunk)

        assert len(result_chunks) == 3
        assert result_chunks[0]["choices"][0]["delta"]["content"] == "Hello"
        assert result_chunks[1]["choices"][0]["delta"]["content"] == " world"
        assert result_chunks[2]["choices"][0]["finish_reason"] == "stop"
