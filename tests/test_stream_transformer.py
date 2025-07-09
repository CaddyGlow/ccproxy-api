"""Tests for the unified OpenAI stream transformer."""

import asyncio
import contextlib
import json
from unittest.mock import MagicMock, Mock

import pytest

from ccproxy.services.stream_transformer import (
    ClaudeSDKEventSource,
    OpenAIStreamTransformer,
    SSEEventSource,
    StreamEvent,
    StreamingConfig,
)


# Test fixtures for Claude SDK responses
@pytest.fixture
def mock_claude_thinking_response():
    """Mock Claude response with thinking blocks."""

    async def generate():
        yield {"type": "message_start", "message": {"model": "claude-3-5-sonnet"}}
        yield {
            "type": "content_block_start",
            "content_block": {"type": "thinking"},
        }
        yield {
            "type": "content_block_delta",
            "delta": {"type": "thinking_delta", "thinking": "Let me think..."},
        }
        yield {"type": "content_block_stop"}
        yield {
            "type": "content_block_start",
            "content_block": {"type": "text"},
        }
        yield {
            "type": "content_block_delta",
            "delta": {"type": "text_delta", "text": "Here's the answer."},
        }
        yield {"type": "content_block_stop"}
        yield {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"input_tokens": 10, "output_tokens": 20},
        }

    return generate()


@pytest.fixture
def mock_claude_tool_response():
    """Mock Claude response with tool calls."""

    async def generate():
        yield {"type": "message_start", "message": {"model": "claude-3-5-sonnet"}}
        yield {
            "type": "content_block_start",
            "content_block": {
                "type": "tool_use",
                "id": "tool_123",
                "name": "calculator",
            },
        }
        yield {
            "type": "content_block_delta",
            "delta": {"type": "input_json_delta", "partial_json": '{"num'},
        }
        yield {
            "type": "content_block_delta",
            "delta": {"type": "input_json_delta", "partial_json": 'ber": 42}'},
        }
        yield {"type": "content_block_stop"}
        yield {
            "type": "message_delta",
            "delta": {"stop_reason": "tool_use"},
        }

    return generate()


@pytest.fixture
def mock_sse_response():
    """Mock SSE response."""

    class MockResponse:
        async def aiter_bytes(self):
            # Simulate SSE format
            yield b'data: {"type": "message_start", "message": {"model": "claude-3-5-sonnet"}}\n\n'
            yield b'data: {"type": "content_block_start", "content_block": {"type": "text"}}\n\n'
            yield b'data: {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "Hello "}}\n\n'
            yield b'data: {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "world!"}}\n\n'
            yield b'data: {"type": "message_delta", "delta": {"stop_reason": "end_turn"}}\n\n'

    return MockResponse()


@pytest.mark.unit
class TestStreamEvent:
    """Test StreamEvent data class."""

    def test_stream_event_creation(self):
        """Test creating a stream event."""
        event = StreamEvent(type="test", data={"key": "value"})
        assert event.type == "test"
        assert event.data == {"key": "value"}

    def test_stream_event_default_data(self):
        """Test stream event with default data."""
        event = StreamEvent(type="test")
        assert event.data == {}


@pytest.mark.unit
class TestClaudeSDKEventSource:
    """Test ClaudeSDKEventSource."""

    @pytest.mark.asyncio
    async def test_claude_sdk_events(self, mock_claude_thinking_response):
        """Test converting Claude SDK chunks to events."""
        source = ClaudeSDKEventSource(mock_claude_thinking_response)
        events = []

        async for event in source.get_events():
            events.append(event)

        assert len(events) == 8
        assert events[0].type == "start"
        assert events[1].type == "content_block_start"
        assert events[1].data["block"]["type"] == "thinking"
        assert events[2].type == "content_block_delta"
        assert events[2].data["delta"]["thinking"] == "Let me think..."


@pytest.mark.unit
class TestSSEEventSource:
    """Test SSEEventSource."""

    @pytest.mark.asyncio
    async def test_sse_parsing(self, mock_sse_response):
        """Test parsing SSE stream."""
        source = SSEEventSource(mock_sse_response)
        events = []

        async for event in source.get_events():
            events.append(event)

        assert len(events) == 5
        assert events[0].type == "start"
        assert events[2].type == "content_block_delta"
        assert events[2].data["delta"]["text"] == "Hello "


@pytest.mark.unit
class TestStreamingConfig:
    """Test StreamingConfig."""

    def test_default_config(self):
        """Test default streaming configuration."""
        config = StreamingConfig()
        assert config.enable_text_chunking is True
        assert config.enable_tool_calls is True
        assert config.enable_usage_info is True
        assert config.chunk_delay_ms == 10.0
        assert config.chunk_size_words == 3

    def test_custom_config(self):
        """Test custom streaming configuration."""
        config = StreamingConfig(
            enable_text_chunking=False,
            enable_tool_calls=False,
            chunk_delay_ms=5.0,
        )
        assert config.enable_text_chunking is False
        assert config.enable_tool_calls is False
        assert config.chunk_delay_ms == 5.0


@pytest.mark.unit
class TestOpenAIStreamTransformer:
    """Test OpenAIStreamTransformer."""

    @pytest.mark.asyncio
    async def test_thinking_block_transformation(self, mock_claude_thinking_response):
        """Test transforming thinking blocks."""
        transformer = OpenAIStreamTransformer.from_claude_sdk(
            mock_claude_thinking_response,
            message_id="test_123",
            model="gpt-4",
        )

        chunks: list[str] = []
        async for chunk in transformer.transform():
            chunks.append(chunk)

        # Verify thinking marker
        assert any("[Thinking]" in chunk for chunk in chunks)
        assert any("---" in chunk for chunk in chunks)
        assert any("Let me think..." in chunk for chunk in chunks)
        assert any("Here's the answer." in chunk for chunk in chunks)

        # Verify structure
        assert chunks[0].startswith(
            'data: {"id":"test_123"'
        )  # No space after colon in JSON
        assert chunks[-1] == "data: [DONE]\n\n"

    @pytest.mark.asyncio
    async def test_tool_call_transformation(self, mock_claude_tool_response):
        """Test transforming tool calls."""
        config = StreamingConfig(enable_tool_calls=True)
        transformer = OpenAIStreamTransformer.from_claude_sdk(
            mock_claude_tool_response,
            message_id="test_456",
            model="gpt-4",
            config=config,
        )

        chunks: list[str] = []
        async for chunk in transformer.transform():
            chunks.append(chunk)

        # Find tool call chunks
        tool_chunks = [c for c in chunks if "tool_calls" in c]
        assert len(tool_chunks) > 0

        # Verify tool call content
        assert any("calculator" in chunk for chunk in tool_chunks)

        # Check if we have the expected tool call structure
        # The tool calls should have the function name and arguments
        tool_call_found = False
        for chunk in tool_chunks:
            if "calculator" in chunk:
                tool_call_found = True
                break
        assert tool_call_found

    @pytest.mark.asyncio
    async def test_usage_information(self, mock_claude_thinking_response):
        """Test including usage information."""
        config = StreamingConfig(enable_usage_info=True)
        transformer = OpenAIStreamTransformer.from_claude_sdk(
            mock_claude_thinking_response,
            config=config,
        )

        chunks: list[str] = []
        async for chunk in transformer.transform():
            chunks.append(chunk)

        # Find chunk with usage
        usage_chunks = [c for c in chunks if "usage" in c and "prompt_tokens" in c]
        assert len(usage_chunks) > 0

        # Parse and verify usage
        for chunk in usage_chunks:
            if chunk.startswith("data: "):
                data = json.loads(chunk[6:])
                if "usage" in data:
                    assert data["usage"]["prompt_tokens"] == 10
                    assert data["usage"]["completion_tokens"] == 20
                    assert data["usage"]["total_tokens"] == 30

    @pytest.mark.asyncio
    async def test_text_chunking(self):
        """Test text chunking feature."""

        async def generate():
            yield {"type": "message_start"}
            yield {"type": "content_block_start", "content_block": {"type": "text"}}
            yield {
                "type": "content_block_delta",
                "delta": {
                    "type": "text_delta",
                    "text": "This is a long sentence that should be split into chunks.",
                },
            }
            yield {"type": "message_delta", "delta": {"stop_reason": "end_turn"}}

        config = StreamingConfig(
            enable_text_chunking=True,
            chunk_size_words=3,
            chunk_delay_ms=0,  # No delay for testing
        )

        transformer = OpenAIStreamTransformer.from_claude_sdk(
            generate(),
            config=config,
        )

        text_chunks = []
        async for chunk in transformer.transform():
            if '"content":' in chunk and "[DONE]" not in chunk:
                # Extract content
                data = json.loads(chunk[6:])
                content = data.get("choices", [{}])[0].get("delta", {}).get("content")
                if content:
                    text_chunks.append(content)

        # Should be split into multiple chunks
        assert len(text_chunks) > 1
        # Reconstruct should match original
        assert (
            "".join(text_chunks)
            == "This is a long sentence that should be split into chunks."
        )

    @pytest.mark.asyncio
    async def test_sse_stream_transformation(self, mock_sse_response):
        """Test transforming SSE streams."""
        transformer = OpenAIStreamTransformer.from_sse_stream(
            mock_sse_response,
            message_id="sse_test",
            model="gpt-4",
        )

        chunks: list[str] = []
        async for chunk in transformer.transform():
            chunks.append(chunk)

        # Verify content - text is split between chunks
        all_content = "".join(chunks)
        assert "Hello " in all_content
        assert "world!" in all_content
        assert chunks[0].startswith('data: {"id":"sse_test"')  # No space after colon
        assert chunks[-1] == "data: [DONE]\n\n"

    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling in transformation."""

        async def error_generator():
            yield {"type": "message_start"}
            raise ValueError("Test error")

        transformer = OpenAIStreamTransformer.from_claude_sdk(error_generator())

        chunks: list[str] = []
        async for chunk in transformer.transform():
            chunks.append(chunk)

        # Should have error chunk
        error_chunks = [c for c in chunks if "error" in c and "Test error" in c]
        assert len(error_chunks) > 0
        assert chunks[-1] == "data: [DONE]\n\n"

    @pytest.mark.asyncio
    async def test_cancellation_handling(self):
        """Test handling of cancelled streams."""

        async def cancellable_generator():
            yield {"type": "message_start"}
            await asyncio.sleep(0.1)
            yield {"type": "content_block_start", "content_block": {"type": "text"}}
            await asyncio.sleep(10)  # Will be cancelled
            yield {
                "type": "content_block_delta",
                "delta": {"type": "text_delta", "text": "Never reached"},
            }

        transformer = OpenAIStreamTransformer.from_claude_sdk(cancellable_generator())

        chunks: list[str] = []
        task = asyncio.create_task(self._collect_chunks(transformer, chunks))

        # Cancel after short delay
        await asyncio.sleep(0.2)
        task.cancel()

        with contextlib.suppress(asyncio.CancelledError):
            await task

        # Should have cancellation chunk
        assert any("cancelled" in chunk for chunk in chunks)
        assert chunks[-1] == "data: [DONE]\n\n"

    async def _collect_chunks(self, transformer, chunks):
        """Helper to collect chunks."""
        async for chunk in transformer.transform():
            chunks.append(chunk)

    def test_text_splitting(self):
        """Test the text splitting logic."""
        transformer = OpenAIStreamTransformer.from_claude_sdk(
            self._empty_generator(),
            config=StreamingConfig(chunk_size_words=2),
        )

        # Test various text inputs
        assert transformer._split_text_for_streaming("") == [""]
        assert transformer._split_text_for_streaming("Hi") == ["Hi"]
        assert transformer._split_text_for_streaming("Hello world") == ["Hello world"]
        # Note: The splitting includes spaces with the following word
        result = transformer._split_text_for_streaming("One two three four")
        assert len(result) == 2
        assert "One two" in result[0]
        assert "three four" in result[1]
        # Newline handling
        newline_result = transformer._split_text_for_streaming("Word1  word2\nword3")
        assert len(newline_result) == 2
        assert "Word1  word2" in newline_result[0]
        assert "word3" in newline_result[1]

    async def _empty_generator(self):
        """Empty generator for testing."""
        yield {"type": "message_stop"}
