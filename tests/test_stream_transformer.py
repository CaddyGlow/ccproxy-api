"""Tests for the unified OpenAI stream transformer."""

import asyncio
import contextlib
import json
import time
from collections.abc import AsyncIterator

# Compatibility classes and functions to replace the old stream_transformer functionality
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, Mock

import pytest

from ccproxy.adapters.anthropic.streaming import (
    AnthropicStreamingFormatter,
    AnthropicStreamProcessor,
)
from ccproxy.adapters.openai.streaming import (
    OpenAISSEFormatter,
    OpenAIStreamProcessor,
)


@dataclass
class StreamEvent:
    """Compatibility class for StreamEvent."""

    type: str
    data: dict[str, Any] = None

    def __post_init__(self):
        if self.data is None:
            self.data = {}


@dataclass
class StreamingConfig:
    """Compatibility class for StreamingConfig."""

    enable_text_chunking: bool = True
    enable_tool_calls: bool = True
    enable_usage_info: bool = True
    chunk_delay_ms: float = 10.0
    chunk_size_words: int = 3


class ClaudeSDKEventSource:
    """Compatibility class for ClaudeSDKEventSource."""

    def __init__(self, stream: AsyncIterator[dict[str, Any]]):
        self.stream = stream

    async def get_events(self) -> AsyncIterator[StreamEvent]:
        """Convert Claude SDK chunks to events."""
        async for chunk in self.stream:
            chunk_type = chunk.get("type", "")

            if chunk_type == "message_start":
                yield StreamEvent("start", chunk)
            elif chunk_type == "content_block_start":
                yield StreamEvent(
                    "content_block_start", {"block": chunk.get("content_block", {})}
                )
            elif chunk_type == "content_block_delta":
                yield StreamEvent(
                    "content_block_delta", {"delta": chunk.get("delta", {})}
                )
            elif chunk_type == "content_block_stop":
                yield StreamEvent("content_block_stop", chunk)
            elif chunk_type == "message_delta":
                yield StreamEvent(
                    "message_delta",
                    {"delta": chunk.get("delta", {}), "usage": chunk.get("usage")},
                )
            elif chunk_type == "message_stop":
                yield StreamEvent("stop", chunk)
            else:
                yield StreamEvent(chunk_type, chunk)


class SSEEventSource:
    """Compatibility class for SSEEventSource."""

    def __init__(self, response):
        self.response = response

    async def get_events(self) -> AsyncIterator[StreamEvent]:
        """Parse SSE stream and convert to events."""
        async for data in self.response.aiter_bytes():
            line = data.decode("utf-8").strip()
            if line.startswith("data: "):
                json_str = line[6:]  # Remove 'data: ' prefix
                try:
                    import json

                    chunk = json.loads(json_str)
                    chunk_type = chunk.get("type", "")

                    if chunk_type == "message_start":
                        yield StreamEvent("start", chunk)
                    elif chunk_type == "content_block_start":
                        yield StreamEvent(
                            "content_block_start",
                            {"block": chunk.get("content_block", {})},
                        )
                    elif chunk_type == "content_block_delta":
                        yield StreamEvent(
                            "content_block_delta", {"delta": chunk.get("delta", {})}
                        )
                    elif chunk_type == "content_block_stop":
                        yield StreamEvent("content_block_stop", chunk)
                    elif chunk_type == "message_delta":
                        yield StreamEvent(
                            "message_delta",
                            {
                                "delta": chunk.get("delta", {}),
                                "usage": chunk.get("usage"),
                            },
                        )
                    elif chunk_type == "message_stop":
                        yield StreamEvent("stop", chunk)
                    else:
                        yield StreamEvent(chunk_type, chunk)
                except Exception:
                    continue


class OpenAIStreamTransformer:
    """Compatibility class for OpenAIStreamTransformer."""

    def __init__(
        self,
        event_source,
        message_id: str = None,
        model: str = "gpt-4",
        config: StreamingConfig = None,
    ):
        self.event_source = event_source
        self.message_id = message_id or f"chatcmpl-{int(time.time())}"
        self.model = model
        self.config = config or StreamingConfig()
        self.formatter = OpenAISSEFormatter()
        self.created = int(time.time())

    @classmethod
    def from_claude_sdk(
        cls,
        stream: AsyncIterator[dict[str, Any]],
        message_id: str = None,
        model: str = "gpt-4",
        config: StreamingConfig = None,
    ):
        """Create transformer from Claude SDK stream."""
        event_source = ClaudeSDKEventSource(stream)
        return cls(event_source, message_id, model, config)

    @classmethod
    def from_sse_stream(
        cls,
        response,
        message_id: str = None,
        model: str = "gpt-4",
        config: StreamingConfig = None,
    ):
        """Create transformer from SSE stream."""
        event_source = SSEEventSource(response)
        return cls(event_source, message_id, model, config)

    async def transform(self) -> AsyncIterator[str]:
        """Transform events to OpenAI streaming format."""
        # Generate initial chunk
        yield self.formatter.format_first_chunk(
            self.message_id, self.model, self.created
        )

        usage_data = None
        tool_calls = {}
        current_tool_index = 0

        try:
            async for event in self.event_source.get_events():
                if event.type == "content_block_delta":
                    delta = event.data.get("delta", {})

                    if delta.get("type") == "text_delta":
                        text = delta.get("text", "")
                        if text and self.config.enable_text_chunking:
                            # Split text into chunks
                            chunks = self._split_text_for_streaming(text)
                            for chunk_text in chunks:
                                if chunk_text.strip():
                                    yield self.formatter.format_content_chunk(
                                        self.message_id,
                                        self.model,
                                        self.created,
                                        chunk_text,
                                    )
                                    if self.config.chunk_delay_ms > 0:
                                        await asyncio.sleep(
                                            self.config.chunk_delay_ms / 1000
                                        )
                        elif text:
                            yield self.formatter.format_content_chunk(
                                self.message_id, self.model, self.created, text
                            )

                    elif delta.get("type") == "thinking_delta":
                        # Handle thinking blocks
                        thinking = delta.get("thinking", "")
                        if thinking:
                            thinking_content = f"[Thinking]\n{thinking}\n---\n"
                            yield self.formatter.format_content_chunk(
                                self.message_id,
                                self.model,
                                self.created,
                                thinking_content,
                            )

                    elif (
                        delta.get("type") == "input_json_delta"
                        and self.config.enable_tool_calls
                    ):
                        # Handle tool call arguments
                        partial_json = delta.get("partial_json", "")
                        if partial_json and tool_calls:
                            last_tool_id = list(tool_calls.keys())[-1]
                            yield self.formatter.format_tool_call_chunk(
                                self.message_id,
                                self.model,
                                self.created,
                                last_tool_id,
                                function_arguments=partial_json,
                                tool_call_index=current_tool_index - 1,
                            )

                elif event.type == "content_block_start":
                    block = event.data.get("block", {})
                    if (
                        block.get("type") == "tool_use"
                        and self.config.enable_tool_calls
                    ):
                        tool_id = block.get("id")
                        function_name = block.get("name")
                        if tool_id and function_name:
                            tool_calls[tool_id] = {
                                "name": function_name,
                                "arguments": "",
                            }
                            yield self.formatter.format_tool_call_chunk(
                                self.message_id,
                                self.model,
                                self.created,
                                tool_id,
                                function_name,
                                tool_call_index=current_tool_index,
                            )
                            current_tool_index += 1
                    elif block.get("type") == "thinking":
                        yield self.formatter.format_content_chunk(
                            self.message_id, self.model, self.created, "[Thinking]\n"
                        )

                elif event.type == "message_delta":
                    delta = event.data.get("delta", {})
                    stop_reason = delta.get("stop_reason")

                    # Extract usage data if present and enabled
                    if self.config.enable_usage_info and event.data.get("usage"):
                        usage_info = event.data["usage"]
                        usage_data = {
                            "prompt_tokens": usage_info.get("input_tokens", 0),
                            "completion_tokens": usage_info.get("output_tokens", 0),
                            "total_tokens": usage_info.get("input_tokens", 0)
                            + usage_info.get("output_tokens", 0),
                        }

                    if stop_reason:
                        # Map Claude stop reasons to OpenAI
                        openai_stop_reason = {
                            "end_turn": "stop",
                            "max_tokens": "length",
                            "stop_sequence": "stop",
                            "tool_use": "tool_calls",
                        }.get(stop_reason, "stop")

                        final_usage = (
                            usage_data if self.config.enable_usage_info else None
                        )
                        yield self.formatter.format_final_chunk(
                            self.message_id,
                            self.model,
                            self.created,
                            openai_stop_reason,
                            usage=final_usage,
                        )

        except asyncio.CancelledError:
            yield self.formatter.format_final_chunk(
                self.message_id, self.model, self.created, "cancelled"
            )
            yield self.formatter.format_done()
            raise
        except Exception as e:
            yield self.formatter.format_error_chunk(
                self.message_id, self.model, self.created, "stream_error", str(e)
            )

        yield self.formatter.format_done()

    def _split_text_for_streaming(self, text: str) -> list[str]:
        """Split text into chunks for streaming."""
        if not text or not self.config.enable_text_chunking:
            return [text]

        # Split by whitespace while preserving original spacing
        import re

        # Split on whitespace but keep the whitespace with the following word
        parts = re.split(r"(\s+)", text)
        words_with_spaces = []

        i = 0
        while i < len(parts):
            if i + 1 < len(parts) and parts[i + 1].isspace():
                # Combine word with following whitespace
                words_with_spaces.append(parts[i] + parts[i + 1])
                i += 2
            else:
                words_with_spaces.append(parts[i])
                i += 1

        # Filter out empty parts
        words_with_spaces = [w for w in words_with_spaces if w.strip()]

        if len(words_with_spaces) <= self.config.chunk_size_words:
            return [text]

        chunks = []
        current_chunk = []

        for word_with_space in words_with_spaces:
            current_chunk.append(word_with_space)
            if len(current_chunk) >= self.config.chunk_size_words:
                chunks.append("".join(current_chunk))
                current_chunk = []

        if current_chunk:
            chunks.append("".join(current_chunk))

        return chunks


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
