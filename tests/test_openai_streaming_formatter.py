"""
Tests for OpenAI streaming functionality.

This module contains comprehensive tests for the OpenAI streaming service
including the formatter class and streaming functions.
"""

import asyncio
import json
import time
from unittest.mock import AsyncMock, Mock, patch

import pytest

from ccproxy.adapters.openai.streaming import (
    OpenAISSEFormatter,
    OpenAIStreamProcessor,
)


# Compatibility functions to replace the old streaming functions
async def stream_claude_response_openai(
    stream, message_id: str, model: str, created: int = None
):
    """Compatibility wrapper for the old stream_claude_response_openai function."""
    if created is None:
        created = int(time.time())

    formatter = OpenAISSEFormatter()

    # Generate initial chunk with role
    yield formatter.format_first_chunk(message_id, model, created)

    # Process the stream
    tool_calls = {}
    current_tool_index = 0

    try:
        async for chunk in stream:
            chunk_type = chunk.get("type", "")

            if chunk_type == "content_block_delta":
                delta = chunk.get("delta", {})
                if delta.get("type") == "text_delta":
                    text = delta.get("text", "")
                    if text:
                        # Split text into chunks for better streaming
                        words = text.split()
                        chunk_size = 10  # Split into chunks of 10 words
                        if len(words) > chunk_size:
                            for i in range(0, len(words), chunk_size):
                                chunk_text = " ".join(words[i : i + chunk_size])
                                yield formatter.format_content_chunk(
                                    message_id, model, created, chunk_text
                                )
                        else:
                            yield formatter.format_content_chunk(
                                message_id, model, created, text
                            )
                elif delta.get("type") == "input_json_delta":
                    # Handle tool call arguments
                    partial_json = delta.get("partial_json", "")
                    if partial_json and tool_calls:
                        # Add to the last tool call's arguments
                        last_tool_id = list(tool_calls.keys())[-1]
                        yield formatter.format_tool_call_chunk(
                            message_id,
                            model,
                            created,
                            last_tool_id,
                            function_arguments=partial_json,
                            tool_call_index=current_tool_index - 1,
                        )

            elif chunk_type == "content_block_start":
                content_block = chunk.get("content_block", {})
                if content_block.get("type") == "tool_use":
                    tool_id = content_block.get("id")
                    function_name = content_block.get("name")
                    if tool_id and function_name:
                        tool_calls[tool_id] = {"name": function_name, "arguments": ""}
                        yield formatter.format_tool_call_chunk(
                            message_id,
                            model,
                            created,
                            tool_id,
                            function_name,
                            tool_call_index=current_tool_index,
                        )
                        current_tool_index += 1

            elif chunk_type == "message_delta":
                delta = chunk.get("delta", {})
                stop_reason = delta.get("stop_reason")
                if stop_reason:
                    # Map Claude stop reasons to OpenAI
                    openai_stop_reason = {
                        "end_turn": "stop",
                        "max_tokens": "length",
                        "stop_sequence": "stop",
                        "tool_use": "tool_calls",
                    }.get(stop_reason, "stop")

                    yield formatter.format_final_chunk(
                        message_id, model, created, openai_stop_reason
                    )

    except asyncio.CancelledError:
        yield formatter.format_final_chunk(message_id, model, created, "cancelled")
        yield formatter.format_done()
        raise
    except Exception as e:
        yield formatter.format_error_chunk(
            message_id, model, created, "stream_error", str(e)
        )

    yield formatter.format_done()


async def stream_claude_response_openai_simple(
    stream, message_id: str, model: str, created: int = None
):
    """Simplified compatibility wrapper for streaming Claude responses to OpenAI format."""
    if created is None:
        created = int(time.time())

    formatter = OpenAISSEFormatter()

    # Generate initial chunk with role
    yield formatter.format_first_chunk(message_id, model, created)

    try:
        async for chunk in stream:
            chunk_type = chunk.get("type", "")

            if chunk_type == "content_block_delta":
                delta = chunk.get("delta", {})
                if delta.get("type") == "text_delta":
                    text = delta.get("text", "")
                    if text:
                        yield formatter.format_content_chunk(
                            message_id, model, created, text
                        )

            elif chunk_type == "message_delta":
                delta = chunk.get("delta", {})
                stop_reason = delta.get("stop_reason")
                if stop_reason:
                    # Map Claude stop reasons to OpenAI (simplified)
                    openai_stop_reason = {
                        "end_turn": "stop",
                        "max_tokens": "length",
                        "stop_sequence": "stop",
                        "tool_use": "stop",
                    }.get(stop_reason, "stop")

                    yield formatter.format_final_chunk(
                        message_id, model, created, openai_stop_reason
                    )

    except asyncio.CancelledError:
        yield formatter.format_final_chunk(message_id, model, created, "cancelled")
        yield formatter.format_done()
        raise
    except Exception as e:
        yield formatter.format_error_chunk(
            message_id, model, created, "stream_error", str(e)
        )

    yield formatter.format_done()


class TestOpenAISSEFormatter:
    """Test the OpenAISSEFormatter class."""

    def test_format_data_event(self):
        """Test format_data_event method."""
        data = {"type": "test", "message": "hello"}
        result = OpenAISSEFormatter.format_data_event(data)

        expected = 'data: {"type":"test","message":"hello"}\n\n'
        assert result == expected

    def test_format_data_event_empty(self):
        """Test format_data_event with empty data."""
        result = OpenAISSEFormatter.format_data_event({})
        assert result == "data: {}\n\n"

    def test_format_data_event_complex(self):
        """Test format_data_event with complex nested data."""
        data = {
            "type": "test",
            "nested": {"key": "value", "number": 42},
            "array": [1, 2, 3],
        }
        result = OpenAISSEFormatter.format_data_event(data)

        # Parse the JSON to verify it's valid
        json_str = result.replace("data: ", "").replace("\n\n", "")
        parsed = json.loads(json_str)
        assert parsed == data

    def test_format_first_chunk(self):
        """Test format_first_chunk method."""
        message_id = "msg_123"
        model = "claude-3-5-sonnet-20241022"
        created = 1234567890

        result = OpenAISSEFormatter.format_first_chunk(message_id, model, created)

        assert "data: " in result
        assert result.endswith("\n\n")

        # Parse and verify structure
        json_str = result.replace("data: ", "").replace("\n\n", "")
        data = json.loads(json_str)

        assert data["id"] == message_id
        assert data["object"] == "chat.completion.chunk"
        assert data["created"] == created
        assert data["model"] == model
        assert data["choices"][0]["delta"]["role"] == "assistant"
        assert data["choices"][0]["finish_reason"] is None

    def test_format_first_chunk_custom_role(self):
        """Test format_first_chunk with custom role."""
        result = OpenAISSEFormatter.format_first_chunk(
            "msg_123", "claude-3-5-sonnet-20241022", 1234567890, "user"
        )

        json_str = result.replace("data: ", "").replace("\n\n", "")
        data = json.loads(json_str)

        assert data["choices"][0]["delta"]["role"] == "user"

    def test_format_content_chunk(self):
        """Test format_content_chunk method."""
        message_id = "msg_123"
        model = "claude-3-5-sonnet-20241022"
        created = 1234567890
        content = "Hello world"

        result = OpenAISSEFormatter.format_content_chunk(
            message_id, model, created, content
        )

        assert "data: " in result
        assert result.endswith("\n\n")

        # Parse and verify structure
        json_str = result.replace("data: ", "").replace("\n\n", "")
        data = json.loads(json_str)

        assert data["id"] == message_id
        assert data["model"] == model
        assert data["created"] == created
        assert data["choices"][0]["delta"]["content"] == content
        assert data["choices"][0]["index"] == 0

    def test_format_content_chunk_custom_index(self):
        """Test format_content_chunk with custom choice index."""
        result = OpenAISSEFormatter.format_content_chunk(
            "msg_123", "claude-3-5-sonnet-20241022", 1234567890, "test", 2
        )

        json_str = result.replace("data: ", "").replace("\n\n", "")
        data = json.loads(json_str)

        assert data["choices"][0]["index"] == 2

    def test_format_content_chunk_empty_content(self):
        """Test format_content_chunk with empty content."""
        result = OpenAISSEFormatter.format_content_chunk(
            "msg_123", "claude-3-5-sonnet-20241022", 1234567890, ""
        )

        json_str = result.replace("data: ", "").replace("\n\n", "")
        data = json.loads(json_str)

        assert data["choices"][0]["delta"]["content"] == ""

    def test_format_tool_call_chunk_name_only(self):
        """Test format_tool_call_chunk with function name only."""
        message_id = "msg_123"
        model = "claude-3-5-sonnet-20241022"
        created = 1234567890
        tool_call_id = "tool_123"
        function_name = "get_weather"

        result = OpenAISSEFormatter.format_tool_call_chunk(
            message_id, model, created, tool_call_id, function_name
        )

        json_str = result.replace("data: ", "").replace("\n\n", "")
        data = json.loads(json_str)

        tool_call = data["choices"][0]["delta"]["tool_calls"][0]
        assert tool_call["id"] == tool_call_id
        assert tool_call["type"] == "function"
        assert tool_call["function"]["name"] == function_name
        assert "arguments" not in tool_call["function"]

    def test_format_tool_call_chunk_arguments_only(self):
        """Test format_tool_call_chunk with function arguments only."""
        result = OpenAISSEFormatter.format_tool_call_chunk(
            "msg_123",
            "claude-3-5-sonnet-20241022",
            1234567890,
            "tool_123",
            function_arguments='{"location": "NYC"}',
        )

        json_str = result.replace("data: ", "").replace("\n\n", "")
        data = json.loads(json_str)

        tool_call = data["choices"][0]["delta"]["tool_calls"][0]
        assert tool_call["function"]["arguments"] == '{"location": "NYC"}'
        assert "name" not in tool_call["function"]

    def test_format_tool_call_chunk_complete(self):
        """Test format_tool_call_chunk with both name and arguments."""
        result = OpenAISSEFormatter.format_tool_call_chunk(
            "msg_123",
            "claude-3-5-sonnet-20241022",
            1234567890,
            "tool_123",
            "get_weather",
            '{"location": "NYC"}',
            1,
            0,
        )

        json_str = result.replace("data: ", "").replace("\n\n", "")
        data = json.loads(json_str)

        tool_call = data["choices"][0]["delta"]["tool_calls"][0]
        assert tool_call["index"] == 1
        assert tool_call["function"]["name"] == "get_weather"
        assert tool_call["function"]["arguments"] == '{"location": "NYC"}'

    def test_format_final_chunk(self):
        """Test format_final_chunk method."""
        message_id = "msg_123"
        model = "claude-3-5-sonnet-20241022"
        created = 1234567890

        result = OpenAISSEFormatter.format_final_chunk(message_id, model, created)

        json_str = result.replace("data: ", "").replace("\n\n", "")
        data = json.loads(json_str)

        assert data["id"] == message_id
        assert data["model"] == model
        assert data["created"] == created
        assert data["choices"][0]["finish_reason"] == "stop"
        assert data["choices"][0]["delta"] == {}

    def test_format_final_chunk_custom_reason(self):
        """Test format_final_chunk with custom finish reason."""
        result = OpenAISSEFormatter.format_final_chunk(
            "msg_123", "claude-3-5-sonnet-20241022", 1234567890, "length", 1
        )

        json_str = result.replace("data: ", "").replace("\n\n", "")
        data = json.loads(json_str)

        assert data["choices"][0]["finish_reason"] == "length"
        assert data["choices"][0]["index"] == 1

    def test_format_error_chunk(self):
        """Test format_error_chunk method."""
        message_id = "msg_123"
        model = "claude-3-5-sonnet-20241022"
        created = 1234567890
        error_type = "internal_server_error"
        error_message = "Something went wrong"

        result = OpenAISSEFormatter.format_error_chunk(
            message_id, model, created, error_type, error_message
        )

        json_str = result.replace("data: ", "").replace("\n\n", "")
        data = json.loads(json_str)

        assert data["id"] == message_id
        assert data["model"] == model
        assert data["created"] == created
        assert data["choices"][0]["finish_reason"] == "error"
        assert data["error"]["type"] == error_type
        assert data["error"]["message"] == error_message

    def test_format_done(self):
        """Test format_done method."""
        result = OpenAISSEFormatter.format_done()
        assert result == "data: [DONE]\n\n"


class TestStreamClaudeResponseOpenAI:
    """Test the stream_claude_response_openai function."""

    @pytest.mark.asyncio
    async def test_successful_streaming(self, mock_claude_streaming_response):
        """Test successful streaming conversion."""
        chunks = []
        async for chunk in stream_claude_response_openai(
            mock_claude_streaming_response,
            "msg_123",
            "claude-3-5-sonnet-20241022",
            1234567890,
        ):
            chunks.append(chunk)

        # Should have multiple chunks and end with DONE
        assert len(chunks) > 1
        assert chunks[-1] == "data: [DONE]\n\n"

        # Should have first chunk with role
        first_chunk_data = json.loads(
            chunks[0].replace("data: ", "").replace("\n\n", "")
        )
        assert first_chunk_data["choices"][0]["delta"]["role"] == "assistant"

        # Should have content chunks
        content_chunks = [c for c in chunks if '"content":' in c]
        assert len(content_chunks) > 0

        # Should have final chunk with finish_reason
        final_chunk = [c for c in chunks if '"finish_reason":"stop"' in c]
        assert len(final_chunk) == 1

    @pytest.mark.asyncio
    async def test_streaming_with_default_timestamp(
        self, mock_claude_streaming_response
    ):
        """Test streaming with default timestamp."""
        with patch("time.time", return_value=1234567890):
            chunks = []
            async for chunk in stream_claude_response_openai(
                mock_claude_streaming_response, "msg_123", "claude-3-5-sonnet-20241022"
            ):
                chunks.append(chunk)

            # Check that timestamp is used
            first_chunk_data = json.loads(
                chunks[0].replace("data: ", "").replace("\n\n", "")
            )
            assert first_chunk_data["created"] == 1234567890

    @pytest.mark.asyncio
    async def test_streaming_with_tool_calls(self, mock_claude_tool_streaming_response):
        """Test streaming with tool calls."""
        chunks = []
        async for chunk in stream_claude_response_openai(
            mock_claude_tool_streaming_response,
            "msg_123",
            "claude-3-5-sonnet-20241022",
            1234567890,
        ):
            chunks.append(chunk)

        # Should have tool call chunks
        tool_chunks = [c for c in chunks if '"tool_calls"' in c]
        assert len(tool_chunks) > 0

        # Should have finish_reason as tool_calls
        final_chunk = [c for c in chunks if '"finish_reason":"tool_calls"' in c]
        assert len(final_chunk) == 1

        # Should end with DONE
        assert chunks[-1] == "data: [DONE]\n\n"

    @pytest.mark.asyncio
    async def test_streaming_with_error(self, mock_claude_error_streaming_response):
        """Test streaming with error."""
        chunks = []
        async for chunk in stream_claude_response_openai(
            mock_claude_error_streaming_response,
            "msg_123",
            "claude-3-5-sonnet-20241022",
            1234567890,
        ):
            chunks.append(chunk)

        # Should have error chunk
        error_chunks = [c for c in chunks if '"error"' in c]
        assert len(error_chunks) > 0

        # Should still end with DONE
        assert chunks[-1] == "data: [DONE]\n\n"

    @pytest.mark.asyncio
    async def test_streaming_with_cancellation(
        self, mock_claude_cancelled_streaming_response
    ):
        """Test streaming with cancellation."""
        chunks = []
        with pytest.raises(asyncio.CancelledError):
            async for chunk in stream_claude_response_openai(
                mock_claude_cancelled_streaming_response,
                "msg_123",
                "claude-3-5-sonnet-20241022",
                1234567890,
            ):
                chunks.append(chunk)

        # Should have some chunks before cancellation
        assert len(chunks) > 0

        # Should have final chunk with cancelled reason
        cancelled_chunks = [c for c in chunks if '"finish_reason":"cancelled"' in c]
        assert len(cancelled_chunks) == 1

        # Should end with DONE
        assert chunks[-1] == "data: [DONE]\n\n"

    @pytest.mark.asyncio
    async def test_streaming_empty_response(self, mock_claude_empty_streaming_response):
        """Test streaming with empty response."""
        chunks = []
        async for chunk in stream_claude_response_openai(
            mock_claude_empty_streaming_response,
            "msg_123",
            "claude-3-5-sonnet-20241022",
            1234567890,
        ):
            chunks.append(chunk)

        # Should have first chunk, final chunk, and DONE
        assert len(chunks) >= 3

        # Should have final chunk with stop reason
        final_chunk = [c for c in chunks if '"finish_reason":"stop"' in c]
        assert len(final_chunk) >= 1

        # Should end with DONE
        assert chunks[-1] == "data: [DONE]\n\n"

    @pytest.mark.asyncio
    async def test_streaming_stop_reason_mapping(self):
        """Test stop reason mapping from Claude to OpenAI."""

        async def mock_response():
            yield {
                "type": "message_start",
                "message": {"id": "msg_123", "role": "assistant"},
            }
            yield {"type": "message_delta", "delta": {"stop_reason": "max_tokens"}}
            yield {"type": "message_stop"}

        chunks = []
        async for chunk in stream_claude_response_openai(
            mock_response(), "msg_123", "claude-3-5-sonnet-20241022", 1234567890
        ):
            chunks.append(chunk)

        # Should map max_tokens to length
        length_chunks = [c for c in chunks if '"finish_reason":"length"' in c]
        assert len(length_chunks) == 1

    @pytest.mark.asyncio
    async def test_streaming_large_text_splitting(self):
        """Test streaming with large text that gets split."""

        async def mock_response():
            yield {
                "type": "message_start",
                "message": {"id": "msg_123", "role": "assistant"},
            }
            yield {
                "type": "content_block_start",
                "content_block": {"type": "text", "text": ""},
            }
            yield {
                "type": "content_block_delta",
                "delta": {
                    "type": "text_delta",
                    "text": "This is a very long text that should be split into multiple chunks for better streaming experience",
                },
            }
            yield {"type": "content_block_stop"}
            yield {"type": "message_delta", "delta": {"stop_reason": "end_turn"}}
            yield {"type": "message_stop"}

        chunks = []
        async for chunk in stream_claude_response_openai(
            mock_response(), "msg_123", "claude-3-5-sonnet-20241022", 1234567890
        ):
            chunks.append(chunk)

        # Should have multiple content chunks due to splitting
        content_chunks = [c for c in chunks if '"content":' in c]
        assert len(content_chunks) > 1

    @pytest.mark.asyncio
    async def test_streaming_unknown_chunk_type(self):
        """Test streaming with unknown chunk type."""

        async def mock_response():
            yield {
                "type": "message_start",
                "message": {"id": "msg_123", "role": "assistant"},
            }
            yield {"type": "unknown_type", "data": "some data"}
            yield {"type": "message_delta", "delta": {"stop_reason": "end_turn"}}
            yield {"type": "message_stop"}

        chunks = []
        async for chunk in stream_claude_response_openai(
            mock_response(), "msg_123", "claude-3-5-sonnet-20241022", 1234567890
        ):
            chunks.append(chunk)

        # Should handle unknown types gracefully
        assert len(chunks) >= 3
        assert chunks[-1] == "data: [DONE]\n\n"

    @pytest.mark.asyncio
    async def test_streaming_tool_call_without_existing_calls(self):
        """Test streaming tool call delta without existing tool calls."""

        async def mock_response():
            yield {
                "type": "message_start",
                "message": {"id": "msg_123", "role": "assistant"},
            }
            # Tool use input delta without first creating a tool call (edge case)
            yield {
                "type": "content_block_delta",
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": '{"test": "value"}',
                },
            }
            yield {"type": "message_delta", "delta": {"stop_reason": "end_turn"}}
            yield {"type": "message_stop"}

        chunks = []
        async for chunk in stream_claude_response_openai(
            mock_response(), "msg_123", "claude-3-5-sonnet-20241022", 1234567890
        ):
            chunks.append(chunk)

        # Should handle gracefully even without existing tool calls
        assert len(chunks) >= 3
        assert chunks[-1] == "data: [DONE]\n\n"

    @pytest.mark.asyncio
    async def test_streaming_tool_call_empty_partial_json(self):
        """Test streaming tool call with empty partial JSON."""

        async def mock_response():
            yield {
                "type": "message_start",
                "message": {"id": "msg_123", "role": "assistant"},
            }
            # Tool use content block start
            yield {
                "type": "content_block_start",
                "content_block": {
                    "type": "tool_use",
                    "id": "tool_123",
                    "name": "get_weather",
                },
            }
            # Tool use input delta with empty partial_json
            yield {
                "type": "content_block_delta",
                "delta": {"type": "input_json_delta", "partial_json": ""},
            }
            yield {"type": "message_delta", "delta": {"stop_reason": "tool_use"}}
            yield {"type": "message_stop"}

        chunks = []
        async for chunk in stream_claude_response_openai(
            mock_response(), "msg_123", "claude-3-5-sonnet-20241022", 1234567890
        ):
            chunks.append(chunk)

        # Should handle empty partial JSON gracefully
        tool_chunks = [c for c in chunks if '"tool_calls"' in c]
        assert len(tool_chunks) >= 1  # At least the initial tool call chunk
        assert chunks[-1] == "data: [DONE]\n\n"


class TestStreamClaudeResponseOpenAISimple:
    """Test the stream_claude_response_openai_simple function."""

    @pytest.mark.asyncio
    async def test_successful_streaming(self, mock_claude_streaming_response):
        """Test successful streaming conversion (simple)."""
        chunks = []
        async for chunk in stream_claude_response_openai_simple(
            mock_claude_streaming_response,
            "msg_123",
            "claude-3-5-sonnet-20241022",
            1234567890,
        ):
            chunks.append(chunk)

        # Should have multiple chunks and end with DONE
        assert len(chunks) > 1
        assert chunks[-1] == "data: [DONE]\n\n"

        # Should have first chunk with role
        first_chunk_data = json.loads(
            chunks[0].replace("data: ", "").replace("\n\n", "")
        )
        assert first_chunk_data["choices"][0]["delta"]["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_streaming_with_default_timestamp(
        self, mock_claude_streaming_response
    ):
        """Test streaming with default timestamp (simple)."""
        with patch("time.time", return_value=1234567890):
            chunks = []
            async for chunk in stream_claude_response_openai_simple(
                mock_claude_streaming_response, "msg_123", "claude-3-5-sonnet-20241022"
            ):
                chunks.append(chunk)

            # Check that timestamp is used
            first_chunk_data = json.loads(
                chunks[0].replace("data: ", "").replace("\n\n", "")
            )
            assert first_chunk_data["created"] == 1234567890

    @pytest.mark.asyncio
    async def test_streaming_with_error(self, mock_claude_error_streaming_response):
        """Test streaming with error (simple)."""
        chunks = []
        async for chunk in stream_claude_response_openai_simple(
            mock_claude_error_streaming_response,
            "msg_123",
            "claude-3-5-sonnet-20241022",
            1234567890,
        ):
            chunks.append(chunk)

        # Should have error chunk
        error_chunks = [c for c in chunks if '"error"' in c]
        assert len(error_chunks) > 0

        # Should still end with DONE
        assert chunks[-1] == "data: [DONE]\n\n"

    @pytest.mark.asyncio
    async def test_streaming_with_cancellation(
        self, mock_claude_cancelled_streaming_response
    ):
        """Test streaming with cancellation (simple)."""
        chunks = []
        with pytest.raises(asyncio.CancelledError):
            async for chunk in stream_claude_response_openai_simple(
                mock_claude_cancelled_streaming_response,
                "msg_123",
                "claude-3-5-sonnet-20241022",
                1234567890,
            ):
                chunks.append(chunk)

        # Should have some chunks before cancellation
        assert len(chunks) > 0

        # Should have final chunk with cancelled reason
        cancelled_chunks = [c for c in chunks if '"finish_reason":"cancelled"' in c]
        assert len(cancelled_chunks) == 1

        # Should end with DONE
        assert chunks[-1] == "data: [DONE]\n\n"

    @pytest.mark.asyncio
    async def test_streaming_empty_response(self, mock_claude_empty_streaming_response):
        """Test streaming with empty response (simple)."""
        chunks = []
        async for chunk in stream_claude_response_openai_simple(
            mock_claude_empty_streaming_response,
            "msg_123",
            "claude-3-5-sonnet-20241022",
            1234567890,
        ):
            chunks.append(chunk)

        # Should have first chunk, final chunk, and DONE
        assert len(chunks) >= 3

        # Should have final chunk with stop reason
        final_chunk = [c for c in chunks if '"finish_reason":"stop"' in c]
        assert len(final_chunk) >= 1

        # Should end with DONE
        assert chunks[-1] == "data: [DONE]\n\n"

    @pytest.mark.asyncio
    async def test_streaming_no_text_content(self):
        """Test streaming with no text content (simple)."""

        async def mock_response():
            yield {
                "type": "message_start",
                "message": {"id": "msg_123", "role": "assistant"},
            }
            yield {
                "type": "content_block_delta",
                "delta": {"type": "text_delta", "text": ""},
            }
            yield {"type": "message_delta", "delta": {"stop_reason": "end_turn"}}
            yield {"type": "message_stop"}

        chunks = []
        async for chunk in stream_claude_response_openai_simple(
            mock_response(), "msg_123", "claude-3-5-sonnet-20241022", 1234567890
        ):
            chunks.append(chunk)

        # Should handle empty text content
        assert len(chunks) >= 3
        assert chunks[-1] == "data: [DONE]\n\n"

    @pytest.mark.asyncio
    async def test_streaming_stop_reason_mapping(self):
        """Test stop reason mapping from Claude to OpenAI (simple)."""

        async def mock_response():
            yield {
                "type": "message_start",
                "message": {"id": "msg_123", "role": "assistant"},
            }
            yield {"type": "message_delta", "delta": {"stop_reason": "stop_sequence"}}
            yield {"type": "message_stop"}

        chunks = []
        async for chunk in stream_claude_response_openai_simple(
            mock_response(), "msg_123", "claude-3-5-sonnet-20241022", 1234567890
        ):
            chunks.append(chunk)

        # Should map stop_sequence to stop
        stop_chunks = [c for c in chunks if '"finish_reason":"stop"' in c]
        assert len(stop_chunks) >= 1

    @pytest.mark.asyncio
    async def test_streaming_unknown_stop_reason(self):
        """Test streaming with unknown stop reason (simple)."""

        async def mock_response():
            yield {
                "type": "message_start",
                "message": {"id": "msg_123", "role": "assistant"},
            }
            yield {"type": "message_delta", "delta": {"stop_reason": "unknown_reason"}}
            yield {"type": "message_stop"}

        chunks = []
        async for chunk in stream_claude_response_openai_simple(
            mock_response(), "msg_123", "claude-3-5-sonnet-20241022", 1234567890
        ):
            chunks.append(chunk)

        # Should default to stop for unknown reasons
        stop_chunks = [c for c in chunks if '"finish_reason":"stop"' in c]
        assert len(stop_chunks) >= 1
