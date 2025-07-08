"""Tests for streaming functionality."""

import pytest

from ccproxy.services.streaming import (
    StreamingFormatter,
    stream_claude_response,
)


@pytest.mark.integration
class TestStreamingFormatter:
    """Test StreamingFormatter class."""

    def test_format_data_only(self):
        """Test format_data_only method."""
        data = {"type": "test", "message": "hello"}
        result = StreamingFormatter.format_data_only(data)

        assert result == 'data: {"type":"test","message":"hello"}\n\n'

    def test_format_message_start(self):
        """Test format_message_start method."""
        result = StreamingFormatter.format_message_start(
            "msg_123", "claude-3-5-sonnet-20241022"
        )

        assert "data: " in result
        assert '"type":"message_start"' in result
        assert '"id":"msg_123"' in result
        assert '"model":"claude-3-5-sonnet-20241022"' in result
        assert '"role":"assistant"' in result
        assert result.endswith("\n\n")

    def test_format_content_block_start(self):
        """Test format_content_block_start method."""
        result = StreamingFormatter.format_content_block_start(index=0)

        assert "data: " in result
        assert '"type":"content_block_start"' in result
        assert '"index":0' in result
        assert '"content_block":{"type":"text","text":""}' in result

    def test_format_content_block_delta(self):
        """Test format_content_block_delta method."""
        result = StreamingFormatter.format_content_block_delta("Hello", index=0)

        assert "data: " in result
        assert '"type":"content_block_delta"' in result
        assert '"index":0' in result
        assert '"delta":{"type":"text_delta","text":"Hello"}' in result

    def test_format_content_block_stop(self):
        """Test format_content_block_stop method."""
        result = StreamingFormatter.format_content_block_stop(index=0)

        assert "data: " in result
        assert '"type":"content_block_stop"' in result
        assert '"index":0' in result

    def test_format_message_delta(self):
        """Test format_message_delta method."""
        result = StreamingFormatter.format_message_delta(
            stop_reason="end_turn", stop_sequence=None
        )

        assert "data: " in result
        assert '"type":"message_delta"' in result
        assert '"stop_reason":"end_turn"' in result
        assert '"stop_sequence":null' in result
        assert '"usage":{"output_tokens":0}' in result

    def test_format_message_stop(self):
        """Test format_message_stop method."""
        result = StreamingFormatter.format_message_stop()

        assert "data: " in result
        assert '"type":"message_stop"' in result

    def test_format_error(self):
        """Test format_error method."""
        result = StreamingFormatter.format_error(
            "internal_server_error", "Something went wrong"
        )

        assert "data: " in result
        assert '"type":"error"' in result
        assert (
            '"error":{"type":"internal_server_error","message":"Something went wrong"}'
            in result
        )

    def test_format_done(self):
        """Test format_done method."""
        result = StreamingFormatter.format_done()

        assert result == "data: [DONE]\n\n"


@pytest.mark.integration
class TestStreamClaudeResponse:
    """Test stream_claude_response function."""

    @pytest.mark.asyncio
    async def test_successful_streaming(self, sample_streaming_response):
        """Test successful streaming response."""
        chunks = []
        async for chunk in stream_claude_response(
            sample_streaming_response, "msg_test123", "claude-3-5-sonnet-20241022"
        ):
            chunks.append(chunk)

        # Should have: message_start, content_block_start, deltas, content_block_stop,
        # message_delta, message_stop, done
        assert len(chunks) >= 7

        # Check that we have the expected sequence
        chunk_types = []
        for chunk in chunks:
            if '"type":"message_start"' in chunk:
                chunk_types.append("message_start")
            elif '"type":"content_block_start"' in chunk:
                chunk_types.append("content_block_start")
            elif '"type":"content_block_delta"' in chunk:
                chunk_types.append("content_block_delta")
            elif '"type":"content_block_stop"' in chunk:
                chunk_types.append("content_block_stop")
            elif '"type":"message_delta"' in chunk:
                chunk_types.append("message_delta")
            elif '"type":"message_stop"' in chunk:
                chunk_types.append("message_stop")
            elif "data: [DONE]" in chunk:
                chunk_types.append("done")

        # Verify the sequence
        assert "message_start" in chunk_types
        assert "content_block_start" in chunk_types
        assert "content_block_delta" in chunk_types
        assert "content_block_stop" in chunk_types
        assert "message_delta" in chunk_types
        assert "message_stop" in chunk_types
        assert "done" in chunk_types

        # Verify DONE is last
        assert chunk_types[-1] == "done"

    @pytest.mark.asyncio
    async def test_streaming_with_error(self):
        """Test streaming response with error."""

        async def error_generator():
            yield {"type": "content_block_delta", "delta": {"text": "Hello"}}
            raise Exception("Test error")

        chunks = []
        async for chunk in stream_claude_response(
            error_generator(), "msg_test123", "claude-3-5-sonnet-20241022"
        ):
            chunks.append(chunk)

        # Should still have DONE at the end
        assert any("data: [DONE]" in chunk for chunk in chunks)

        # Should have an error chunk
        error_chunks = [chunk for chunk in chunks if '"type":"error"' in chunk]
        assert len(error_chunks) > 0

    @pytest.mark.asyncio
    async def test_empty_streaming_response(self):
        """Test streaming with no content."""

        async def empty_generator():
            # Generator that yields nothing and ends
            return
            yield  # pragma: no cover

        chunks = []
        async for chunk in stream_claude_response(
            empty_generator(), "msg_test123", "claude-3-5-sonnet-20241022"
        ):
            chunks.append(chunk)

        # Should still have proper structure
        assert len(chunks) >= 6  # start, content_start, content_stop, delta, stop, done
        assert any("data: [DONE]" in chunk for chunk in chunks)
