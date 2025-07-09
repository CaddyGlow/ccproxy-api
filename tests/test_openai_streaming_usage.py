"""Tests for OpenAI streaming service with usage support."""

import asyncio
import json
from typing import Any

import pytest

from ccproxy.services.openai_streaming_formatter import (
    OpenAIStreamingFormatter,
    stream_claude_response_openai,
    stream_claude_response_openai_simple,
)


@pytest.mark.unit
class TestOpenAIStreamingFormatter:
    """Test OpenAI streaming formatter with usage support."""

    def test_format_final_chunk_with_usage(self):
        """Test formatting final chunk with usage data."""
        formatter = OpenAIStreamingFormatter()

        usage_data = {
            "prompt_tokens": 50,
            "completion_tokens": 100,
            "total_tokens": 150,
        }

        result = formatter.format_final_chunk(
            message_id="msg_123",
            model="claude-3-5-sonnet-20241022",
            created=1234567890,
            finish_reason="stop",
            usage=usage_data,
        )

        # Parse the SSE data
        assert result.startswith("data: ")
        data = json.loads(result[6:].strip())

        assert data["id"] == "msg_123"
        assert data["model"] == "claude-3-5-sonnet-20241022"
        assert data["choices"][0]["finish_reason"] == "stop"
        assert data["usage"] == usage_data

    def test_format_final_chunk_without_usage(self):
        """Test formatting final chunk without usage data."""
        formatter = OpenAIStreamingFormatter()

        result = formatter.format_final_chunk(
            message_id="msg_123",
            model="claude-3-5-sonnet-20241022",
            created=1234567890,
            finish_reason="stop",
            usage=None,
        )

        # Parse the SSE data
        data = json.loads(result[6:].strip())

        assert "usage" not in data


@pytest.mark.unit
class TestOpenAIStreamingWithUsage:
    """Test OpenAI streaming functions with usage support."""

    @pytest.mark.asyncio
    async def test_stream_claude_response_with_usage(self):
        """Test streaming with usage included."""

        async def mock_claude_stream():
            yield {"type": "message_start"}
            yield {
                "type": "content_block_delta",
                "delta": {"type": "text_delta", "text": "Hello"},
            }
            yield {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {"input_tokens": 25, "output_tokens": 50},
            }

        chunks = []
        async for chunk in stream_claude_response_openai(
            mock_claude_stream(),
            message_id="test_123",
            model="claude-3-5-sonnet-20241022",
            created=1234567890,
            include_usage=True,
        ):
            chunks.append(chunk)

        # Find the DONE marker
        done_chunk = next(c for c in chunks if c == "data: [DONE]\n\n")
        done_index = chunks.index(done_chunk)

        # The chunk before DONE should be the final chunk with usage
        final_chunk_str = chunks[done_index - 1]
        assert final_chunk_str.startswith("data: ")
        final_data = json.loads(final_chunk_str[6:].strip())

        assert final_data["choices"][0]["finish_reason"] == "stop"
        assert "usage" in final_data
        assert final_data["usage"]["prompt_tokens"] == 25
        assert final_data["usage"]["completion_tokens"] == 50
        assert final_data["usage"]["total_tokens"] == 75

    @pytest.mark.asyncio
    async def test_stream_claude_response_without_usage(self):
        """Test streaming without usage when include_usage is False."""

        async def mock_claude_stream():
            yield {"type": "message_start"}
            yield {
                "type": "content_block_delta",
                "delta": {"type": "text_delta", "text": "Hello"},
            }
            yield {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {"input_tokens": 25, "output_tokens": 50},
            }

        chunks = []
        async for chunk in stream_claude_response_openai(
            mock_claude_stream(),
            message_id="test_123",
            model="claude-3-5-sonnet-20241022",
            created=1234567890,
            include_usage=False,
        ):
            chunks.append(chunk)

        # Find the final chunk
        done_chunk = next(c for c in chunks if c == "data: [DONE]\n\n")
        done_index = chunks.index(done_chunk)
        final_chunk_str = chunks[done_index - 1]
        final_data = json.loads(final_chunk_str[6:].strip())

        # Should not have usage
        assert "usage" not in final_data

    @pytest.mark.asyncio
    async def test_stream_claude_response_simple_with_usage(self):
        """Test simple streaming with usage included."""

        async def mock_claude_stream():
            yield {"type": "message_start"}
            yield {
                "type": "content_block_delta",
                "delta": {"type": "text_delta", "text": "Hello"},
            }
            yield {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {"input_tokens": 15, "output_tokens": 30},
            }

        chunks = []
        async for chunk in stream_claude_response_openai_simple(
            mock_claude_stream(),
            message_id="test_456",
            model="claude-3-5-sonnet-20241022",
            created=1234567890,
            include_usage=True,
        ):
            chunks.append(chunk)

        # Find the final chunk
        done_chunk = next(c for c in chunks if c == "data: [DONE]\n\n")
        done_index = chunks.index(done_chunk)
        final_chunk_str = chunks[done_index - 1]
        final_data = json.loads(final_chunk_str[6:].strip())

        assert final_data["choices"][0]["finish_reason"] == "stop"
        assert "usage" in final_data
        assert final_data["usage"]["prompt_tokens"] == 15
        assert final_data["usage"]["completion_tokens"] == 30
        assert final_data["usage"]["total_tokens"] == 45

    @pytest.mark.asyncio
    async def test_stream_with_thinking_blocks_and_usage(self):
        """Test streaming with thinking blocks and usage data."""

        async def mock_claude_stream():
            yield {"type": "message_start"}
            yield {"type": "content_block_start", "content_block": {"type": "thinking"}}
            yield {
                "type": "content_block_delta",
                "delta": {"type": "thinking_delta", "thinking": "Let me think..."},
            }
            yield {"type": "content_block_stop"}
            yield {
                "type": "content_block_delta",
                "delta": {"type": "text_delta", "text": "The answer is 42."},
            }
            yield {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {"input_tokens": 20, "output_tokens": 40},
            }

        chunks = []
        async for chunk in stream_claude_response_openai(
            mock_claude_stream(),
            message_id="test_789",
            model="claude-3-5-sonnet-20241022",
            created=1234567890,
            include_usage=True,
        ):
            chunks.append(chunk)

        # Collect all content chunks
        content_chunks = []
        for chunk_str in chunks:
            if chunk_str.startswith("data: ") and chunk_str != "data: [DONE]\n\n":
                data = json.loads(chunk_str[6:].strip())
                if (
                    "delta" in data["choices"][0]
                    and "content" in data["choices"][0]["delta"]
                ):
                    content_chunks.append(data["choices"][0]["delta"]["content"])

        # Check thinking marker and content
        full_content = "".join(content_chunks)
        assert "[Thinking]" in full_content
        assert "Let me think..." in full_content
        assert "---" in full_content  # separator
        assert "The answer is 42." in full_content

        # Check final chunk has usage
        done_index = chunks.index("data: [DONE]\n\n")
        final_chunk_str = chunks[done_index - 1]
        final_data = json.loads(final_chunk_str[6:].strip())

        assert final_data["usage"]["prompt_tokens"] == 20
        assert final_data["usage"]["completion_tokens"] == 40
        assert final_data["usage"]["total_tokens"] == 60

    @pytest.mark.asyncio
    async def test_stream_with_new_stop_reasons(self):
        """Test streaming with new stop reasons (pause_turn, refusal)."""
        stop_reason_mappings = [("pause_turn", "stop"), ("refusal", "content_filter")]

        for anthropic_reason, expected_openai_reason in stop_reason_mappings:

            async def mock_claude_stream(reason=anthropic_reason):
                yield {"type": "message_start"}
                yield {
                    "type": "content_block_delta",
                    "delta": {"type": "text_delta", "text": "Content"},
                }
                yield {
                    "type": "message_delta",
                    "delta": {"stop_reason": reason},
                    "usage": {"input_tokens": 10, "output_tokens": 5},
                }

            chunks = []
            async for chunk in stream_claude_response_openai(
                mock_claude_stream(),
                message_id="test_stop",
                model="claude-3-5-sonnet-20241022",
                created=1234567890,
                include_usage=True,
            ):
                chunks.append(chunk)

            # Find final chunk
            done_index = chunks.index("data: [DONE]\n\n")
            final_chunk_str = chunks[done_index - 1]
            final_data = json.loads(final_chunk_str[6:].strip())

            assert final_data["choices"][0]["finish_reason"] == expected_openai_reason
            assert final_data["usage"]["total_tokens"] == 15
