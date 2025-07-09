"""Tests for streaming functionality."""

from collections.abc import AsyncGenerator
from typing import Any

import pytest

from ccproxy.services.anthropic_streaming import (
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

        assert result == 'event: test\ndata: {"type":"test","message":"hello"}\n\n'

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

    def test_sse_format_compliance(self):
        """Test that all SSE events follow the correct Server-Sent Events format."""
        # Test that all events (except DONE) have both event: and data: lines

        # Test message_start event
        result = StreamingFormatter.format_message_start(
            "msg_123", "claude-3-5-sonnet-20241022"
        )
        assert result.startswith("event: message_start\ndata: ")
        assert result.endswith("\n\n")

        # Test content_block_start event
        result = StreamingFormatter.format_content_block_start()
        assert result.startswith("event: content_block_start\ndata: ")
        assert result.endswith("\n\n")

        # Test content_block_delta event
        result = StreamingFormatter.format_content_block_delta("Hello")
        assert result.startswith("event: content_block_delta\ndata: ")
        assert result.endswith("\n\n")

        # Test content_block_stop event
        result = StreamingFormatter.format_content_block_stop()
        assert result.startswith("event: content_block_stop\ndata: ")
        assert result.endswith("\n\n")

        # Test message_delta event
        result = StreamingFormatter.format_message_delta()
        assert result.startswith("event: message_delta\ndata: ")
        assert result.endswith("\n\n")

        # Test message_stop event
        result = StreamingFormatter.format_message_stop()
        assert result.startswith("event: message_stop\ndata: ")
        assert result.endswith("\n\n")

        # Test error event
        result = StreamingFormatter.format_error("test_error", "test message")
        assert result.startswith("event: error\ndata: ")
        assert result.endswith("\n\n")

        # Test generic data event
        result = StreamingFormatter.format_data_only(
            {"type": "custom_event", "data": "test"}
        )
        assert result.startswith("event: custom_event\ndata: ")
        assert result.endswith("\n\n")

    def test_event_type_extraction(self):
        """Test that event types are correctly extracted from data for SSE event headers."""

        # Test various event types
        test_cases: list[tuple[dict[str, Any], str]] = [
            ({"type": "message_start", "message": {}}, "message_start"),
            ({"type": "content_block_start", "index": 0}, "content_block_start"),
            ({"type": "content_block_delta", "delta": {}}, "content_block_delta"),
            ({"type": "content_block_stop", "index": 0}, "content_block_stop"),
            ({"type": "message_delta", "delta": {}}, "message_delta"),
            ({"type": "message_stop"}, "message_stop"),
            ({"type": "error", "error": {}}, "error"),
            ({"type": "custom_type", "data": "test"}, "custom_type"),
            ({"message": "no type field"}, "unknown"),  # Missing type field
        ]

        for data, expected_type in test_cases:
            result = StreamingFormatter.format_data_only(data)
            assert result.startswith(f"event: {expected_type}\ndata: ")

    def test_anthropic_sdk_compatibility(self):
        """Test that our SSE format is compatible with Anthropic SDK expectations."""
        # This test ensures our format matches what the Anthropic SDK expects

        # Test a complete streaming sequence
        events = [
            StreamingFormatter.format_message_start(
                "msg_123", "claude-3-5-sonnet-20241022"
            ),
            StreamingFormatter.format_content_block_start(),
            StreamingFormatter.format_content_block_delta("Hello world"),
            StreamingFormatter.format_content_block_stop(),
            StreamingFormatter.format_message_delta("end_turn"),
            StreamingFormatter.format_message_stop(),
            StreamingFormatter.format_done(),
        ]

        # Parse each event as if we were the Anthropic SDK
        for event in events[:-1]:  # Skip DONE event
            lines = event.strip().split("\n")
            assert len(lines) == 2, (
                f"Event should have exactly 2 lines (event: and data:), got: {lines}"
            )

            # First line should be event type
            assert lines[0].startswith("event: "), (
                f"First line should start with 'event: ', got: {lines[0]}"
            )

            # Second line should be data
            assert lines[1].startswith("data: "), (
                f"Second line should start with 'data: ', got: {lines[1]}"
            )

            # Data should be valid JSON
            import json

            try:
                json_data = lines[1][6:]  # Remove 'data: ' prefix
                parsed = json.loads(json_data)
                assert "type" in parsed, f"Data should contain 'type' field: {parsed}"
            except json.JSONDecodeError:
                # DONE event is not JSON
                if "data: [DONE]" not in lines[1]:
                    pytest.fail(f"Data line should contain valid JSON: {lines[1]}")

        # DONE event should be data-only
        done_event = events[-1]
        assert done_event == "data: [DONE]\n\n"

    def test_json_format_consistency(self):
        """Test that all JSON in SSE events is consistently formatted."""
        # Test that JSON is compact (no spaces after separators)

        result = StreamingFormatter.format_message_start(
            "msg_123", "claude-3-5-sonnet-20241022"
        )
        data_line = result.split("\n")[1]  # Get the data: line
        json_str = data_line[6:]  # Remove 'data: ' prefix

        # Should not have spaces after separators
        assert ", " not in json_str, "JSON should not have spaces after commas"
        assert ": " not in json_str, "JSON should not have spaces after colons"

        # Should be valid JSON
        import json

        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)
        assert "type" in parsed


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


@pytest.mark.integration
class TestAnthropicSDKCompatibility:
    """Integration tests for Anthropic SDK compatibility."""

    @pytest.mark.asyncio
    async def test_sse_format_sdk_parsing(self):
        """Test that our SSE format can be parsed by SSE parsers."""
        # This test simulates what the Anthropic SDK does internally

        # Generate a complete streaming response
        chunks = []
        async for chunk in stream_claude_response(
            self._sample_claude_response(), "msg_test123", "claude-3-5-sonnet-20241022"
        ):
            chunks.append(chunk)

        # Parse each chunk as SSE events
        for chunk in chunks:
            lines = chunk.strip().split("\n")

            if chunk.startswith("event: "):
                # Standard SSE event with event type and data
                assert len(lines) == 2, f"SSE event should have 2 lines, got: {lines}"

                event_line = lines[0]
                data_line = lines[1]

                # Validate event line
                assert event_line.startswith("event: "), (
                    f"First line should be event type: {event_line}"
                )
                event_type = event_line[7:]  # Remove "event: " prefix
                assert event_type in [
                    "message_start",
                    "content_block_start",
                    "content_block_delta",
                    "content_block_stop",
                    "message_delta",
                    "message_stop",
                    "error",
                ], f"Unknown event type: {event_type}"

                # Validate data line
                assert data_line.startswith("data: "), (
                    f"Second line should be data: {data_line}"
                )

                # Validate JSON in data line
                import json

                json_str = data_line[6:]  # Remove "data: " prefix
                try:
                    parsed_data = json.loads(json_str)
                    assert "type" in parsed_data, (
                        f"Data should have 'type' field: {parsed_data}"
                    )
                    assert parsed_data["type"] == event_type, (
                        f"Event type mismatch: {event_type} vs {parsed_data['type']}"
                    )
                except json.JSONDecodeError:
                    pytest.fail(f"Invalid JSON in data line: {json_str}")

            elif chunk.startswith("data: [DONE]"):
                # DONE event is data-only
                assert chunk == "data: [DONE]\n\n", (
                    f"DONE event format incorrect: {repr(chunk)}"
                )

            else:
                pytest.fail(f"Unexpected chunk format: {repr(chunk)}")

    @pytest.mark.asyncio
    async def test_event_sequence_completeness(self):
        """Test that streaming produces a complete event sequence."""

        chunks = []
        async for chunk in stream_claude_response(
            self._sample_claude_response(), "msg_test123", "claude-3-5-sonnet-20241022"
        ):
            chunks.append(chunk)

        # Extract event types from chunks
        event_types = []
        for chunk in chunks:
            if chunk.startswith("event: "):
                event_type = chunk.split("\n")[0][7:]  # Extract event type
                event_types.append(event_type)
            elif "data: [DONE]" in chunk:
                event_types.append("done")

        # Verify required event sequence
        required_events = [
            "message_start",
            "content_block_start",
            "content_block_delta",
            "content_block_stop",
            "message_delta",
            "message_stop",
            "done",
        ]

        for required_event in required_events:
            assert required_event in event_types, (
                f"Missing required event: {required_event}"
            )

        # Verify DONE is last
        assert event_types[-1] == "done", (
            f"DONE should be last event, got: {event_types[-1]}"
        )

        # Verify event order makes sense
        message_start_idx = event_types.index("message_start")
        content_start_idx = event_types.index("content_block_start")
        content_stop_idx = event_types.index("content_block_stop")
        message_delta_idx = event_types.index("message_delta")
        message_stop_idx = event_types.index("message_stop")
        done_idx = event_types.index("done")

        # Check logical order
        assert message_start_idx < content_start_idx, (
            "message_start should come before content_block_start"
        )
        assert content_start_idx < content_stop_idx, (
            "content_block_start should come before content_block_stop"
        )
        assert content_stop_idx < message_delta_idx, (
            "content_block_stop should come before message_delta"
        )
        assert message_delta_idx < message_stop_idx, (
            "message_delta should come before message_stop"
        )
        assert message_stop_idx < done_idx, "message_stop should come before done"

    @pytest.mark.asyncio
    async def test_error_handling_sse_format(self):
        """Test that errors are properly formatted as SSE events."""

        async def error_generator():
            yield {"type": "content_block_delta", "delta": {"text": "Hello"}}
            raise ValueError("Test error")

        chunks = []
        async for chunk in stream_claude_response(
            error_generator(), "msg_test123", "claude-3-5-sonnet-20241022"
        ):
            chunks.append(chunk)

        # Should have at least one error event
        error_chunks = [chunk for chunk in chunks if chunk.startswith("event: error\n")]
        assert len(error_chunks) > 0, "Should have at least one error event"

        # Verify error event format
        error_chunk = error_chunks[0]
        lines = error_chunk.strip().split("\n")
        assert len(lines) == 2, f"Error event should have 2 lines: {lines}"
        assert lines[0] == "event: error", (
            f"First line should be 'event: error': {lines[0]}"
        )
        assert lines[1].startswith("data: "), (
            f"Second line should start with 'data: ': {lines[1]}"
        )

        # Verify error data structure
        import json

        error_data = json.loads(lines[1][6:])  # Remove "data: " prefix
        assert error_data["type"] == "error", (
            f"Error data should have type 'error': {error_data}"
        )
        assert "error" in error_data, (
            f"Error data should have 'error' field: {error_data}"
        )
        assert "type" in error_data["error"], (
            f"Error should have nested type: {error_data['error']}"
        )
        assert "message" in error_data["error"], (
            f"Error should have message: {error_data['error']}"
        )

    async def _sample_claude_response(self) -> AsyncGenerator[dict[str, Any], None]:
        """Generate a sample Claude response for testing."""
        yield {"type": "content_block_delta", "delta": {"text": "Hello"}}
        yield {"type": "content_block_delta", "delta": {"text": " world"}}
        yield {"type": "message_delta", "delta": {"stop_reason": "end_turn"}}
