"""Anthropic streaming support for Server-Sent Events (SSE)."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import Any

from ccproxy.models.errors import ErrorDetail, StreamingError


logger = logging.getLogger(__name__)


class AnthropicStreamingFormatter:
    """Formats streaming responses to match Anthropic's SSE format."""

    @staticmethod
    def format_event(event_type: str, data: dict[str, Any]) -> str:
        """Format an event for Server-Sent Events.

        Args:
            event_type: Type of the event
            data: Event data dictionary

        Returns:
            Formatted SSE string
        """
        json_data = json.dumps(data, separators=(",", ":"))
        return f"event: {event_type}\ndata: {json_data}\n\n"

    @staticmethod
    def format_data_only(data: dict[str, Any]) -> str:
        """Format SSE event with explicit event type.

        Args:
            data: Event data dictionary

        Returns:
            Formatted SSE string with event type
        """
        json_data = json.dumps(data, separators=(",", ":"))
        event_type = data.get("type", "unknown")
        return f"event: {event_type}\ndata: {json_data}\n\n"

    @staticmethod
    def format_message_start(
        message_id: str, model: str, role: str = "assistant"
    ) -> str:
        """Format message start event."""
        data = {
            "type": "message_start",
            "message": {
                "id": message_id,
                "type": "message",
                "role": role,
                "content": [],
                "model": model,
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 0, "output_tokens": 0},
            },
        }
        return AnthropicStreamingFormatter.format_data_only(data)

    @staticmethod
    def format_content_block_start(index: int = 0) -> str:
        """Format content block start event."""
        data = {
            "type": "content_block_start",
            "index": index,
            "content_block": {"type": "text", "text": ""},
        }
        return AnthropicStreamingFormatter.format_data_only(data)

    @staticmethod
    def format_content_block_delta(text: str, index: int = 0) -> str:
        """Format content block delta event."""
        data = {
            "type": "content_block_delta",
            "index": index,
            "delta": {"type": "text_delta", "text": text},
        }
        return AnthropicStreamingFormatter.format_data_only(data)

    @staticmethod
    def format_content_block_stop(index: int = 0) -> str:
        """Format content block stop event."""
        data = {"type": "content_block_stop", "index": index}
        return AnthropicStreamingFormatter.format_data_only(data)

    @staticmethod
    def format_message_delta(
        stop_reason: str = "end_turn", stop_sequence: str | None = None
    ) -> str:
        """Format message delta event."""
        data = {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": stop_sequence},
            "usage": {"output_tokens": 0},
        }
        return AnthropicStreamingFormatter.format_data_only(data)

    @staticmethod
    def format_message_stop() -> str:
        """Format message stop event."""
        data = {"type": "message_stop"}
        return AnthropicStreamingFormatter.format_data_only(data)

    @staticmethod
    def format_error(error_type: str, message: str) -> str:
        """Format streaming error event."""
        error = StreamingError(error=ErrorDetail(type=error_type, message=message))
        return AnthropicStreamingFormatter.format_data_only(error.model_dump())

    @staticmethod
    def format_done() -> str:
        """Format the final DONE event."""
        return "data: [DONE]\n\n"


class AnthropicStreamProcessor:
    """Processes streaming responses for Anthropic format."""

    def __init__(self, formatter: AnthropicStreamingFormatter | None = None) -> None:
        """Initialize the stream processor.

        Args:
            formatter: Optional formatter instance to use
        """
        self.formatter = formatter or AnthropicStreamingFormatter()

    async def process_stream(
        self, stream: AsyncIterator[dict[str, Any]]
    ) -> AsyncIterator[str]:
        """Process a stream of response data into Anthropic SSE format.

        Args:
            stream: Stream of response data dictionaries

        Yields:
            Formatted SSE strings
        """
        try:
            async for chunk in stream:
                if "type" in chunk:
                    yield self.formatter.format_data_only(chunk)
                else:
                    logger.warning("Received chunk without type field: %s", chunk)
        except Exception as e:
            logger.error("Error processing stream: %s", e)
            yield self.formatter.format_error("stream_error", str(e))
        finally:
            yield self.formatter.format_done()


__all__ = [
    "AnthropicStreamingFormatter",
    "AnthropicStreamProcessor",
]
