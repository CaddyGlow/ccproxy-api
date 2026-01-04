"""Claude API streaming metrics extraction utilities.

This module provides utilities for extracting token usage from
Anthropic streaming responses.
"""

from typing import Any, TypedDict


class UsageData(TypedDict, total=False):
    """Token usage data extracted from streaming or non-streaming responses."""

    input_tokens: int | None
    output_tokens: int | None
    cache_read_input_tokens: int | None
    cache_creation_input_tokens: int | None
    event_type: str | None  # Extra field for tracking event source
    model: str | None  # Extra field for model information


def extract_usage_from_streaming_chunk(chunk_data: Any) -> UsageData | None:
    """Extract usage information from Anthropic streaming response chunk.

    This function looks for usage information in both message_start and message_delta events
    from Anthropic's streaming API responses. message_start contains initial input tokens,
    message_delta contains final output tokens.

    Args:
        chunk_data: Streaming response chunk dictionary

    Returns:
        UsageData with token counts or None if no usage found
    """
    if not isinstance(chunk_data, dict):
        return None

    chunk_type = chunk_data.get("type")

    # Look for message_start events with initial usage (input tokens)
    if chunk_type == "message_start" and "message" in chunk_data:
        message = chunk_data["message"]
        # Extract model name if present
        model = message.get("model")
        if "usage" in message:
            usage = message["usage"]
            return UsageData(
                input_tokens=usage.get("input_tokens"),
                output_tokens=usage.get(
                    "output_tokens"
                ),  # Initial output tokens (usually small)
                cache_read_input_tokens=usage.get("cache_read_input_tokens"),
                cache_creation_input_tokens=usage.get("cache_creation_input_tokens"),
                event_type="message_start",
                model=model,  # Include model in usage data
            )

    # Look for message_delta events with final usage (output tokens)
    elif chunk_type == "message_delta" and "usage" in chunk_data:
        usage = chunk_data["usage"]
        return UsageData(
            input_tokens=usage.get("input_tokens"),  # Usually None in delta
            output_tokens=usage.get("output_tokens"),  # Final output token count
            cache_read_input_tokens=usage.get("cache_read_input_tokens"),
            cache_creation_input_tokens=usage.get("cache_creation_input_tokens"),
            event_type="message_delta",
        )

    return None
