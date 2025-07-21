"""Streaming metrics extraction utilities.

This module provides utilities for extracting token usage and calculating costs
from Anthropic streaming responses in a testable, modular way.
"""

import json
from typing import Any

import structlog

from ccproxy.models.types import StreamingTokenMetrics, UsageData
from ccproxy.utils.cost_calculator import calculate_token_cost


logger = structlog.get_logger(__name__)


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


class StreamingMetricsCollector:
    """Collects and manages token metrics during streaming responses."""

    def __init__(self, request_id: str | None = None) -> None:
        """Initialize the metrics collector.

        Args:
            request_id: Optional request ID for logging context
        """
        self.request_id = request_id
        self.metrics = StreamingTokenMetrics(
            tokens_input=None,
            tokens_output=None,
            cache_read_tokens=None,
            cache_write_tokens=None,
            cost_usd=None,
        )

    def process_chunk(self, chunk_str: str) -> bool:
        """Process a streaming chunk to extract token metrics.

        Args:
            chunk_str: Raw chunk string from streaming response

        Returns:
            True if this was the final chunk with complete metrics, False otherwise
        """
        # Check if this chunk contains usage information
        # Look for usage data in any chunk - the event type will be determined from the JSON
        if "usage" not in chunk_str:
            return False

        logger.debug(
            "Processing chunk with usage",
            chunk_preview=chunk_str[:300],
            request_id=self.request_id,
        )

        try:
            # Parse SSE data lines to find usage information
            for line in chunk_str.split("\n"):
                if line.startswith("data: "):
                    data_str = line[6:].strip()
                    if data_str and data_str != "[DONE]":
                        event_data = json.loads(data_str)
                        usage_data = extract_usage_from_streaming_chunk(event_data)

                        if usage_data:
                            event_type = usage_data.get("event_type")

                            # Handle message_start: get input tokens and initial cache tokens
                            if event_type == "message_start":
                                self.metrics["tokens_input"] = usage_data.get(
                                    "input_tokens"
                                )
                                self.metrics["cache_read_tokens"] = (
                                    usage_data.get("cache_read_input_tokens")
                                    or self.metrics["cache_read_tokens"]
                                )
                                self.metrics["cache_write_tokens"] = (
                                    usage_data.get("cache_creation_input_tokens")
                                    or self.metrics["cache_write_tokens"]
                                )
                                logger.debug(
                                    "Extracted input tokens from message_start",
                                    tokens_input=self.metrics["tokens_input"],
                                    cache_read_tokens=self.metrics["cache_read_tokens"],
                                    cache_write_tokens=self.metrics[
                                        "cache_write_tokens"
                                    ],
                                    request_id=self.request_id,
                                )
                                return False  # Not final yet

                            # Handle message_delta: get final output tokens
                            elif event_type == "message_delta":
                                self.metrics["tokens_output"] = usage_data.get(
                                    "output_tokens"
                                )
                                logger.debug(
                                    "Extracted output tokens from message_delta",
                                    tokens_output=self.metrics["tokens_output"],
                                    request_id=self.request_id,
                                )
                                return True  # This is the final event

                        break  # Only process first valid data line

        except (json.JSONDecodeError, KeyError) as e:
            logger.debug(
                "Failed to parse streaming token metrics",
                error=str(e),
                request_id=self.request_id,
            )

        return False

    def calculate_final_cost(self, model: str | None) -> float | None:
        """Calculate the final cost based on collected metrics.

        Args:
            model: Model name for pricing lookup

        Returns:
            Final cost in USD or None if calculation fails
        """
        cost_usd = calculate_token_cost(
            self.metrics["tokens_input"],
            self.metrics["tokens_output"],
            model,
            self.metrics["cache_read_tokens"],
            self.metrics["cache_write_tokens"],
        )
        self.metrics["cost_usd"] = cost_usd

        logger.debug(
            "Final streaming token metrics",
            tokens_input=self.metrics["tokens_input"],
            tokens_output=self.metrics["tokens_output"],
            cache_read_tokens=self.metrics["cache_read_tokens"],
            cache_write_tokens=self.metrics["cache_write_tokens"],
            cost_usd=cost_usd,
            request_id=self.request_id,
        )

        return cost_usd

    def get_metrics(self) -> StreamingTokenMetrics:
        """Get the current collected metrics.

        Returns:
            Current token metrics
        """
        return self.metrics.copy()
