"""Codex-specific streaming metrics extraction utilities.

This module provides utilities for extracting token usage from
OpenAI/Codex streaming responses.
"""

import json
from typing import Any

import structlog

from ccproxy.streaming.interfaces import StreamingMetrics


logger = structlog.get_logger(__name__)


def extract_usage_from_codex_chunk(chunk_data: Any) -> dict[str, Any] | None:
    """Extract usage information from OpenAI/Codex streaming response chunk.

    OpenAI/Codex sends usage information in the final streaming chunk where
    usage is not null. Earlier chunks have usage=null.

    Args:
        chunk_data: Streaming response chunk dictionary

    Returns:
        Dictionary with token counts or None if no usage found
    """
    if not isinstance(chunk_data, dict):
        return None

    # Check for different Codex response formats
    # 1. Standard OpenAI format (chat.completion.chunk)
    object_type = chunk_data.get("object", "")
    usage = chunk_data.get("usage")

    if usage and object_type.startswith(("chat.completion", "codex.response")):
        return {
            "input_tokens": usage.get("prompt_tokens"),  # OpenAI uses prompt_tokens
            "output_tokens": usage.get(
                "completion_tokens"
            ),  # OpenAI uses completion_tokens
            "total_tokens": usage.get("total_tokens"),
            "event_type": "openai_completion",
        }

    # 2. Codex CLI response format (response.completed event)
    event_type = chunk_data.get("type", "")
    if event_type == "response.completed" and "response" in chunk_data:
        response_data = chunk_data["response"]
        if isinstance(response_data, dict) and "usage" in response_data:
            usage = response_data["usage"]
            if usage:
                # Codex CLI uses input_tokens/output_tokens directly
                return {
                    "input_tokens": usage.get("input_tokens"),
                    "output_tokens": usage.get("output_tokens"),
                    "total_tokens": usage.get("total_tokens"),
                    "event_type": "codex_cli_response",
                }

    return None


class CodexStreamingMetricsCollector:
    """Collects and manages token metrics during Codex streaming responses.

    Implements IStreamingMetricsCollector interface for Codex/OpenAI.
    """

    def __init__(self, request_id: str | None = None) -> None:
        """Initialize the metrics collector.

        Args:
            request_id: Optional request ID for logging context
        """
        self.request_id = request_id
        self.metrics: StreamingMetrics = {
            "tokens_input": None,
            "tokens_output": None,
            "cache_read_tokens": None,  # OpenAI doesn't support cache tokens yet
            "cache_write_tokens": None,
            "cost_usd": None,
        }

    def process_raw_chunk(self, chunk_str: str) -> bool:
        """Process raw Codex format chunk before any conversion.

        This handles Codex's native response.completed event format.
        """
        return self.process_chunk(chunk_str)

    def process_converted_chunk(self, chunk_str: str) -> bool:
        """Process chunk after conversion to OpenAI format.

        When Codex responses are converted to OpenAI chat completion format,
        this method extracts metrics from the converted OpenAI format.
        """
        # After conversion, we'd see standard OpenAI format
        # For now, delegate to main process_chunk which handles both
        return self.process_chunk(chunk_str)

    def process_chunk(self, chunk_str: str) -> bool:
        """Process a streaming chunk to extract OpenAI/Codex token metrics.

        Args:
            chunk_str: Raw chunk string from streaming response

        Returns:
            True if this was the final chunk with complete metrics, False otherwise
        """
        # Check if this chunk contains usage information
        if "usage" not in chunk_str:
            return False

        logger.debug(
            "codex_processing_chunk_with_usage",
            chunk_preview=chunk_str[:300],
            request_id=self.request_id,
        )

        try:
            # Parse SSE data lines to find usage information
            # Codex sends complete JSON on a single line after "data: "
            for line in chunk_str.split("\n"):
                if line.startswith("data: "):
                    data_str = line[6:].strip()
                    if data_str and data_str != "[DONE]":
                        event_data = json.loads(data_str)

                        # Log event type for debugging
                        event_type = event_data.get("type", "")
                        if event_type == "response.completed":
                            logger.debug(
                                "codex_found_completed_event",
                                has_response=("response" in event_data),
                                has_usage=("usage" in event_data.get("response", {}))
                                if "response" in event_data
                                else False,
                                request_id=self.request_id,
                            )

                        usage_data = extract_usage_from_codex_chunk(event_data)

                        if usage_data:
                            # Store token counts from the event
                            self.metrics["tokens_input"] = usage_data.get(
                                "input_tokens"
                            )
                            self.metrics["tokens_output"] = usage_data.get(
                                "output_tokens"
                            )

                            logger.debug(
                                "codex_extracted_tokens",
                                tokens_input=self.metrics["tokens_input"],
                                tokens_output=self.metrics["tokens_output"],
                                total_tokens=usage_data.get("total_tokens"),
                                event_type=usage_data.get("event_type"),
                                request_id=self.request_id,
                            )
                            return True  # This is the final event with complete metrics

                        break  # Only process first valid data line

        except (json.JSONDecodeError, KeyError) as e:
            logger.debug(
                "codex_failed_to_parse_metrics",
                error=str(e),
                request_id=self.request_id,
            )

        return False

    def get_metrics(self) -> StreamingMetrics:
        """Get the current collected metrics.

        Returns:
            Current token metrics
        """
        return self.metrics.copy()
