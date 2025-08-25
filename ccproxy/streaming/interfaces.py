"""Streaming interfaces for provider implementations.

This module defines interfaces that providers can implement to extend
streaming functionality without coupling core code to specific providers.
"""

from typing import Protocol, TypedDict


class StreamingMetrics(TypedDict, total=False):
    """Standard streaming metrics structure."""

    tokens_input: int | None
    tokens_output: int | None
    cache_read_tokens: int | None
    cache_write_tokens: int | None
    cost_usd: float | None


class IStreamingMetricsCollector(Protocol):
    """Interface for provider-specific streaming metrics collection.

    Providers implement this interface to extract token usage and other
    metrics from their specific streaming response formats.
    """

    def process_chunk(self, chunk_str: str) -> bool:
        """Process a streaming chunk to extract metrics.

        Args:
            chunk_str: Raw chunk string from streaming response

        Returns:
            True if this was the final chunk with complete metrics, False otherwise
        """
        ...

    def process_raw_chunk(self, chunk_str: str) -> bool:
        """Process a raw provider chunk before any format conversion.

        This method is called with chunks in the provider's native format,
        before any OpenAI/Anthropic format conversion happens.

        Args:
            chunk_str: Raw chunk string in provider's native format

        Returns:
            True if this was the final chunk with complete metrics, False otherwise
        """
        ...

    def process_converted_chunk(self, chunk_str: str) -> bool:
        """Process a chunk after format conversion.

        This method is called with chunks after they've been converted
        to a different format (e.g., OpenAI format).

        Args:
            chunk_str: Chunk string after format conversion

        Returns:
            True if this was the final chunk with complete metrics, False otherwise
        """
        ...

    def get_metrics(self) -> StreamingMetrics:
        """Get the collected metrics.

        Returns:
            Dictionary with provider-specific metrics (tokens, costs, etc.)
        """
        ...
