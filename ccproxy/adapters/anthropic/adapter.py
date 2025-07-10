"""Anthropic API adapter for request/response transformations."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any

from ccproxy.adapters.anthropic.models import (
    MessageCreateParams,
    MessageResponse,
)
from ccproxy.adapters.anthropic.streaming import AnthropicStreamProcessor
from ccproxy.core.interfaces import APIAdapter


logger = logging.getLogger(__name__)


class AnthropicAPIAdapter(APIAdapter):
    """Adapter for handling Anthropic API format transformations.

    This adapter provides pure transformation logic for converting
    between internal formats and Anthropic API format.
    """

    def __init__(self) -> None:
        """Initialize the Anthropic API adapter."""
        self.stream_processor = AnthropicStreamProcessor()

    def adapt_request(self, request: dict[str, Any]) -> dict[str, Any]:
        """Convert a request to Anthropic API format.

        Args:
            request: The request data to convert

        Returns:
            The converted request data in Anthropic format

        Raises:
            ValueError: If the request format is invalid or unsupported
        """
        try:
            # Validate the request against Anthropic schema
            message_params = MessageCreateParams.model_validate(request)

            # Convert to dict for processing
            adapted_request = message_params.model_dump(exclude_none=True)

            logger.debug("Adapted request to Anthropic format: %s", adapted_request)
            return adapted_request

        except Exception as e:
            logger.error("Failed to adapt request to Anthropic format: %s", e)
            raise ValueError(f"Invalid request format: {e}") from e

    def adapt_response(self, response: dict[str, Any]) -> dict[str, Any]:
        """Convert a response from Anthropic API format.

        Args:
            response: The response data to convert

        Returns:
            The converted response data

        Raises:
            ValueError: If the response format is invalid or unsupported
        """
        try:
            # Validate the response against Anthropic schema
            message_response = MessageResponse.model_validate(response)

            # Convert to dict for processing
            adapted_response = message_response.model_dump(exclude_none=True)

            logger.debug("Adapted response from Anthropic format: %s", adapted_response)
            return adapted_response

        except Exception as e:
            logger.error("Failed to adapt response from Anthropic format: %s", e)
            raise ValueError(f"Invalid response format: {e}") from e

    def adapt_stream(
        self, stream: AsyncIterator[dict[str, Any]]
    ) -> AsyncIterator[dict[str, Any]]:
        """Convert a streaming response from Anthropic API format.

        Args:
            stream: The streaming response data to convert

        Yields:
            The converted streaming response chunks

        Raises:
            ValueError: If the stream format is invalid or unsupported
        """
        return self._adapt_stream_impl(stream)

    async def _adapt_stream_impl(
        self, stream: AsyncIterator[dict[str, Any]]
    ) -> AsyncIterator[dict[str, Any]]:
        """Implementation of the adapt_stream method as an async generator."""
        try:
            async for chunk in stream:
                # Validate and transform each chunk
                if isinstance(chunk, dict):
                    # Pass through validated chunks
                    yield chunk
                else:
                    logger.warning("Received non-dict chunk in stream: %s", chunk)  # type: ignore[unreachable]

        except Exception as e:
            logger.error("Failed to adapt stream from Anthropic format: %s", e)
            raise ValueError(f"Invalid stream format: {e}") from e

    async def format_sse_stream(
        self, stream: AsyncIterator[dict[str, Any]]
    ) -> AsyncIterator[str]:
        """Format a stream into Anthropic SSE format.

        Args:
            stream: The streaming response data to format

        Yields:
            Formatted SSE strings
        """
        async for sse_chunk in self.stream_processor.process_stream(stream):
            yield sse_chunk


__all__ = [
    "AnthropicAPIAdapter",
]
