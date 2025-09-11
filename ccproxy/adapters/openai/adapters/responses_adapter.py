"""OpenAI Response API format adapter.

This adapter handles pure Response API format operations without
cross-format conversion.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, AsyncIterator
from typing import Any

from ccproxy.adapters.base import BaseAPIAdapter
from ccproxy.adapters.openai.models.responses import (
    ResponseCompleted,
    ResponseRequest,
)


class ResponsesAdapter(BaseAPIAdapter):
    """Adapter for OpenAI Response API format."""

    def __init__(self) -> None:
        super().__init__("responses")

    async def adapt_request(self, request_data: dict[str, Any]) -> dict[str, Any]:
        """Adapt Response API request.

        For pure Response API, this is mostly validation and normalization.

        Args:
            request_data: Response API request

        Returns:
            Validated Response API request
        """
        # Validate using Pydantic model
        request = ResponseRequest(**request_data)
        return request.model_dump(exclude_none=True)

    async def adapt_response(self, response_data: dict[str, Any]) -> dict[str, Any]:
        """Adapt Response API response.

        For pure Response API, this is mostly validation.

        Args:
            response_data: Response API response

        Returns:
            Validated Response API response
        """
        # Handle both wrapped and unwrapped response formats
        if "response" in response_data and "type" in response_data:
            # Wrapped format (e.g., response.completed event)
            response = ResponseCompleted(**response_data)
            return response.model_dump(exclude_none=True)
        else:
            # Unwrapped format - return as-is with basic validation
            return response_data

    def adapt_stream(
        self, stream: AsyncIterator[dict[str, Any]]
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Adapt Response API streaming response.

        For pure Response API, this is passthrough.

        Args:
            stream: Response API streaming events

        Yields:
            Response API streaming events
        """
        return self._adapt_stream_impl(stream)

    async def _adapt_stream_impl(
        self, stream: AsyncIterator[dict[str, Any]]
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Implementation of stream adaptation."""
        async for event in stream:
            # For pure Response API, pass through events
            yield event

    async def adapt_error(self, error: dict[str, Any]) -> dict[str, Any]:
        """Convert error format - pass through for Response API.

        Args:
            error: Error response

        Returns:
            Pass-through error response
        """
        # For pure Response API, pass through errors
        return error


__all__ = ["ResponsesAdapter"]
