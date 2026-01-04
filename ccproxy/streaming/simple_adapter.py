"""Simplified streaming adapter that bypasses complex type conversions.

This adapter provides a direct dict-based interface for streaming without
the complexity of the shim layer.
"""

from collections.abc import AsyncGenerator, AsyncIterator
from typing import Any


class SimpleStreamingAdapter:
    """Simple adapter for streaming responses that works directly with dicts."""

    def __init__(self, name: str = "simple_streaming"):
        """Initialize the simple adapter."""
        self.name = name

    async def adapt_request(self, request: dict[str, Any]) -> dict[str, Any]:
        """Pass through request - no adaptation needed for streaming."""
        return request

    async def adapt_response(self, response: dict[str, Any]) -> dict[str, Any]:
        """Pass through response - no adaptation needed for streaming."""
        return response

    def adapt_stream(
        self, stream: AsyncIterator[dict[str, Any]]
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Pass through stream - no adaptation needed for simple streaming."""

        async def passthrough_stream() -> AsyncGenerator[dict[str, Any], None]:
            async for chunk in stream:
                yield chunk

        return passthrough_stream()

    async def adapt_error(self, error: dict[str, Any]) -> dict[str, Any]:
        """Pass through error - no adaptation needed."""
        return error
