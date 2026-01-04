"""Service interfaces for explicit dependency injection.

This module defines protocol interfaces for core services that adapters need,
enabling explicit dependency injection and removing the service locator pattern.
"""

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any, Protocol

import httpx
from starlette.responses import Response


if TYPE_CHECKING:
    from ccproxy.core.request_context import RequestContext


class IRequestHandler(Protocol):
    """Protocol for request handling functionality.

    Note: The dispatch_request method has been removed in favor of
    using plugin adapters' handle_request() method directly.
    """

    pass


class IRequestTracer(Protocol):
    """Request tracing interface."""

    async def trace_request(
        self,
        request_id: str,
        method: str,
        url: str,
        headers: dict[str, str],
        body: bytes | None = None,
    ) -> None:
        """Trace an outgoing request.

        Args:
            request_id: Unique request identifier
            method: HTTP method
            url: Target URL
            headers: Request headers
            body: Request body if available
        """
        ...

    async def trace_response(
        self,
        request_id: str,
        status: int,
        headers: dict[str, str],
        body: bytes | None = None,
    ) -> None:
        """Trace an incoming response.

        Args:
            request_id: Unique request identifier
            status: HTTP status code
            headers: Response headers
            body: Response body if available
        """
        ...

    def should_trace(self) -> bool:
        """Check if tracing is enabled.

        Returns:
            True if tracing should be performed
        """
        ...


class IMetricsCollector(Protocol):
    """Metrics collection interface."""

    def track_request(
        self, method: str, path: str, provider: str | None = None
    ) -> None:
        """Track an incoming request.

        Args:
            method: HTTP method
            path: Request path
            provider: Optional provider identifier
        """
        ...

    def track_response(
        self, status: int, duration: float, provider: str | None = None
    ) -> None:
        """Track a response.

        Args:
            status: HTTP status code
            duration: Response time in seconds
            provider: Optional provider identifier
        """
        ...

    def track_error(self, error_type: str, provider: str | None = None) -> None:
        """Track an error occurrence.

        Args:
            error_type: Type of error
            provider: Optional provider identifier
        """
        ...

    def track_tokens(
        self,
        input_tokens: int,
        output_tokens: int,
        provider: str | None = None,
        model: str | None = None,
    ) -> None:
        """Track token usage.

        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            provider: Optional provider identifier
            model: Optional model identifier
        """
        ...


class StreamingMetrics(Protocol):
    """Streaming response handler interface."""

    async def handle_stream(
        self,
        response: httpx.Response,
        request_context: "RequestContext | None" = None,
    ) -> AsyncIterator[bytes]:
        """Handle a streaming response.

        Args:
            response: HTTP response object
            request_context: Optional request context

        Yields:
            Response chunks
        """
        ...

    def create_streaming_response(
        self,
        stream: AsyncIterator[bytes],
        headers: dict[str, str] | None = None,
    ) -> Response:
        """Create a streaming response.

        Args:
            stream: Async iterator of response chunks
            headers: Optional response headers

        Returns:
            Streaming response object
        """
        ...

    async def handle_streaming_request(
        self,
        method: str,
        url: str,
        headers: dict[str, str],
        body: bytes,
        handler_config: Any,
        request_context: Any,
        client_config: dict[str, Any] | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> Any:
        """Handle a streaming request.

        Args:
            method: HTTP method
            url: Target URL
            headers: Request headers
            body: Request body
            handler_config: Handler configuration
            request_context: Request context
            client_config: Optional client configuration
            client: Optional HTTP client

        Returns:
            Deferred streaming response
        """
        ...


# Null implementations for optional dependencies


class NullRequestTracer:
    """Null implementation of request tracer (no-op)."""

    async def trace_request(
        self,
        request_id: str,
        method: str,
        url: str,
        headers: dict[str, str],
        body: bytes | None = None,
    ) -> None:
        """No-op request tracing."""
        pass

    async def trace_response(
        self,
        request_id: str,
        status: int,
        headers: dict[str, str],
        body: bytes | None = None,
    ) -> None:
        """No-op response tracing."""
        pass

    def should_trace(self) -> bool:
        """Always return False for null tracer."""
        return False


class NullMetricsCollector:
    """Null implementation of metrics collector (no-op)."""

    def track_request(
        self, method: str, path: str, provider: str | None = None
    ) -> None:
        """No-op request tracking."""
        pass

    def track_response(
        self, status: int, duration: float, provider: str | None = None
    ) -> None:
        """No-op response tracking."""
        pass

    def track_error(self, error_type: str, provider: str | None = None) -> None:
        """No-op error tracking."""
        pass

    def track_tokens(
        self,
        input_tokens: int,
        output_tokens: int,
        provider: str | None = None,
        model: str | None = None,
    ) -> None:
        """No-op token tracking."""
        pass


class NullStreamingHandler:
    """Null implementation of streaming handler."""

    async def handle_stream(
        self,
        response: httpx.Response,
        request_context: "RequestContext | None" = None,
    ) -> AsyncIterator[bytes]:
        """Return empty stream."""
        # Make this a proper async generator
        for _ in []:
            yield b""

    def create_streaming_response(
        self,
        stream: AsyncIterator[bytes],
        headers: dict[str, str] | None = None,
    ) -> Response:
        """Create empty response."""
        from starlette.responses import Response

        return Response(content=b"", headers=headers or {})

    async def handle_streaming_request(
        self,
        method: str,
        url: str,
        headers: dict[str, str],
        body: bytes,
        handler_config: Any,
        request_context: Any,
        client_config: dict[str, Any] | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> Any:
        """Null implementation - returns a simple error response."""
        # For null implementation, return a regular response instead of trying to stream
        from starlette.responses import JSONResponse

        return JSONResponse(
            content={"error": "Streaming handler not available"},
            status_code=503,  # Service Unavailable
            headers={"X-Error": "NullStreamingHandler"},
        )
