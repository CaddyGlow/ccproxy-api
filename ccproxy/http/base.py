"""Base HTTP handler abstraction for better separation of concerns."""

from abc import ABC, abstractmethod
from typing import Any, Protocol, runtime_checkable

from starlette.responses import Response, StreamingResponse

from ccproxy.services.handler_config import HandlerConfig
from ccproxy.streaming import DeferredStreaming


@runtime_checkable
class HTTPRequestHandler(Protocol):
    """Protocol for HTTP request handlers."""

    async def handle_request(
        self,
        method: str,
        url: str,
        headers: dict[str, str],
        body: bytes,
        handler_config: HandlerConfig,
        is_streaming: bool = False,
        streaming_handler: Any | None = None,
        request_context: dict[str, Any] | None = None,
    ) -> Response | StreamingResponse | DeferredStreaming:
        """Handle an HTTP request."""
        ...

    async def prepare_request(
        self,
        request_body: bytes,
        handler_config: HandlerConfig,
        auth_headers: dict[str, str] | None = None,
        request_headers: dict[str, str] | None = None,
        **extra_kwargs: Any,
    ) -> tuple[bytes, dict[str, str], bool]:
        """Prepare request for sending."""
        ...


class BaseHTTPHandler(ABC):
    """Abstract base class for HTTP handlers with common functionality."""

    @abstractmethod
    async def handle_request(
        self,
        method: str,
        url: str,
        headers: dict[str, str],
        body: bytes,
        handler_config: HandlerConfig,
        **kwargs: Any,
    ) -> Response | StreamingResponse | DeferredStreaming:
        """Handle an HTTP request.

        Args:
            method: HTTP method
            url: Target URL
            headers: Request headers
            body: Request body
            handler_config: Handler configuration
            **kwargs: Additional handler-specific arguments

        Returns:
            Response or StreamingResponse
        """
        pass

    @abstractmethod
    async def prepare_request(
        self,
        request_body: bytes,
        handler_config: HandlerConfig,
        **kwargs: Any,
    ) -> tuple[bytes, dict[str, str], bool]:
        """Prepare request for sending.

        Args:
            request_body: Original request body
            handler_config: Handler configuration
            **kwargs: Additional preparation parameters

        Returns:
            Tuple of (transformed_body, headers, is_streaming)
        """
        pass

    async def cleanup(self) -> None:
        """Cleanup handler resources.

        Default implementation does nothing.
        Override in subclasses if cleanup is needed.
        """
        return None
