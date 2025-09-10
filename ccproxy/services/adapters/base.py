"""Base adapter for provider plugins."""

from abc import ABC, abstractmethod
from typing import Any

from fastapi import Request
from starlette.responses import Response, StreamingResponse

from ccproxy.streaming import DeferredStreaming


class BaseAdapter(ABC):
    """Base adapter for provider-specific request handling."""

    def __init__(self, config: Any, **kwargs: Any) -> None:
        """Initialize the base adapter.

        Args:
            config: Plugin configuration
            **kwargs: Additional keyword arguments for subclasses
        """
        self.config = config

    @abstractmethod
    async def handle_request(
        self, request: Request
    ) -> Response | StreamingResponse | DeferredStreaming:
        """Handle a provider-specific request.

        Args:
            request: FastAPI request object with endpoint and method in request.state.context

        Returns:
            Response, StreamingResponse, or DeferredStreaming object
        """
        ...

    @abstractmethod
    async def handle_streaming(
        self, request: Request, endpoint: str, **kwargs: Any
    ) -> StreamingResponse | DeferredStreaming:
        """Handle a streaming request.

        Args:
            request: FastAPI request object
            endpoint: Target endpoint path
            **kwargs: Additional provider-specific arguments

        Returns:
            StreamingResponse or DeferredStreaming object
        """
        ...

    async def validate_request(
        self, request: Request, endpoint: str
    ) -> dict[str, Any] | None:
        """Validate request before processing.

        Args:
            request: FastAPI request object
            endpoint: Target endpoint path

        Returns:
            Validation result or None if valid
        """
        return None

    async def transform_request(self, request_data: dict[str, Any]) -> dict[str, Any]:
        """Transform request data if needed.

        Args:
            request_data: Original request data

        Returns:
            Transformed request data
        """
        return request_data

    async def transform_response(self, response_data: dict[str, Any]) -> dict[str, Any]:
        """Transform response data if needed.

        Args:
            response_data: Original response data

        Returns:
            Transformed response data
        """
        return response_data

    @abstractmethod
    async def cleanup(self) -> None:
        """Cleanup adapter resources.

        This method should be overridden by concrete adapters to clean up
        any resources like HTTP clients, sessions, or background tasks.
        Called during application shutdown.
        """
        ...
