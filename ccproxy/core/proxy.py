"""Core proxy abstractions for handling HTTP and WebSocket connections."""

from abc import ABC, abstractmethod
from typing import Any, Optional, Protocol, runtime_checkable

from ccproxy.core.types import ProxyRequest, ProxyResponse


class BaseProxy(ABC):
    """Abstract base class for all proxy implementations."""

    @abstractmethod
    async def forward(self, request: ProxyRequest) -> ProxyResponse:
        """Forward a request and return the response.

        Args:
            request: The proxy request to forward

        Returns:
            The proxy response

        Raises:
            ProxyError: If the request cannot be forwarded
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close any resources held by the proxy."""
        pass


class HTTPProxy(BaseProxy):
    """HTTP proxy implementation placeholder."""

    async def forward(self, request: ProxyRequest) -> ProxyResponse:
        """Forward an HTTP request."""
        raise NotImplementedError("HTTPProxy.forward not yet implemented")

    async def close(self) -> None:
        """Close HTTP proxy resources."""
        pass


class WebSocketProxy(BaseProxy):
    """WebSocket proxy implementation placeholder."""

    async def forward(self, request: ProxyRequest) -> ProxyResponse:
        """Forward a WebSocket request."""
        raise NotImplementedError("WebSocketProxy.forward not yet implemented")

    async def close(self) -> None:
        """Close WebSocket proxy resources."""
        pass


@runtime_checkable
class ProxyProtocol(Protocol):
    """Protocol defining the proxy interface."""

    async def forward(self, request: ProxyRequest) -> ProxyResponse:
        """Forward a request and return the response."""
        ...

    async def close(self) -> None:
        """Close any resources held by the proxy."""
        ...
