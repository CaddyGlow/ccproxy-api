"""Generic HTTP client abstractions for pure forwarding without business logic."""

from abc import ABC, abstractmethod
from typing import Any


class HTTPClient(ABC):
    """Abstract HTTP client interface for generic HTTP operations."""

    @abstractmethod
    async def request(
        self,
        method: str,
        url: str,
        headers: dict[str, str],
        body: bytes | None = None,
        timeout: float | None = None,
    ) -> tuple[int, dict[str, str], bytes]:
        """Make an HTTP request.

        Args:
            method: HTTP method (GET, POST, etc.)
            url: Target URL
            headers: HTTP headers
            body: Request body (optional)
            timeout: Request timeout in seconds (optional)

        Returns:
            Tuple of (status_code, response_headers, response_body)

        Raises:
            HTTPError: If the request fails
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close any resources held by the HTTP client."""
        pass


class BaseProxyClient:
    """Generic proxy client with no business logic - pure forwarding."""

    def __init__(self, http_client: HTTPClient) -> None:
        """Initialize with an HTTP client.

        Args:
            http_client: The HTTP client to use for requests
        """
        self.http_client = http_client

    async def forward(
        self,
        method: str,
        url: str,
        headers: dict[str, str],
        body: bytes | None = None,
        timeout: float | None = None,
    ) -> tuple[int, dict[str, str], bytes]:
        """Forward an HTTP request without any transformations.

        Args:
            method: HTTP method
            url: Target URL
            headers: HTTP headers
            body: Request body (optional)
            timeout: Request timeout in seconds (optional)

        Returns:
            Tuple of (status_code, response_headers, response_body)

        Raises:
            HTTPError: If the request fails
        """
        return await self.http_client.request(method, url, headers, body, timeout)

    async def close(self) -> None:
        """Close any resources held by the proxy client."""
        await self.http_client.close()


class HTTPError(Exception):
    """Base exception for HTTP client errors."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        """Initialize HTTP error.

        Args:
            message: Error message
            status_code: HTTP status code (optional)
        """
        super().__init__(message)
        self.status_code = status_code


class HTTPTimeoutError(HTTPError):
    """Exception raised when HTTP request times out."""

    def __init__(self, message: str = "Request timed out") -> None:
        """Initialize timeout error.

        Args:
            message: Error message
        """
        super().__init__(message, status_code=408)


class HTTPConnectionError(HTTPError):
    """Exception raised when HTTP connection fails."""

    def __init__(self, message: str = "Connection failed") -> None:
        """Initialize connection error.

        Args:
            message: Error message
        """
        super().__init__(message, status_code=503)
