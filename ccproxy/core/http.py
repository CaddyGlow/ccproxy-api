"""Generic HTTP client abstractions for pure forwarding without business logic."""

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog


logger = structlog.get_logger(__name__)


if TYPE_CHECKING:
    import httpx


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


class HTTPXClient(HTTPClient):
    """HTTPX-based HTTP client implementation."""

    def __init__(
        self,
        timeout: float = 240.0,
        proxy: str | None = None,
        verify: bool | str = True,
    ) -> None:
        """Initialize HTTPX client.

        Args:
            timeout: Request timeout in seconds
            proxy: HTTP proxy URL (optional)
            verify: SSL verification (True/False or path to CA bundle)
        """
        import httpx

        self.timeout = timeout
        self.proxy = proxy
        self.verify = verify
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> "httpx.AsyncClient":
        """Get or create the HTTPX client."""
        if self._client is None:
            import httpx

            self._client = httpx.AsyncClient(
                timeout=self.timeout,
                proxy=self.proxy,
                verify=self.verify,
            )
        return self._client

    async def request(
        self,
        method: str,
        url: str,
        headers: dict[str, str],
        body: bytes | None = None,
        timeout: float | None = None,
    ) -> tuple[int, dict[str, str], bytes]:
        """Make an HTTP request using HTTPX.

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
        import httpx

        try:
            client = await self._get_client()

            # Use provided timeout if available
            if timeout is not None:
                # Create a new client with different timeout if needed
                import httpx

                client = httpx.AsyncClient(
                    timeout=timeout,
                    proxy=self.proxy,
                    verify=self.verify,
                )

            response = await client.request(
                method=method,
                url=url,
                headers=headers,
                content=body,
            )

            # Always return the response, even for error status codes
            # This allows the proxy to forward upstream errors directly
            return (
                response.status_code,
                dict(response.headers),
                response.content,
            )

        except httpx.TimeoutException as e:
            raise HTTPTimeoutError(f"Request timed out: {e}") from e
        except httpx.ConnectError as e:
            raise HTTPConnectionError(f"Connection failed: {e}") from e
        except httpx.HTTPStatusError as e:
            # This shouldn't happen with the default raise_for_status=False
            # but keep it just in case
            raise HTTPError(
                f"HTTP {e.response.status_code}: {e.response.reason_phrase}",
                status_code=e.response.status_code,
            ) from e
        except Exception as e:
            raise HTTPError(f"HTTP request failed: {e}") from e

    async def stream(
        self,
        method: str,
        url: str,
        headers: dict[str, str],
        content: bytes | None = None,
    ) -> Any:
        """Create a streaming HTTP request.

        Args:
            method: HTTP method
            url: Target URL
            headers: HTTP headers
            content: Request body (optional)

        Returns:
            HTTPX streaming response context manager
        """
        client = await self._get_client()
        return client.stream(
            method=method,
            url=url,
            headers=headers,
            content=content,
        )

    async def close(self) -> None:
        """Close the HTTPX client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None


def get_proxy_url() -> str | None:
    """Get proxy URL from environment variables.

    Returns:
        str or None: Proxy URL if any proxy is set
    """
    # Check for standard proxy environment variables
    # For HTTPS requests, prioritize HTTPS_PROXY
    https_proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
    all_proxy = os.environ.get("ALL_PROXY")
    http_proxy = os.environ.get("HTTP_PROXY") or os.environ.get("http_proxy")

    proxy_url = https_proxy or all_proxy or http_proxy

    if proxy_url:
        logger.debug(
            "proxy_configured",
            proxy_url=proxy_url,
            operation="get_proxy_url",
        )

    return proxy_url


def get_ssl_context() -> str | bool:
    """Get SSL context configuration from environment variables.

    Returns:
        SSL verification configuration:
        - Path to CA bundle file
        - True for default verification
        - False to disable verification (insecure)
    """
    # Check for custom CA bundle
    ca_bundle = os.environ.get("REQUESTS_CA_BUNDLE") or os.environ.get("SSL_CERT_FILE")

    # Check if SSL verification should be disabled (NOT RECOMMENDED)
    ssl_verify = os.environ.get("SSL_VERIFY", "true").lower()

    if ca_bundle and Path(ca_bundle).exists():
        logger.info(
            "ssl_ca_bundle_configured",
            ca_bundle_path=ca_bundle,
            operation="get_ssl_context",
        )
        return ca_bundle
    elif ssl_verify in ("false", "0", "no"):
        logger.warning(
            "ssl_verification_disabled",
            ssl_verify_value=ssl_verify,
            operation="get_ssl_context",
            security_warning=True,
        )
        return False
    else:
        logger.debug(
            "ssl_default_verification",
            ssl_verify_value=ssl_verify,
            operation="get_ssl_context",
        )
        return True
