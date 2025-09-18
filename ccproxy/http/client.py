"""Centralized HTTP client configuration and abstractions for CCProxy.

This module provides:
- HTTP client factory with optimized configuration for proxy use cases
- Generic HTTP client abstractions for pure forwarding without business logic
- Lifecycle managed by the ServiceContainer
"""

import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx
import structlog

from ccproxy.config.settings import Settings
from ccproxy.http.hooks import HookableHTTPClient


logger = structlog.get_logger(__name__)


if TYPE_CHECKING:
    import httpx


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

    # Note: legacy HTTPXClient and BaseProxyClient removed in favor of using
    # HTTPClientFactory + httpx.AsyncClient directly.


class HTTPClientFactory:
    """Factory for creating optimized HTTP clients.

    Provides centralized configuration for HTTP clients with:
    - Consistent timeout/retry configuration
    - Unified connection limits
    - HTTP/2 multiplexing for non-streaming endpoints
    - Centralized observability hooks (via HookableHTTPClient)
    """

    @staticmethod
    def create_client(
        *,
        settings: Settings | None = None,
        timeout_connect: float = 5.0,
        timeout_read: float = 240.0,  # Long timeout for streaming
        max_keepalive_connections: int = 100,  # For non-streaming endpoints
        max_connections: int = 1000,  # High limit for concurrent streams
        http2: bool = True,  # Enable multiplexing (requires httpx[http2])
        verify: bool | str = True,
        hook_manager: Any | None = None,
        **kwargs: Any,
    ) -> httpx.AsyncClient:
        """Create an optimized HTTP client with recommended configuration.

        Args:
            settings: Optional settings object for additional configuration
            timeout_connect: Connection timeout in seconds
            timeout_read: Read timeout in seconds (long for streaming)
            max_keepalive_connections: Max keep-alive connections for reuse
            max_connections: Max total concurrent connections
            http2: Enable HTTP/2 multiplexing
            verify: SSL verification (True/False or path to CA bundle)
            hook_manager: Optional HookManager for request/response interception
            **kwargs: Additional httpx.AsyncClient arguments

        Returns:
            Configured httpx.AsyncClient instance
        """
        # Get proxy configuration from environment
        proxy = get_proxy_url()

        # Get SSL context configuration
        if isinstance(verify, bool) and verify:
            verify = get_ssl_context()

        # Create timeout configuration
        timeout = httpx.Timeout(
            connect=timeout_connect,
            read=timeout_read,
            write=30.0,  # Write timeout
            pool=30.0,  # Pool timeout
        )

        # Create connection limits
        limits = httpx.Limits(
            max_keepalive_connections=max_keepalive_connections,
            max_connections=max_connections,
        )

        # Create transport
        transport = httpx.AsyncHTTPTransport(
            limits=limits,
            http2=http2,
            verify=verify,
            proxy=proxy,
        )

        # Note: Transport wrapping for logging is now handled by the raw_http_logger plugin

        # Handle compression settings
        default_headers = {}
        if settings and hasattr(settings, "http"):
            http_settings = settings.http
            if not http_settings.compression_enabled:
                # Disable compression by setting identity encoding
                # "identity" means no compression
                default_headers["accept-encoding"] = "identity"
            elif http_settings.accept_encoding:
                # Use custom Accept-Encoding value
                default_headers["accept-encoding"] = http_settings.accept_encoding
            # else: let httpx use its default compression handling
        else:
            logger.warning(
                "http_settings_not_found", settings_present=settings is not None
            )

        # Merge headers with any provided in kwargs
        if "headers" in kwargs:
            default_headers.update(kwargs["headers"])
            kwargs["headers"] = default_headers
        elif default_headers:
            kwargs["headers"] = default_headers

        # Merge with any additional kwargs
        client_config = {
            "timeout": timeout,
            "transport": transport,
            **kwargs,
        }

        # Determine effective compression status
        compression_status = "httpx default"
        if "accept-encoding" in default_headers:
            if default_headers["accept-encoding"] == "identity":
                compression_status = "disabled"
            else:
                compression_status = default_headers["accept-encoding"]

        logger.debug(
            "http_client_created",
            timeout_connect=timeout_connect,
            timeout_read=timeout_read,
            max_keepalive_connections=max_keepalive_connections,
            max_connections=max_connections,
            http2=http2,
            has_proxy=proxy is not None,
            has_hooks=hook_manager is not None,
            compression_enabled=settings.http.compression_enabled
            if settings and hasattr(settings, "http")
            else True,
            accept_encoding=compression_status,
        )

        # Create client with or without hook support
        if hook_manager:
            return HookableHTTPClient(hook_manager=hook_manager, **client_config)
        else:
            return httpx.AsyncClient(**client_config)

    @staticmethod
    def create_shared_client(settings: Settings | None = None) -> httpx.AsyncClient:
        """Create an optimized HTTP client.

        Prefer managing lifecycle via ServiceContainer + HTTPPoolManager.
        Kept for compatibility with existing factory call sites.
        """
        return HTTPClientFactory.create_client(settings=settings)

    @staticmethod
    def create_short_lived_client(
        timeout: float = 15.0,
        **kwargs: Any,
    ) -> httpx.AsyncClient:
        """Create a client for short-lived operations like version checks.

        Args:
            timeout: Short timeout for quick operations
            **kwargs: Additional client configuration

        Returns:
            Configured httpx.AsyncClient instance for short operations
        """
        return HTTPClientFactory.create_client(
            timeout_connect=5.0,
            timeout_read=timeout,
            max_keepalive_connections=10,
            max_connections=50,
            **kwargs,
        )

    @staticmethod
    @asynccontextmanager
    async def managed_client(
        settings: Settings | None = None, **kwargs: Any
    ) -> AsyncGenerator[httpx.AsyncClient, None]:
        """Create a managed HTTP client with automatic cleanup.

        This context manager ensures proper cleanup of HTTP clients
        in error cases and provides a clean resource management pattern.

        Args:
            settings: Optional settings for configuration
            **kwargs: Additional client configuration

        Yields:
            Configured httpx.AsyncClient instance

        Example:
            async with HTTPClientFactory.managed_client() as client:
                response = await client.get("https://api.example.com")
        """
        client = HTTPClientFactory.create_client(settings=settings, **kwargs)
        try:
            logger.debug("managed_http_client_created")
            yield client
        finally:
            try:
                await client.aclose()
                logger.debug("managed_http_client_closed")
            except Exception as e:
                logger.warning(
                    "managed_http_client_close_failed",
                    error=str(e),
                    exc_info=e,
                )


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
        return True
