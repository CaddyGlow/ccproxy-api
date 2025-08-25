"""HTTP connection pool manager for efficient client reuse."""

import asyncio
from collections.abc import MutableMapping
from typing import Any

import httpx
import structlog

from ccproxy.config.constants import (
    HTTP_CLIENT_POOL_SIZE,
    HTTP_CLIENT_TIMEOUT,
    HTTP_STREAMING_TIMEOUT,
)


logger = structlog.get_logger(__name__)


class ConnectionPoolManager:
    """Manages HTTP connection pools for different configurations."""

    def __init__(
        self,
        default_timeout: float = HTTP_CLIENT_TIMEOUT,
        pool_size: int = HTTP_CLIENT_POOL_SIZE,
    ) -> None:
        """Initialize the connection pool manager.

        Args:
            default_timeout: Default request timeout in seconds
            pool_size: Maximum number of connections per pool
        """
        self.default_timeout = default_timeout
        self.pool_size = pool_size
        self._pools: MutableMapping[str, httpx.AsyncClient] = {}
        self._lock = asyncio.Lock()
        self.logger = logger

    async def get_client(
        self,
        base_url: str | None = None,
        timeout: float | None = None,
        proxy: str | None = None,
        verify: bool | str = True,
    ) -> httpx.AsyncClient:
        """Get or create an HTTP client for the given configuration.

        Args:
            base_url: Base URL for the client
            timeout: Request timeout in seconds
            proxy: HTTP proxy URL
            verify: SSL verification configuration

        Returns:
            HTTPX AsyncClient instance
        """
        # Create a unique key for this configuration
        key = self._create_pool_key(base_url, timeout, proxy, verify)

        # Check if we already have a client for this configuration
        if key in self._pools:
            return self._pools[key]

        # Create new client with lock to prevent race conditions
        async with self._lock:
            # Double-check after acquiring lock
            if key in self._pools:
                return self._pools[key]

            # Create new client
            client = self._create_client(base_url, timeout, proxy, verify)
            self._pools[key] = client

            self.logger.debug(
                "connection_pool_created",
                pool_key=key,
                base_url=base_url,
                timeout=timeout or self.default_timeout,
                pool_size=self.pool_size,
            )

            return client

    def _create_pool_key(
        self,
        base_url: str | None,
        timeout: float | None,
        proxy: str | None,
        verify: bool | str,
    ) -> str:
        """Create a unique key for the connection pool.

        Args:
            base_url: Base URL for the client
            timeout: Request timeout
            proxy: HTTP proxy URL
            verify: SSL verification configuration

        Returns:
            Unique string key for this configuration
        """
        parts = [
            f"url:{base_url or 'default'}",
            f"timeout:{timeout or self.default_timeout}",
            f"proxy:{proxy or 'none'}",
            f"verify:{verify}",
        ]
        return "|".join(parts)

    def _create_client(
        self,
        base_url: str | None,
        timeout: float | None,
        proxy: str | None,
        verify: bool | str,
    ) -> httpx.AsyncClient:
        """Create a new HTTP client with the given configuration.

        Args:
            base_url: Base URL for the client
            timeout: Request timeout
            proxy: HTTP proxy URL
            verify: SSL verification configuration

        Returns:
            New HTTPX AsyncClient instance
        """
        limits = httpx.Limits(
            max_keepalive_connections=self.pool_size,
            max_connections=self.pool_size * 2,
            keepalive_expiry=30,  # Keep connections alive for 30 seconds
        )

        client_kwargs: dict[str, Any] = {
            "limits": limits,
            "timeout": httpx.Timeout(timeout or self.default_timeout),
            "verify": verify,
            "http2": True,  # Enable HTTP/2 for better performance
            "follow_redirects": False,  # Don't follow redirects by default
        }

        if base_url:
            client_kwargs["base_url"] = base_url

        if proxy:
            client_kwargs["proxy"] = proxy

        return httpx.AsyncClient(**client_kwargs)

    async def get_streaming_client(
        self,
        base_url: str | None = None,
        proxy: str | None = None,
        verify: bool | str = True,
    ) -> httpx.AsyncClient:
        """Get or create an HTTP client optimized for streaming.

        Args:
            base_url: Base URL for the client
            proxy: HTTP proxy URL
            verify: SSL verification configuration

        Returns:
            HTTPX AsyncClient instance configured for streaming
        """
        return await self.get_client(
            base_url=base_url,
            timeout=HTTP_STREAMING_TIMEOUT,
            proxy=proxy,
            verify=verify,
        )

    async def close_all(self) -> None:
        """Close all connection pools."""
        async with self._lock:
            close_tasks = []
            for key, client in self._pools.items():
                close_tasks.append(self._close_client(key, client))

            if close_tasks:
                await asyncio.gather(*close_tasks, return_exceptions=True)

            self._pools.clear()
            self.logger.info("connection_pools_closed", pool_count=len(close_tasks))

    async def _close_client(self, key: str, client: httpx.AsyncClient) -> None:
        """Close a single HTTP client.

        Args:
            key: Pool key for logging
            client: Client to close
        """
        try:
            await client.aclose()
            self.logger.debug("connection_pool_closed", pool_key=key)
        except Exception as e:
            self.logger.error(
                "connection_pool_close_failed",
                pool_key=key,
                error=str(e),
                exc_info=e,
            )

    async def cleanup_idle(self) -> None:
        """Clean up idle connection pools.

        This method can be called periodically to close
        clients that haven't been used recently.
        """
        # TODO: Implement idle cleanup with last-used tracking
        pass

    @property
    def pool_count(self) -> int:
        """Get the current number of connection pools."""
        return len(self._pools)

    @property
    def pool_keys(self) -> list[str]:
        """Get the list of current pool keys."""
        return list(self._pools.keys())
