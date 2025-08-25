"""HTTP Connection Pool Manager for CCProxy.

This module provides centralized management of HTTP connection pools,
ensuring efficient resource usage and preventing duplicate client creation.
Implements Phase 2.3 of the refactoring plan.
"""

import asyncio
from typing import Any
from urllib.parse import urlparse

import httpx
import structlog

from ccproxy.config.settings import Settings
from ccproxy.core.http_client import HTTPClientFactory


logger = structlog.get_logger(__name__)


class HTTPPoolManager:
    """Manages HTTP connection pools for different base URLs.

    This manager ensures that:
    - Each unique base URL gets its own optimized connection pool
    - Connection pools are reused across all components
    - Resources are properly cleaned up on shutdown
    - Configuration is consistent across all clients
    """

    def __init__(self, settings: Settings | None = None) -> None:
        """Initialize the HTTP pool manager.

        Args:
            settings: Optional application settings for configuration
        """
        self.settings = settings
        self._pools: dict[str, httpx.AsyncClient] = {}
        self._shared_client: httpx.AsyncClient | None = None
        self._lock = asyncio.Lock()

        logger.debug("http_pool_manager_initialized", category="lifecycle")

    async def get_client(
        self,
        base_url: str | None = None,
        *,
        timeout: float | None = None,
        headers: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> httpx.AsyncClient:
        """Get or create an HTTP client for the specified base URL.

        Args:
            base_url: Optional base URL for the client. If None, returns shared client
            timeout: Optional custom timeout for this client
            headers: Optional default headers for this client
            **kwargs: Additional configuration for the client

        Returns:
            Configured httpx.AsyncClient instance
        """
        # If no base URL, return the shared general-purpose client
        if not base_url:
            return await self.get_shared_client()

        # Normalize the base URL to use as a key
        pool_key = self._normalize_base_url(base_url)

        async with self._lock:
            # Check if we already have a client for this base URL
            if pool_key in self._pools:
                logger.debug(
                    "reusing_existing_pool",
                    base_url=base_url,
                    pool_key=pool_key,
                    category="lifecycle",
                )
                return self._pools[pool_key]

            # Create a new client for this base URL
            logger.info(
                "creating_new_pool",
                base_url=base_url,
                pool_key=pool_key,
            )

            # Build client configuration
            client_config: dict[str, Any] = {
                "base_url": base_url,
            }

            if headers:
                client_config["headers"] = headers

            if timeout is not None:
                client_config["timeout_read"] = timeout

            # Merge with any additional kwargs
            client_config.update(kwargs)

            # Create the client using the factory with HTTP/2 enabled for better multiplexing
            client = HTTPClientFactory.create_client(
                settings=self.settings,
                http2=True,  # Enable HTTP/2 for connection multiplexing
                **client_config,
            )

            # Store in the pool
            self._pools[pool_key] = client

            return client

    async def get_shared_client(self) -> httpx.AsyncClient:
        """Get the shared general-purpose HTTP client.

        This client is used for requests without a specific base URL.

        Returns:
            The shared httpx.AsyncClient instance
        """
        async with self._lock:
            if self._shared_client is None:
                logger.info("creating_shared_client")
                self._shared_client = HTTPClientFactory.create_client(
                    settings=self.settings,
                    http2=True,  # Enable HTTP/2 for shared client
                )
            return self._shared_client

    def get_shared_client_sync(self) -> httpx.AsyncClient:
        """Get or create the shared client synchronously.

        This is used during initialization when we're not in an async context.
        Note: This doesn't use locking, so it should only be called during
        single-threaded initialization.

        Returns:
            The shared httpx.AsyncClient instance
        """
        if self._shared_client is None:
            logger.debug("creating_shared_client_sync")
            self._shared_client = HTTPClientFactory.create_client(
                settings=self.settings,
                http2=False,  # Disable HTTP/2 to ensure logging transport works
            )
        return self._shared_client

    def get_pool_client(self, base_url: str) -> httpx.AsyncClient | None:
        """Get an existing client for a base URL without creating one.

        Args:
            base_url: The base URL to look up

        Returns:
            Existing client or None if not found
        """
        pool_key = self._normalize_base_url(base_url)
        return self._pools.get(pool_key)

    def _normalize_base_url(self, base_url: str) -> str:
        """Normalize a base URL to use as a pool key.

        Args:
            base_url: The base URL to normalize

        Returns:
            Normalized URL suitable for use as a dictionary key
        """
        parsed = urlparse(base_url)
        # Use scheme + netloc as the key (ignore path/query/fragment)
        # This ensures all requests to the same host share a pool
        return f"{parsed.scheme}://{parsed.netloc}"

    async def close_pool(self, base_url: str) -> None:
        """Close and remove a specific connection pool.

        Args:
            base_url: The base URL of the pool to close
        """
        pool_key = self._normalize_base_url(base_url)

        async with self._lock:
            if pool_key in self._pools:
                client = self._pools.pop(pool_key)
                await client.aclose()
                logger.info(
                    "pool_closed",
                    base_url=base_url,
                    pool_key=pool_key,
                )

    async def close_all(self) -> None:
        """Close all connection pools and clean up resources.

        This should be called during application shutdown.
        """
        async with self._lock:
            # Close all URL-specific pools
            for pool_key, client in self._pools.items():
                try:
                    await client.aclose()
                    logger.debug("pool_closed", pool_key=pool_key)
                except Exception as e:
                    logger.error(
                        "pool_close_error",
                        pool_key=pool_key,
                        error=str(e),
                        exc_info=e,
                    )

            self._pools.clear()

            # Close the shared client
            if self._shared_client:
                try:
                    await self._shared_client.aclose()
                    logger.debug("shared_client_closed")
                except Exception as e:
                    logger.error(
                        "shared_client_close_error",
                        error=str(e),
                        exc_info=e,
                    )
                self._shared_client = None

            logger.info("all_pools_closed")

    def get_pool_stats(self) -> dict[str, Any]:
        """Get statistics about the current connection pools.

        Returns:
            Dictionary with pool statistics
        """
        return {
            "total_pools": len(self._pools),
            "pool_keys": list(self._pools.keys()),
            "has_shared_client": self._shared_client is not None,
        }


# Global instance for convenience
_global_pool_manager: HTTPPoolManager | None = None


def get_pool_manager(settings: Settings | None = None) -> HTTPPoolManager:
    """Get the global HTTP pool manager instance.

    Args:
        settings: Optional settings for configuration

    Returns:
        The global HTTPPoolManager instance
    """
    global _global_pool_manager

    if _global_pool_manager is None:
        _global_pool_manager = HTTPPoolManager(settings)
        logger.info("global_pool_manager_created")

    return _global_pool_manager


async def close_global_pool_manager() -> None:
    """Close the global pool manager and clean up resources.

    This should be called during application shutdown.
    """
    global _global_pool_manager

    if _global_pool_manager is not None:
        await _global_pool_manager.close_all()
        _global_pool_manager = None
        logger.info("global_pool_manager_closed")
