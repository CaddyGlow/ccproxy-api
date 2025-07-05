"""Singleton pool manager for Claude instance connection pool."""

import logging
from typing import Any

from claude_code_proxy.config.settings import Settings
from claude_code_proxy.services.claude_pool import ClaudeInstancePool, PooledConnection


logger = logging.getLogger(__name__)


class PoolManager:
    """
    Singleton manager for the Claude instance connection pool.

    This class ensures only one pool instance exists throughout the
    application lifecycle and provides a centralized interface for
    pool operations.
    """

    _instance: "PoolManager | None" = None
    _pool: ClaudeInstancePool | None = None

    def __new__(cls) -> "PoolManager":
        """Ensure singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initialize the pool manager (only runs once due to singleton)."""
        # Initialization only happens once
        if not hasattr(self, "_initialized"):
            self._initialized = True
            self._settings: Settings | None = None

    def reset(self) -> None:
        """Reset the pool manager state (for testing purposes)."""
        self._pool = None
        self._settings = None
        logger.debug("[POOL_MANAGER] Pool manager state reset")

    def configure(self, settings: Settings) -> None:
        """
        Configure the pool manager with settings.

        Args:
            settings: Application settings containing pool configuration
        """
        self._settings = settings

        if not settings.pool_settings.enabled:
            logger.info("[POOL_MANAGER] Connection pooling is disabled")
            self._pool = None
            return

        # Create pool if enabled
        if self._pool is None:
            self._pool = ClaudeInstancePool(
                min_size=settings.pool_settings.min_size,
                max_size=settings.pool_settings.max_size,
                idle_timeout=settings.pool_settings.idle_timeout,
                health_check_interval=settings.pool_settings.health_check_interval,
                acquire_timeout=settings.pool_settings.acquire_timeout,
            )
            logger.info(
                f"[POOL_MANAGER] Claude instance pool configured "
                f"(min: {settings.pool_settings.min_size}, max: {settings.pool_settings.max_size})"
            )

    async def initialize(self) -> None:
        """Initialize the pool (called on application startup)."""
        if (
            self._pool
            and self._settings
            and self._settings.pool_settings.warmup_on_startup
        ):
            await self._pool.initialize()
            logger.info("[POOL_MANAGER] Claude instance pool initialized with warmup")

    async def acquire_client(self) -> tuple[Any, PooledConnection | None]:
        """
        Acquire a Claude client from the pool or create a new one.

        Returns:
            Tuple of (ClaudeClient, PooledConnection or None if pooling disabled)
        """
        if self._pool is None:
            # Pooling disabled, create a new client directly
            from claude_code_proxy.services.claude_client import ClaudeClient

            logger.debug("[POOL_MANAGER] Creating new client (pooling disabled)")
            return ClaudeClient(), None

        # Get from pool
        logger.debug("[POOL_MANAGER] Acquiring client from pool")
        connection = await self._pool.acquire()
        logger.info(
            f"[POOL_MANAGER] Acquired pooled connection {connection.id} "
            f"(use_count: {connection.use_count})"
        )
        return connection.client, connection

    async def release_client(self, connection: PooledConnection | None) -> None:
        """
        Release a client back to the pool.

        Args:
            connection: The pooled connection to release (None if pooling disabled)
        """
        if self._pool and connection:
            logger.debug(
                f"[POOL_MANAGER] Releasing connection {connection.id} back to pool"
            )
            await self._pool.release(connection)
        elif not self._pool:
            logger.debug("[POOL_MANAGER] No release needed (pooling disabled)")

    async def shutdown(self) -> None:
        """Shutdown the pool (called on application shutdown)."""
        if self._pool:
            await self._pool.shutdown()
            self._pool = None
            logger.info("[POOL_MANAGER] Claude instance pool shut down")

    def get_stats(self) -> dict[str, Any]:
        """Get pool statistics."""
        if self._pool:
            return self._pool.get_stats()
        return {"pool_enabled": False}

    @property
    def is_enabled(self) -> bool:
        """Check if pooling is enabled."""
        return self._pool is not None


# Global pool manager instance
pool_manager = PoolManager()
