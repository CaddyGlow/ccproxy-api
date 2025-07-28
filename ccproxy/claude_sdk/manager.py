"""
Claude SDK Pool Manager - Eliminates global state with dependency injection architecture.

This module provides a PoolManager class that encapsulates pool lifecycle management
using dependency injection patterns, replacing the problematic global pool functions.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable

# Type alias for metrics factory function
from typing import Any, TypeAlias

from ccproxy.claude_sdk.pool import ClaudeSDKClientPool, PoolConfig


MetricsFactory: TypeAlias = Callable[[], Any | None]


class PoolManager:
    """Manages the lifecycle of the ClaudeSDKClientPool with dependency injection."""

    def __init__(self, metrics_factory: MetricsFactory | None = None) -> None:
        """Initialize PoolManager with optional metrics factory for dependency injection.

        Args:
            metrics_factory: Optional callable that returns a metrics instance.
                           If None, no metrics will be used.
        """
        self._pool: ClaudeSDKClientPool | None = None
        self._lock = asyncio.Lock()
        self._metrics_factory = metrics_factory

    async def get_pool(self, config: PoolConfig | None = None) -> ClaudeSDKClientPool:
        """Get the pool instance, creating it if it doesn't exist. Async-safe.

        Args:
            config: Optional pool configuration. If None, uses defaults.

        Returns:
            The managed ClaudeSDKClientPool instance.

        Note:
            This method is async-safe and will only create one pool instance
            even if called concurrently.
        """
        async with self._lock:
            if self._pool is None:
                # Get metrics instance via dependency injection
                metrics_instance = None
                if self._metrics_factory:
                    metrics_instance = self._metrics_factory()

                # Create and start the pool
                self._pool = ClaudeSDKClientPool(
                    config=config, metrics=metrics_instance
                )
                await self._pool.start()

            return self._pool

    async def shutdown(self) -> None:
        """Gracefully shuts down the managed pool.

        This method is idempotent - calling it multiple times is safe.
        """
        async with self._lock:
            if self._pool:
                await self._pool.stop()
                self._pool = None

    def reset_for_testing(self) -> None:
        """Synchronous reset for test environments.

        Warning:
            This method should only be used in tests. It does not properly
            shut down the pool - use shutdown() for production code.
        """
        self._pool = None

    @property
    def is_active(self) -> bool:
        """Check if the pool manager has an active pool."""
        return self._pool is not None


# Service Locator Pattern (async-safe)
_default_pool_manager: PoolManager | None = None
_manager_lock = asyncio.Lock()


async def get_pool_manager() -> PoolManager:
    """Safely get the default PoolManager instance.

    This function implements the service locator pattern with proper async safety.
    It will create a default PoolManager on first access.

    Returns:
        The default PoolManager instance.
    """
    global _default_pool_manager

    if _default_pool_manager is None:
        async with _manager_lock:
            # Double-check pattern for async safety
            if _default_pool_manager is None:
                # Try to get metrics factory, fallback to None if not available
                metrics_factory = None
                try:
                    from ccproxy.observability.metrics import get_metrics

                    metrics_factory = get_metrics
                except ImportError:
                    # No metrics available, continue without them
                    pass

                _default_pool_manager = PoolManager(metrics_factory=metrics_factory)

    return _default_pool_manager


def set_pool_manager(manager: PoolManager) -> None:
    """Inject a specific PoolManager instance. Primarily for testing.

    Args:
        manager: The PoolManager instance to use as the default.

    Warning:
        This function bypasses async safety and should primarily be used
        in test setup where you control the execution context.
    """
    global _default_pool_manager
    _default_pool_manager = manager


async def reset_pool_manager() -> None:
    """Resets the global manager state. For testing.

    This function properly shuts down any existing pool before resetting.
    """
    global _default_pool_manager

    async with _manager_lock:
        if _default_pool_manager:
            await _default_pool_manager.shutdown()
        _default_pool_manager = None


def reset_pool_manager_sync() -> None:
    """Synchronous reset for test environments.

    Warning:
        This does not properly shut down pools. Use reset_pool_manager()
        for production code.
    """
    global _default_pool_manager
    _default_pool_manager = None
