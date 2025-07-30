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

import structlog

from ccproxy.claude_sdk.pool import ClaudeSDKClientPool, PoolConfig
from ccproxy.claude_sdk.session_pool import SessionPool, SessionPoolConfig
from ccproxy.config.settings import Settings


logger = structlog.get_logger(__name__)


MetricsFactory: TypeAlias = Callable[[], Any | None]


class PoolManager:
    """Manages the lifecycle of the ClaudeSDKClientPool with dependency injection."""

    def __init__(
        self,
        settings: Settings | None = None,
        metrics_factory: MetricsFactory | None = None,
    ) -> None:
        """Initialize PoolManager with optional settings and metrics factory.

        Args:
            settings: Optional settings containing session pool configuration
            metrics_factory: Optional callable that returns a metrics instance.
                           If None, no metrics will be used.
        """
        import structlog

        logger = structlog.get_logger(__name__)

        self._settings = settings
        self._pool: ClaudeSDKClientPool | None = None
        self._session_pool: SessionPool | None = None
        self._lock = asyncio.Lock()
        self._metrics_factory = metrics_factory

        # Initialize session pool if enabled
        session_pool_enabled = self._should_enable_session_pool()
        logger.debug(
            "pool_manager_init",
            has_settings=bool(settings),
            has_metrics_factory=bool(metrics_factory),
            session_pool_enabled=session_pool_enabled,
        )

        if session_pool_enabled:
            session_config = self._create_session_pool_config()
            self._session_pool = SessionPool(session_config)
            logger.info(
                "pool_manager_session_pool_initialized",
                session_ttl=session_config.session_ttl,
                max_sessions=session_config.max_sessions,
                cleanup_interval=session_config.cleanup_interval,
            )
        else:
            logger.debug(
                "pool_manager_session_pool_skipped",
                reason="session_pool_disabled_in_settings",
            )

    def _should_enable_session_pool(self) -> bool:
        """Check if session pool should be enabled."""
        import structlog

        logger = structlog.get_logger(__name__)

        if not self._settings:
            logger.debug("session_pool_check", decision="no_settings", enabled=False)
            return False

        if not hasattr(self._settings, "claude"):
            logger.debug(
                "session_pool_check", decision="no_claude_settings", enabled=False
            )
            return False

        session_pool_settings = getattr(self._settings.claude, "session_pool", None)
        if not session_pool_settings:
            logger.debug(
                "session_pool_check", decision="no_session_pool_settings", enabled=False
            )
            return False

        enabled = getattr(session_pool_settings, "enabled", False)
        logger.debug("session_pool_check", decision="settings_check", enabled=enabled)
        return enabled

    def _create_session_pool_config(self) -> SessionPoolConfig:
        """Create session pool configuration from settings."""
        if not self._settings or not hasattr(self._settings, "claude"):
            return SessionPoolConfig()

        session_settings = getattr(self._settings.claude, "session_pool", None)
        if not session_settings:
            return SessionPoolConfig()

        return SessionPoolConfig(
            enabled=getattr(session_settings, "enabled", True),
            session_ttl=getattr(session_settings, "session_ttl", 3600),
            max_sessions=getattr(session_settings, "max_sessions", 1000),
            cleanup_interval=getattr(session_settings, "cleanup_interval", 300),
            idle_threshold=getattr(session_settings, "idle_threshold", 600),
            connection_recovery=getattr(session_settings, "connection_recovery", True),
        )

    async def start(self) -> None:
        """Start the pool manager and session pool."""
        if self._session_pool:
            await self._session_pool.start()

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
        """Gracefully shuts down the managed pools.

        This method is idempotent - calling it multiple times is safe.
        """
        async with self._lock:
            # Close regular pool
            if self._pool:
                await self._pool.stop()
                self._pool = None

            # Close session pool
            if self._session_pool:
                await self._session_pool.stop()
                self._session_pool = None

    async def get_session_client(
        self,
        session_id: str,
        options: Any,  # ClaudeCodeOptions type - avoiding import cycle
    ) -> Any:  # SessionContext type - avoiding import cycle
        """Get session-aware client."""
        import structlog

        from ccproxy.core.errors import ClaudeProxyError

        logger = structlog.get_logger(__name__)
        logger.debug(
            "pool_manager_get_session_client",
            session_id=session_id,
            has_session_pool=bool(self._session_pool),
        )

        if not self._session_pool:
            logger.error(
                "pool_manager_session_pool_unavailable",
                session_id=session_id,
            )
            raise ClaudeProxyError(
                message="Session pool not available",
                error_type="configuration_error",
                status_code=500,
            )

        return await self._session_pool.get_session_client(session_id, options)

    async def interrupt_session(self, session_id: str) -> bool:
        """Interrupt a specific session due to client disconnection.

        Args:
            session_id: The session ID to interrupt

        Returns:
            True if session was found and interrupted, False otherwise
        """
        if not self._session_pool:
            logger.warning(
                "pool_manager_interrupt_session_no_pool",
                session_id=session_id,
            )
            return False

        logger.info(
            "pool_manager_interrupt_session",
            session_id=session_id,
        )

        return await self._session_pool.interrupt_session(session_id)

    async def interrupt_all_sessions(self) -> int:
        """Interrupt all active sessions (for shutdown or emergency cleanup).

        Returns:
            Number of sessions that were interrupted
        """
        if not self._session_pool:
            logger.warning("pool_manager_interrupt_all_no_pool")
            return 0

        logger.info("pool_manager_interrupt_all_sessions")
        return await self._session_pool.interrupt_all_sessions()

    def get_session_pool_stats(self) -> dict[str, Any]:
        """Get session pool statistics."""
        if not self._session_pool:
            return {"enabled": False}
        return self._session_pool.get_stats()

    def reset_for_testing(self) -> None:
        """Synchronous reset for test environments.

        Warning:
            This method should only be used in tests. It does not properly
            shut down the pool - use shutdown() for production code.
        """
        self._pool = None
        self._session_pool = None

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

                # Try to get current settings for session pool support
                settings = None
                try:
                    from ccproxy.config.settings import get_settings

                    settings = get_settings()
                except ImportError:
                    # Settings not available, continue without them
                    pass

                _default_pool_manager = PoolManager(
                    settings=settings, metrics_factory=metrics_factory
                )

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
