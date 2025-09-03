"""
Claude SDK Session Manager - Pure dependency injection architecture.

This module provides a SessionManager class that encapsulates session pool lifecycle
management using dependency injection patterns without any global state.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable

# Type alias for metrics factory function
from typing import Any, TypeAlias

from claude_code_sdk import ClaudeCodeOptions

from ccproxy.core.errors import ClaudeProxyError
from ccproxy.core.logging import get_plugin_logger

from .config import ClaudeSDKSettings, SessionPoolSettings
from .session_client import SessionClient
from .session_pool import SessionPool


# Type alias for metrics factory function
MetricsFactory: TypeAlias = Callable[[], Any]

logger = get_plugin_logger()


class SessionManager:
    """Manages the lifecycle of session-based Claude SDK clients with dependency injection."""

    def __init__(
        self,
        config: ClaudeSDKSettings,
        metrics_factory: MetricsFactory | None = None,
    ) -> None:
        """Initialize SessionManager with optional settings and metrics factory.

        Args:
            config: Plugin-specific configuration for Claude SDK
            metrics_factory: Optional callable that returns a metrics instance.
                           If None, no metrics will be used.
        """

        self.config = config
        self._session_pool: SessionPool | None = None
        self._lock = asyncio.Lock()
        self._metrics_factory = metrics_factory

        # Initialize session pool if enabled
        session_pool_enabled = self._should_enable_session_pool()
        logger.debug(
            "session_manager_init",
            has_config=bool(config),
            has_metrics_factory=bool(metrics_factory),
            session_pool_enabled=session_pool_enabled,
        )

        if session_pool_enabled:
            # Convert core settings to plugin settings
            core_pool_settings = self.config.sdk_session_pool
            if core_pool_settings is None:
                logger.debug("session_pool_disabled", reason="no_settings")
                return

            plugin_pool_settings = SessionPoolSettings(
                enabled=core_pool_settings.enabled,
                session_ttl=core_pool_settings.session_ttl,
                max_sessions=core_pool_settings.max_sessions,
                cleanup_interval=getattr(core_pool_settings, "cleanup_interval", 300),
                idle_threshold=getattr(core_pool_settings, "idle_threshold", 300),
                connection_recovery=getattr(
                    core_pool_settings, "connection_recovery", True
                ),
                stream_first_chunk_timeout=getattr(
                    core_pool_settings, "stream_first_chunk_timeout", 8
                ),
                stream_ongoing_timeout=getattr(
                    core_pool_settings, "stream_ongoing_timeout", 60
                ),
            )
            self._session_pool = SessionPool(plugin_pool_settings)
            logger.debug(
                "session_manager_session_pool_initialized",
                session_ttl=self._session_pool.config.session_ttl,
                max_sessions=self._session_pool.config.max_sessions,
                cleanup_interval=self._session_pool.config.cleanup_interval,
            )
        else:
            logger.debug(
                "session_manager_session_pool_skipped",
                reason="session_pool_disabled_in_settings",
            )

    def _should_enable_session_pool(self) -> bool:
        """Check if session pool should be enabled."""

        if not self.config:
            logger.debug("session_pool_check", decision="no_config", enabled=False)
            return False

        session_pool_settings = getattr(self.config, "sdk_session_pool", None)
        if not session_pool_settings:
            logger.debug(
                "session_pool_check", decision="no_session_pool_settings", enabled=False
            )
            return False

        enabled = getattr(session_pool_settings, "enabled", False)
        logger.debug("session_pool_check", decision="settings_check", enabled=enabled)
        return enabled

    async def start(self) -> None:
        """Start the session manager and session pool."""
        if self._session_pool:
            await self._session_pool.start()

    async def shutdown(self) -> None:
        """Gracefully shuts down the session pool.

        This method is idempotent - calling it multiple times is safe.
        """
        async with self._lock:
            # Close session pool
            if self._session_pool:
                await self._session_pool.stop()
                self._session_pool = None

    async def get_session_client(
        self,
        session_id: str,
        options: ClaudeCodeOptions,
    ) -> SessionClient:
        """Get session-aware client."""

        logger.debug(
            "session_manager_get_session_client",
            session_id=session_id,
            has_session_pool=bool(self._session_pool),
        )

        if not self._session_pool:
            logger.error(
                "session_manager_session_pool_unavailable",
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
                "session_manager_interrupt_session_no_pool",
                session_id=session_id,
            )
            return False

        logger.debug(
            "session_manager_interrupt_session",
            session_id=session_id,
        )

        return await self._session_pool.interrupt_session(session_id)

    async def interrupt_all_sessions(self) -> int:
        """Interrupt all active sessions (for shutdown or emergency cleanup).

        Returns:
            Number of sessions that were interrupted
        """
        if not self._session_pool:
            logger.warning("session_manager_interrupt_all_no_pool")
            return 0

        logger.debug("session_manager_interrupt_all_sessions")
        return await self._session_pool.interrupt_all_sessions()

    async def get_session_pool_stats(self) -> dict[str, Any]:
        """Get session pool statistics."""
        if not self._session_pool:
            return {"enabled": False}
        return await self._session_pool.get_stats()

    def reset_for_testing(self) -> None:
        """Synchronous reset for test environments.

        Warning:
            This method should only be used in tests. It does not properly
            shut down the session pool - use shutdown() for production code.
        """
        self._session_pool = None

    @property
    def is_active(self) -> bool:
        """Check if the session manager has an active session pool."""
        return self._session_pool is not None

    async def has_session_pool(self) -> bool:
        """Check if session pool is available and enabled."""
        return self._session_pool is not None and self._session_pool.config.enabled

    async def has_session(self, session_id: str) -> bool:
        """Check if a session exists in the session pool.

        Args:
            session_id: The session ID to check

        Returns:
            True if session exists, False otherwise
        """
        if not self._session_pool:
            return False
        return await self._session_pool.has_session(session_id)
