"""Session-aware connection pool for persistent Claude SDK connections."""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any

import structlog
from claude_code_sdk import ClaudeCodeOptions

from ccproxy.claude_sdk.session_context import SessionContext, SessionStatus
from ccproxy.core.errors import ClaudeProxyError, ServiceUnavailableError


logger = structlog.get_logger(__name__)


class SessionPoolConfig:
    """Configuration for session pool behavior."""

    def __init__(
        self,
        enabled: bool = True,
        session_ttl: int = 3600,  # 1 hour
        max_sessions: int = 1000,
        cleanup_interval: int = 300,  # 5 minutes
        idle_threshold: int = 600,  # 10 minutes
        connection_recovery: bool = True,
    ):
        self.enabled = enabled
        self.session_ttl = session_ttl
        self.max_sessions = max_sessions
        self.cleanup_interval = cleanup_interval
        self.idle_threshold = idle_threshold
        self.connection_recovery = connection_recovery


class SessionPool:
    """Manages persistent Claude SDK connections by session."""

    def __init__(self, config: SessionPoolConfig | None = None):
        self.config = config or SessionPoolConfig()
        self.sessions: dict[str, SessionContext] = {}
        self.cleanup_task: asyncio.Task[None] | None = None
        self._shutdown = False

    async def start(self) -> None:
        """Start the session pool and cleanup task."""
        if not self.config.enabled:
            return

        logger.info(
            "session_pool_starting",
            max_sessions=self.config.max_sessions,
            ttl=self.config.session_ttl,
            cleanup_interval=self.config.cleanup_interval,
        )

        self.cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def stop(self) -> None:
        """Stop the session pool and cleanup all sessions."""
        self._shutdown = True

        if self.cleanup_task:
            self.cleanup_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.cleanup_task

        # Disconnect all active sessions
        disconnect_tasks = [
            session_ctx.disconnect() for session_ctx in self.sessions.values()
        ]

        if disconnect_tasks:
            await asyncio.gather(*disconnect_tasks, return_exceptions=True)

        self.sessions.clear()
        logger.info("session_pool_stopped")

    async def get_session_client(
        self, session_id: str, options: ClaudeCodeOptions
    ) -> SessionContext:
        """Get or create a session context for the given session_id."""
        logger.debug(
            "session_pool_get_client_start",
            session_id=session_id,
            pool_enabled=self.config.enabled,
            current_sessions=len(self.sessions),
            max_sessions=self.config.max_sessions,
            session_exists=session_id in self.sessions,
        )

        if not self.config.enabled:
            logger.error("session_pool_disabled", session_id=session_id)
            raise ClaudeProxyError(
                message="Session pool is disabled",
                error_type="configuration_error",
                status_code=500,
            )

        # Check session limit
        if (
            session_id not in self.sessions
            and len(self.sessions) >= self.config.max_sessions
        ):
            logger.error(
                "session_pool_at_capacity",
                session_id=session_id,
                current_sessions=len(self.sessions),
                max_sessions=self.config.max_sessions,
            )
            raise ServiceUnavailableError(
                f"Session pool at capacity: {self.config.max_sessions}"
            )

        # Get existing session or create new one
        if session_id in self.sessions:
            logger.debug("session_pool_existing_session_found", session_id=session_id)
            session_ctx = self.sessions[session_id]

            # Check if session is still valid
            if session_ctx.is_expired():
                logger.info("session_expired", session_id=session_id)
                await self._remove_session(session_id)
                session_ctx = await self._create_session(session_id, options)
            elif not await session_ctx.is_healthy() and self.config.connection_recovery:
                logger.info("session_unhealthy_recovering", session_id=session_id)
                await session_ctx.connect()
            else:
                logger.debug(
                    "session_pool_reusing_healthy_session", session_id=session_id
                )
        else:
            logger.debug("session_pool_creating_new_session", session_id=session_id)
            session_ctx = await self._create_session(session_id, options)

        logger.debug(
            "session_pool_get_client_complete",
            session_id=session_id,
            session_status=session_ctx.status,
            session_age_seconds=session_ctx.metrics.age_seconds,
            session_message_count=session_ctx.metrics.message_count,
        )
        return session_ctx

    async def _create_session(
        self, session_id: str, options: ClaudeCodeOptions
    ) -> SessionContext:
        """Create a new session context."""
        options.continue_conversation = True
        session_ctx = SessionContext(
            session_id=session_id, options=options, ttl_seconds=self.config.session_ttl
        )

        # Connect to Claude SDK
        if not await session_ctx.connect():
            raise ServiceUnavailableError(
                f"Failed to establish session connection: {session_id}"
            )

        self.sessions[session_id] = session_ctx

        logger.info(
            "session_created", session_id=session_id, total_sessions=len(self.sessions)
        )

        return session_ctx

    async def _remove_session(self, session_id: str) -> None:
        """Remove and cleanup a session."""
        if session_id not in self.sessions:
            return

        session_ctx = self.sessions.pop(session_id)
        await session_ctx.disconnect()

        logger.info(
            "session_removed",
            session_id=session_id,
            total_sessions=len(self.sessions),
            age_seconds=session_ctx.metrics.age_seconds,
            message_count=session_ctx.metrics.message_count,
        )

    async def _cleanup_loop(self) -> None:
        """Background task to cleanup expired sessions."""
        while not self._shutdown:
            try:
                await asyncio.sleep(self.config.cleanup_interval)
                await self._cleanup_sessions()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("session_cleanup_error", error=str(e), exc_info=True)

    async def _cleanup_sessions(self) -> None:
        """Remove expired and idle sessions."""
        sessions_to_remove = []

        for session_id, session_ctx in self.sessions.items():
            if session_ctx.should_cleanup(self.config.idle_threshold):
                sessions_to_remove.append(session_id)

        if sessions_to_remove:
            logger.info(
                "session_cleanup_starting",
                sessions_to_remove=len(sessions_to_remove),
                total_sessions=len(self.sessions),
            )

            for session_id in sessions_to_remove:
                await self._remove_session(session_id)

    async def interrupt_session(self, session_id: str) -> bool:
        """Interrupt a specific session due to client disconnection.

        Args:
            session_id: The session ID to interrupt

        Returns:
            True if session was found and interrupted, False otherwise
        """
        if session_id not in self.sessions:
            logger.warning(
                "session_interrupt_not_found",
                session_id=session_id,
                total_sessions=len(self.sessions),
            )
            return False

        session_ctx = self.sessions[session_id]

        logger.info(
            "session_interrupt_requested",
            session_id=session_id,
            session_status=session_ctx.status.value,
        )

        try:
            # Interrupt the session (only stops ongoing operations)
            await session_ctx.interrupt()

            logger.info("session_interrupted", session_id=session_id)
            return True

        except Exception as e:
            logger.error(
                "session_interrupt_failed",
                session_id=session_id,
                error=str(e),
                exc_info=True,
            )
            return False

    async def interrupt_all_sessions(self) -> int:
        """Interrupt all active sessions (stops ongoing operations).

        Returns:
            Number of sessions that were interrupted
        """
        session_ids = list(self.sessions.keys())
        interrupted_count = 0

        logger.info(
            "session_interrupt_all_requested",
            total_sessions=len(session_ids),
        )

        for session_id in session_ids:
            try:
                session_ctx = self.sessions[session_id]
                await session_ctx.interrupt()
                interrupted_count += 1
            except Exception as e:
                logger.error(
                    "session_interrupt_failed_during_all",
                    session_id=session_id,
                    error=str(e),
                )

        logger.info(
            "session_interrupt_all_completed",
            interrupted_count=interrupted_count,
            total_requested=len(session_ids),
        )

        return interrupted_count

    def get_stats(self) -> dict[str, Any]:
        """Get session pool statistics."""
        active_sessions = sum(
            1 for s in self.sessions.values() if s.status == SessionStatus.ACTIVE
        )

        total_messages = sum(s.metrics.message_count for s in self.sessions.values())

        return {
            "enabled": self.config.enabled,
            "total_sessions": len(self.sessions),
            "active_sessions": active_sessions,
            "max_sessions": self.config.max_sessions,
            "total_messages": total_messages,
            "session_ttl": self.config.session_ttl,
            "cleanup_interval": self.config.cleanup_interval,
        }
