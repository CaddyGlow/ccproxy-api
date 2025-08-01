"""Session-aware connection pool for persistent Claude SDK connections."""

from __future__ import annotations

import asyncio
import contextlib
from typing import TYPE_CHECKING, Any

import structlog
from claude_code_sdk import ClaudeCodeOptions

from ccproxy.claude_sdk.session_client import SessionClient, SessionStatus
from ccproxy.core.errors import ClaudeProxyError, ServiceUnavailableError


if TYPE_CHECKING:
    pass


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
        self.sessions: dict[str, SessionClient] = {}
        self.cleanup_task: asyncio.Task[None] | None = None
        self._shutdown = False
        self._lock = asyncio.Lock()

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
        async with self._lock:
            disconnect_tasks = [
                session_client.disconnect() for session_client in self.sessions.values()
            ]

            if disconnect_tasks:
                await asyncio.gather(*disconnect_tasks, return_exceptions=True)

            self.sessions.clear()

        logger.info("session_pool_stopped")

    async def get_session_client(
        self, session_id: str, options: ClaudeCodeOptions
    ) -> SessionClient:
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

        # Check session limit and get/create session
        async with self._lock:
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
            options.continue_conversation = True
            # Get existing session or create new one
            if session_id in self.sessions:
                session_client = self.sessions[session_id]
                logger.debug(
                    "session_pool_existing_session_found",
                    session_id=session_id,
                    client_id=session_client.client_id,
                    session_status=session_client.status.value,
                )

                # Check if session is currently being interrupted
                if session_client.status.value == "interrupting":
                    logger.warning(
                        "session_pool_interrupting_session",
                        session_id=session_id,
                        client_id=session_client.client_id,
                        message="Session is currently being interrupted, creating new session",
                    )
                    # Don't try to reuse a session that's being interrupted
                    await self._remove_session_unlocked(session_id)
                    session_client = await self._create_session_unlocked(
                        session_id, options
                    )
                # Check if session has an active stream that needs cleanup
                elif (
                    session_client.has_active_stream
                    or session_client.active_stream_handle
                ):
                    logger.debug(
                        "session_pool_active_stream_detected",
                        session_id=session_id,
                        client_id=session_client.client_id,
                        has_stream=session_client.has_active_stream,
                        has_handle=bool(session_client.active_stream_handle),
                        idle_seconds=session_client.metrics.idle_seconds,
                        message="Session has active stream/handle, checking if cleanup needed",
                    )

                    # Check if the stream is idle/stale (idle for more than 10 seconds)
                    # If it's been idle for a while, it's likely abandoned and should be interrupted
                    is_stream_stale = session_client.metrics.idle_seconds > 10.0

                    if session_client.active_stream_handle and is_stream_stale:
                        old_handle_id = session_client.active_stream_handle.handle_id
                        logger.info(
                            "session_pool_interrupting_stale_stream",
                            session_id=session_id,
                            old_handle_id=old_handle_id,
                            idle_seconds=session_client.metrics.idle_seconds,
                            message="Interrupting stale stream handle before reuse",
                        )

                        try:
                            # Interrupt the old stream handle to stop its worker
                            interrupted = (
                                await session_client.active_stream_handle.interrupt()
                            )
                            if interrupted:
                                logger.info(
                                    "session_pool_interrupted_stale_stream",
                                    session_id=session_id,
                                    old_handle_id=old_handle_id,
                                    message="Successfully interrupted stale stream handle",
                                )
                            else:
                                logger.debug(
                                    "session_pool_interrupt_stale_stream_not_needed",
                                    session_id=session_id,
                                    old_handle_id=old_handle_id,
                                    message="Stale stream handle was already completed",
                                )
                        except Exception as e:
                            logger.warning(
                                "session_pool_interrupt_stale_stream_failed",
                                session_id=session_id,
                                old_handle_id=old_handle_id,
                                error=str(e),
                                error_type=type(e).__name__,
                                message="Failed to interrupt stale stream handle, clearing anyway",
                            )
                        finally:
                            # Always clear the handle after interrupt attempt
                            session_client.active_stream_handle = None
                            session_client.has_active_stream = False
                    elif session_client.active_stream_handle and not is_stream_stale:
                        # Stream is recent, likely from a previous request that just finished
                        # Just clear the handle without interrupting to allow immediate reuse
                        logger.debug(
                            "session_pool_clearing_recent_stream",
                            session_id=session_id,
                            old_handle_id=session_client.active_stream_handle.handle_id,
                            idle_seconds=session_client.metrics.idle_seconds,
                            message="Clearing recent stream handle for immediate reuse",
                        )
                        session_client.active_stream_handle = None
                        session_client.has_active_stream = False
                    else:
                        # No handle but has_active_stream flag is set, just clear the flag
                        session_client.has_active_stream = False

                    logger.debug(
                        "session_pool_stream_cleared",
                        session_id=session_id,
                        client_id=session_client.client_id,
                        was_interrupted=is_stream_stale,
                        was_recent=not is_stream_stale,
                        message="Stream state cleared, session ready for reuse",
                    )
                # Check if session is still valid
                elif session_client.is_expired():
                    logger.info("session_expired", session_id=session_id)
                    await self._remove_session_unlocked(session_id)
                    session_client = await self._create_session_unlocked(
                        session_id, options
                    )
                elif (
                    not await session_client.is_healthy()
                    and self.config.connection_recovery
                ):
                    logger.info("session_unhealthy_recovering", session_id=session_id)
                    await session_client.connect()
                else:
                    logger.debug(
                        "session_pool_reusing_healthy_session",
                        session_id=session_id,
                        client_id=session_client.client_id,
                    )
            else:
                logger.debug("session_pool_creating_new_session", session_id=session_id)
                session_client = await self._create_session_unlocked(
                    session_id, options
                )

        # Ensure session is connected before returning
        if not await session_client.ensure_connected():
            logger.error(
                "session_pool_connection_failed",
                session_id=session_id,
            )
            raise ServiceUnavailableError(
                f"Failed to establish session connection: {session_id}"
            )

        logger.debug(
            "session_pool_get_client_complete",
            session_id=session_id,
            client_id=session_client.client_id,
            session_status=session_client.status,
            session_age_seconds=session_client.metrics.age_seconds,
            session_message_count=session_client.metrics.message_count,
        )
        return session_client

    async def _create_session(
        self, session_id: str, options: ClaudeCodeOptions
    ) -> SessionClient:
        """Create a new session context (acquires lock)."""
        async with self._lock:
            return await self._create_session_unlocked(session_id, options)

    async def _create_session_unlocked(
        self, session_id: str, options: ClaudeCodeOptions
    ) -> SessionClient:
        """Create a new session context (requires lock to be held)."""
        session_client = SessionClient(
            session_id=session_id, options=options, ttl_seconds=self.config.session_ttl
        )

        # Start connection in background
        connection_task = session_client.connect_background()

        # Add to sessions immediately (will connect in background)
        self.sessions[session_id] = session_client

        # Optionally wait for connection to verify it works
        # For now, we'll let it connect in background and check on first use
        logger.debug(
            "session_connecting_background",
            session_id=session_id,
            client_id=session_client.client_id,
        )

        logger.info(
            "session_created",
            session_id=session_id,
            client_id=session_client.client_id,
            total_sessions=len(self.sessions),
        )

        return session_client

    async def _remove_session(self, session_id: str) -> None:
        """Remove and cleanup a session (acquires lock)."""
        async with self._lock:
            await self._remove_session_unlocked(session_id)

    async def _remove_session_unlocked(self, session_id: str) -> None:
        """Remove and cleanup a session (requires lock to be held)."""
        if session_id not in self.sessions:
            return

        session_client = self.sessions.pop(session_id)
        await session_client.disconnect()

        logger.info(
            "session_removed",
            session_id=session_id,
            total_sessions=len(self.sessions),
            age_seconds=session_client.metrics.age_seconds,
            message_count=session_client.metrics.message_count,
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
        """Remove expired, idle, and stuck sessions."""
        sessions_to_remove = []
        stuck_sessions = []

        # Get a snapshot of sessions to check
        async with self._lock:
            sessions_snapshot = list(self.sessions.items())

        # Check sessions outside the lock to avoid holding it too long
        for session_id, session_client in sessions_snapshot:
            # Check if session is potentially stuck (active too long)
            is_stuck = (
                session_client.status.value == "active"
                and session_client.metrics.idle_seconds < 10
                and session_client.metrics.age_seconds > 900  # 15 minutes
            )

            if is_stuck:
                stuck_sessions.append(session_id)
                logger.warning(
                    "session_stuck_detected",
                    session_id=session_id,
                    age_seconds=session_client.metrics.age_seconds,
                    idle_seconds=session_client.metrics.idle_seconds,
                    message_count=session_client.metrics.message_count,
                    message="Session appears stuck, will interrupt and cleanup",
                )

                # Try to interrupt stuck session before cleanup
                try:
                    await session_client.interrupt()
                except Exception as e:
                    logger.warning(
                        "session_stuck_interrupt_failed",
                        session_id=session_id,
                        error=str(e),
                    )

            # Check normal cleanup criteria (including stuck sessions)
            if session_client.should_cleanup(
                self.config.idle_threshold, stuck_threshold=900
            ):
                sessions_to_remove.append(session_id)

        if sessions_to_remove:
            logger.info(
                "session_cleanup_starting",
                sessions_to_remove=len(sessions_to_remove),
                stuck_sessions=len(stuck_sessions),
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
        async with self._lock:
            if session_id not in self.sessions:
                logger.warning("session_not_found", session_id=session_id)
                return False

            session_client = self.sessions[session_id]

        try:
            # Interrupt the session with 30-second timeout (allows for longer SDK response times)
            await asyncio.wait_for(session_client.interrupt(), timeout=30.0)
            logger.info("session_interrupted", session_id=session_id)

            # Remove the session to prevent reuse
            await self._remove_session(session_id)
            return True

        except (TimeoutError, Exception) as e:
            logger.error(
                "session_interrupt_failed",
                session_id=session_id,
                error=str(e)
                if not isinstance(e, TimeoutError)
                else "Timeout after 30s",
            )
            # Always remove the session on failure
            with contextlib.suppress(Exception):
                await self._remove_session(session_id)
            return False

    async def interrupt_all_sessions(self) -> int:
        """Interrupt all active sessions (stops ongoing operations).

        Returns:
            Number of sessions that were interrupted
        """
        # Get snapshot of all sessions
        async with self._lock:
            session_items = list(self.sessions.items())

        interrupted_count = 0

        logger.info(
            "session_interrupt_all_requested",
            total_sessions=len(session_items),
        )

        for session_id, session_client in session_items:
            try:
                await session_client.interrupt()
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
            total_requested=len(session_items),
        )

        return interrupted_count

    async def has_session(self, session_id: str) -> bool:
        """Check if a session exists in the pool.

        Args:
            session_id: The session ID to check

        Returns:
            True if session exists, False otherwise
        """
        async with self._lock:
            return session_id in self.sessions

    async def get_stats(self) -> dict[str, Any]:
        """Get session pool statistics."""
        async with self._lock:
            sessions_list = list(self.sessions.values())
            total_sessions = len(self.sessions)

        active_sessions = sum(
            1 for s in sessions_list if s.status == SessionStatus.ACTIVE
        )

        total_messages = sum(s.metrics.message_count for s in sessions_list)

        return {
            "enabled": self.config.enabled,
            "total_sessions": total_sessions,
            "active_sessions": active_sessions,
            "max_sessions": self.config.max_sessions,
            "total_messages": total_messages,
            "session_ttl": self.config.session_ttl,
            "cleanup_interval": self.config.cleanup_interval,
        }
