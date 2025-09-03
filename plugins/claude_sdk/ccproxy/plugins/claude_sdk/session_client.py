"""Session client for persistent Claude SDK connections."""

from __future__ import annotations

import asyncio
import time
from enum import Enum
from typing import Any

from claude_code_sdk import ClaudeCodeOptions
from pydantic import BaseModel

from ccproxy.core.async_task_manager import create_managed_task
from ccproxy.core.async_utils import patched_typing
from ccproxy.core.logging import get_plugin_logger
from ccproxy.utils.id_generator import generate_client_id


with patched_typing():
    from claude_code_sdk import ClaudeSDKClient as ImportedClaudeSDKClient

logger = get_plugin_logger()


class SessionStatus(str, Enum):
    """Session lifecycle status."""

    ACTIVE = "active"
    IDLE = "idle"
    CONNECTING = "connecting"
    INTERRUPTING = "interrupting"
    DISCONNECTED = "disconnected"
    ERROR = "error"
    EXPIRED = "expired"


class SessionMetrics(BaseModel):
    """Session performance metrics."""

    created_at: float
    last_used: float
    message_count: int = 0
    error_count: int = 0
    total_duration_ms: float = 0.0

    @property
    def age_seconds(self) -> float:
        return time.time() - self.created_at

    @property
    def idle_seconds(self) -> float:
        return time.time() - self.last_used


class SessionClient:
    """Manages a persistent Claude SDK connection with session state."""

    def __init__(
        self,
        session_id: str,
        options: ClaudeCodeOptions,
        client_id: str | None = None,
        ttl_seconds: int = 3600,
    ):
        self.session_id = session_id
        self.client_id = client_id or generate_client_id()
        self.options = options
        self.ttl_seconds = ttl_seconds

        # SDK client and connection state
        self.claude_client: ImportedClaudeSDKClient | None = None
        self.sdk_session_id: str | None = None

        # Session management
        self.status = SessionStatus.IDLE
        self.lock = asyncio.Lock()  # Prevent concurrent access
        self.metrics = SessionMetrics(created_at=time.time(), last_used=time.time())

        # Error handling
        self.last_error: Exception | None = None
        self.connection_attempts = 0
        self.max_connection_attempts = 3

        # Background connection task
        self._connection_task: asyncio.Task[bool] | None = None

        # Active stream tracking
        self.active_stream_task: asyncio.Task[None] | None = None
        self.has_active_stream: bool = False
        self.active_stream_handle: Any = (
            None  # StreamHandle when using queue-based approach
        )

        # Interrupt synchronization
        self._interrupt_complete_event = asyncio.Event()
        self._interrupt_complete_event.set()  # Initially set (not interrupting)

        # Session reuse tracking
        self.is_newly_created = True  # Flag to track if this is a new session

    async def connect(self) -> bool:
        """Establish connection to Claude SDK."""
        async with self.lock:
            if self.status == SessionStatus.ACTIVE and self.claude_client:
                return True

            try:
                self.status = SessionStatus.CONNECTING
                self.connection_attempts += 1

                logger.debug(
                    "session_connecting",
                    session_id=self.session_id,
                    client_id=self.client_id,
                    attempt=self.connection_attempts,
                )

                self.claude_client = ImportedClaudeSDKClient(self.options)
                await self.claude_client.connect()

                self.status = SessionStatus.ACTIVE
                self.last_error = None

                logger.debug(
                    "session_connected",
                    session_id=self.session_id,
                    client_id=self.client_id,
                    attempt=self.connection_attempts,
                )

                return True

            except ConnectionError as e:
                self.status = SessionStatus.ERROR
                self.last_error = e
                self.metrics.error_count += 1

                logger.error(
                    "session_connection_network_error",
                    session_id=self.session_id,
                    attempt=self.connection_attempts,
                    error=str(e),
                    exc_info=e,
                )
            except TimeoutError as e:
                self.status = SessionStatus.ERROR
                self.last_error = e
                self.metrics.error_count += 1

                logger.error(
                    "session_connection_timeout",
                    session_id=self.session_id,
                    attempt=self.connection_attempts,
                    error=str(e),
                    exc_info=e,
                )

                if self.connection_attempts >= self.max_connection_attempts:
                    logger.error(
                        "session_connection_exhausted",
                        session_id=self.session_id,
                        max_attempts=self.max_connection_attempts,
                    )

                return False
            except Exception as e:
                self.status = SessionStatus.ERROR
                self.last_error = e
                self.metrics.error_count += 1

                logger.error(
                    "session_connection_failed",
                    session_id=self.session_id,
                    attempt=self.connection_attempts,
                    error=str(e),
                    exc_info=e,
                )

                if self.connection_attempts >= self.max_connection_attempts:
                    logger.error(
                        "session_connection_exhausted",
                        session_id=self.session_id,
                        max_attempts=self.max_connection_attempts,
                    )

                return False

            # This should never be reached, but mypy needs it
            return False

    async def connect_background(self) -> asyncio.Task[bool]:
        """Start connection in background without blocking.

        Returns:
            Task that completes when connection is established
        """
        if self._connection_task is None or self._connection_task.done():
            self._connection_task = await create_managed_task(
                self._connect_async(),
                name=f"session_connect_{self.session_id}",
                creator="SessionClient",
            )
            logger.debug(
                "session_background_connection_started",
                session_id=self.session_id,
            )
        return self._connection_task

    async def _connect_async(self) -> bool:
        """Internal async connection method for background task."""
        try:
            return await self.connect()
        except Exception as e:
            logger.error(
                "session_background_connection_failed",
                session_id=self.session_id,
                error=str(e),
                exc_info=e,
            )
            return False

    async def ensure_connected(self) -> bool:
        """Ensure connection is established, waiting for background task if needed."""
        if self._connection_task and not self._connection_task.done():
            # Wait for background connection to complete
            return await self._connection_task
        return await self.connect()

    async def disconnect(self) -> None:
        """Gracefully disconnect from Claude SDK."""
        async with self.lock:
            if self.claude_client:
                try:
                    await self.claude_client.disconnect()
                    logger.debug("session_disconnected", session_id=self.session_id)
                except TimeoutError as e:
                    logger.warning(
                        "session_disconnect_timeout",
                        session_id=self.session_id,
                        error=str(e),
                        exc_info=e,
                    )
                except Exception as e:
                    logger.warning(
                        "session_disconnect_error",
                        session_id=self.session_id,
                        error=str(e),
                        exc_info=False,
                    )
                finally:
                    self.claude_client = None
                    self.status = SessionStatus.DISCONNECTED

    async def interrupt(self) -> None:
        """Interrupt any ongoing operations with timeout and force disconnect fallback."""
        if not self.claude_client:
            logger.debug(
                "session_interrupt_no_client",
                session_id=self.session_id,
            )
            return

        # Check if already interrupting to prevent duplicate interrupt calls
        if self.status == SessionStatus.INTERRUPTING:
            logger.debug(
                "session_interrupt_already_in_progress",
                session_id=self.session_id,
                message="Interrupt already in progress, skipping duplicate call",
            )
            return

        # Set status to INTERRUPTING to prevent reuse during interrupt
        self.status = SessionStatus.INTERRUPTING

        # Clear the interrupt completion event to signal that interrupt is starting
        self._interrupt_complete_event.clear()

        logger.debug(
            "session_interrupting",
            session_id=self.session_id,
            status=self.status.value,
        )

        # Set up a hard timeout for the entire interrupt operation
        start_time = asyncio.get_event_loop().time()
        max_interrupt_time = 15.0  # Maximum 15 seconds for entire interrupt

        try:
            # First, interrupt the stream handle if available
            if self.active_stream_handle:
                logger.debug(
                    "session_interrupt_via_stream_handle",
                    session_id=self.session_id,
                    handle_id=self.active_stream_handle.handle_id,
                    message="Interrupting via stream handle first",
                )

                try:
                    # Interrupt the stream handle - this stops the worker
                    interrupted = await self.active_stream_handle.interrupt()
                    if interrupted:
                        logger.debug(
                            "session_stream_handle_interrupted",
                            session_id=self.session_id,
                            handle_id=self.active_stream_handle.handle_id,
                        )
                        # Clear the handle reference
                        self.active_stream_handle = None
                except asyncio.CancelledError as e:
                    logger.warning(
                        "session_stream_handle_interrupt_cancelled",
                        session_id=self.session_id,
                        error=str(e),
                        exc_info=e,
                        message="Stream handle interrupt was cancelled, continuing with SDK interrupt",
                    )
                except TimeoutError as e:
                    logger.warning(
                        "session_stream_handle_interrupt_timeout",
                        session_id=self.session_id,
                        error=str(e),
                        exc_info=e,
                        message="Stream handle interrupt timed out, continuing with SDK interrupt",
                    )
                except Exception as e:
                    logger.warning(
                        "session_stream_handle_interrupt_error",
                        session_id=self.session_id,
                        error=str(e),
                        exc_info=e,
                        message="Failed to interrupt stream handle, continuing with SDK interrupt",
                    )

            # Now call SDK interrupt - should complete quickly since worker is stopped
            logger.debug(
                "session_interrupt_calling_sdk",
                session_id=self.session_id,
                message="Calling SDK interrupt method",
            )

            try:
                # Call interrupt directly with timeout - avoid creating separate tasks
                await asyncio.wait_for(self.claude_client.interrupt(), timeout=30.0)
                logger.debug(
                    "session_interrupted_gracefully", session_id=self.session_id
                )
                # Reset status after successful interrupt
                self.status = SessionStatus.DISCONNECTED

            except TimeoutError:
                # Interrupt timed out
                logger.warning(
                    "session_interrupt_sdk_timeout",
                    session_id=self.session_id,
                    message="SDK interrupt timed out after 30 seconds",
                )
                raise TimeoutError("Interrupt timed out") from None

        except TimeoutError:
            logger.warning(
                "session_interrupt_timeout",
                session_id=self.session_id,
                message="Graceful interrupt timed out, forcing disconnect",
            )

            # Force disconnect if interrupt hangs
            await self._force_disconnect()

        except asyncio.CancelledError as e:
            logger.warning(
                "session_interrupt_cancelled",
                session_id=self.session_id,
                error=str(e),
                exc_info=e,
            )
            # If interrupt fails, try force disconnect as fallback
            try:
                logger.debug(
                    "session_interrupt_fallback_disconnect",
                    session_id=self.session_id,
                )
                await self._force_disconnect()
            except Exception as disconnect_error:
                logger.error(
                    "session_force_disconnect_failed",
                    session_id=self.session_id,
                    error=str(disconnect_error),
                    exc_info=disconnect_error,
                )
        except Exception as e:
            logger.warning(
                "session_interrupt_error",
                session_id=self.session_id,
                error=str(e),
                exc_info=e,
            )

            # If interrupt fails, try force disconnect as fallback
            try:
                logger.debug(
                    "session_interrupt_fallback_disconnect",
                    session_id=self.session_id,
                )
                await self._force_disconnect()
            except Exception as disconnect_error:
                logger.error(
                    "session_force_disconnect_failed",
                    session_id=self.session_id,
                    error=str(disconnect_error),
                    exc_info=disconnect_error,
                )
        finally:
            # Final safety check - ensure we don't hang forever
            total_elapsed = asyncio.get_event_loop().time() - start_time
            if total_elapsed > max_interrupt_time:
                logger.error(
                    "session_interrupt_max_time_exceeded",
                    session_id=self.session_id,
                    elapsed_seconds=total_elapsed,
                    max_seconds=max_interrupt_time,
                    message="Interrupt operation exceeded maximum time limit",
                )

            # Always reset status from INTERRUPTING
            if self.status == SessionStatus.INTERRUPTING:
                # Force mark as disconnected
                self.status = SessionStatus.DISCONNECTED
                self.claude_client = None

            # Mark stream as no longer active
            self.has_active_stream = False

            # Signal that interrupt has completed (success or failure)
            self._interrupt_complete_event.set()

    async def _force_disconnect(self) -> None:
        """Force disconnect the session when interrupt fails or times out."""
        logger.warning(
            "session_force_disconnecting",
            session_id=self.session_id,
            message="Force disconnecting stuck session",
        )

        # Try to drain any active stream first with timeout
        try:
            await asyncio.wait_for(
                self.drain_active_stream(),
                timeout=5.0,  # 5 second timeout for draining in force disconnect
            )
        except TimeoutError:
            logger.warning(
                "session_force_drain_timeout",
                session_id=self.session_id,
                message="Force disconnect stream draining timed out after 5 seconds",
            )

        try:
            if self.claude_client:
                # Try to disconnect with timeout
                await asyncio.wait_for(
                    self.claude_client.disconnect(),
                    timeout=3.0,  # 3 second timeout for disconnect
                )
        except TimeoutError as e:
            logger.warning(
                "session_force_disconnect_timeout",
                session_id=self.session_id,
                error=str(e),
                exc_info=e,
            )
        except Exception as e:
            logger.warning(
                "session_force_disconnect_error",
                session_id=self.session_id,
                error=str(e),
                exc_info=e,
            )
        finally:
            # Always clean up the client reference and mark as disconnected
            self.claude_client = None
            self.status = SessionStatus.DISCONNECTED
            self.last_error = Exception(
                "Session force disconnected due to hanging operation"
            )

            logger.warning(
                "session_force_disconnected",
                session_id=self.session_id,
                message="Session forcibly disconnected and marked for cleanup",
            )

    async def drain_active_stream(self) -> None:
        """Drain any active stream to prevent stale messages on reconnection."""
        if not self.has_active_stream:
            logger.debug(
                "session_no_active_stream_to_drain",
                session_id=self.session_id,
            )
            return

        logger.debug(
            "session_draining_active_stream",
            session_id=self.session_id,
            message="Draining active stream after client disconnection",
        )

        # With queue-based architecture, we use the stream handle
        if self.active_stream_handle:
            logger.debug(
                "session_draining_via_handle",
                session_id=self.session_id,
                handle_id=self.active_stream_handle.handle_id,
                message="Using stream handle to drain messages",
            )

            try:
                # Wait for the worker to complete
                completed = await self.active_stream_handle.wait_for_completion(
                    timeout=30.0
                )
                if completed:
                    logger.debug(
                        "session_stream_drained_via_handle",
                        session_id=self.session_id,
                        handle_id=self.active_stream_handle.handle_id,
                    )
                else:
                    logger.warning(
                        "session_stream_drain_timeout_via_handle",
                        session_id=self.session_id,
                        handle_id=self.active_stream_handle.handle_id,
                        message="Stream drain timed out after 30 seconds",
                    )
            except TimeoutError as e:
                logger.error(
                    "session_stream_drain_timeout_via_handle",
                    session_id=self.session_id,
                    handle_id=self.active_stream_handle.handle_id,
                    error=str(e),
                    exc_info=e,
                )
            except asyncio.CancelledError as e:
                logger.warning(
                    "session_stream_drain_cancelled_via_handle",
                    session_id=self.session_id,
                    handle_id=self.active_stream_handle.handle_id,
                    error=str(e),
                    exc_info=e,
                )
            except Exception as e:
                logger.error(
                    "session_stream_drain_error_via_handle",
                    session_id=self.session_id,
                    handle_id=self.active_stream_handle.handle_id,
                    error=str(e),
                    exc_info=e,
                )
            finally:
                self.active_stream_handle = None
                self.has_active_stream = False
                self.active_stream_task = None

            return

        # Should not happen with queue-based architecture
        logger.warning(
            "session_no_handle_for_drain",
            session_id=self.session_id,
            message="No stream handle available for draining",
        )
        self.has_active_stream = False
        self.active_stream_task = None

    async def wait_for_interrupt_complete(self, timeout: float = 5.0) -> bool:
        """Wait for any in-progress interrupt to complete.

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            True if interrupt completed within timeout, False if timed out
        """
        try:
            await asyncio.wait_for(
                self._interrupt_complete_event.wait(), timeout=timeout
            )
            logger.debug(
                "session_interrupt_wait_completed",
                session_id=self.session_id,
                message="Interrupt completion event signaled",
            )
            return True
        except TimeoutError:
            logger.warning(
                "session_interrupt_wait_timeout",
                session_id=self.session_id,
                timeout=timeout,
                message="Timeout waiting for interrupt to complete",
            )
            return False

    async def is_healthy(self) -> bool:
        """Check if the session connection is healthy."""
        # Add health check logic here if Claude SDK provides it
        # For now, assume active status means healthy
        return bool(self.claude_client and self.status == SessionStatus.ACTIVE)

    def is_expired(self) -> bool:
        """Check if session has exceeded TTL."""
        return self.metrics.age_seconds > self.ttl_seconds

    def update_usage(self) -> None:
        """Update session usage metrics."""
        old_message_count = self.metrics.message_count
        self.metrics.last_used = time.time()
        self.metrics.message_count += 1

        # Mark session as reused after first usage
        if self.is_newly_created and self.metrics.message_count > 1:
            self.is_newly_created = False

        logger.debug(
            "session_usage_updated",
            session_id=self.session_id,
            message_count=self.metrics.message_count,
            previous_message_count=old_message_count,
            age_seconds=self.metrics.age_seconds,
            idle_seconds=self.metrics.idle_seconds,
            is_newly_created=self.is_newly_created,
        )

    def mark_as_reused(self) -> None:
        """Mark this session as being reused (not newly created)."""
        self.is_newly_created = False

    def should_cleanup(
        self, idle_threshold: int = 300, stuck_threshold: int = 900
    ) -> bool:
        """Determine if session should be cleaned up.

        Args:
            idle_threshold: Max idle time in seconds before cleanup
            stuck_threshold: Max time a session can be ACTIVE without going idle (indicating stuck)
        """
        # Check if session has been stuck in ACTIVE state too long
        is_potentially_stuck = (
            self.status == SessionStatus.ACTIVE
            and self.metrics.idle_seconds < 10  # Still being used but...
            and self.metrics.age_seconds
            > stuck_threshold  # ...has been active way too long
        )

        return (
            self.is_expired()
            or self.metrics.idle_seconds > idle_threshold
            or self.status in (SessionStatus.ERROR, SessionStatus.DISCONNECTED)
            or is_potentially_stuck
        )
