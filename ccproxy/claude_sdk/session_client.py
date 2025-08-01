"""Session client for persistent Claude SDK connections."""

from __future__ import annotations

import asyncio
import contextlib
import time
from enum import Enum

import structlog
from claude_code_sdk import ClaudeCodeOptions
from pydantic import BaseModel

from ccproxy.core.async_utils import patched_typing
from ccproxy.utils.id_generator import generate_client_id


with patched_typing():
    from claude_code_sdk import ClaudeSDKClient as ImportedClaudeSDKClient

logger = structlog.get_logger(__name__)


class SessionStatus(str, Enum):
    """Session lifecycle status."""

    ACTIVE = "active"
    IDLE = "idle"
    CONNECTING = "connecting"
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

    async def connect(self) -> bool:
        """Establish connection to Claude SDK."""
        async with self.lock:
            if self.status == SessionStatus.ACTIVE and self.claude_client:
                return True

            try:
                self.status = SessionStatus.CONNECTING
                self.connection_attempts += 1

                logger.info(
                    "session_connecting",
                    session_id=self.session_id,
                    client_id=self.client_id,
                    attempt=self.connection_attempts,
                )

                self.claude_client = ImportedClaudeSDKClient(self.options)
                await self.claude_client.connect()

                self.status = SessionStatus.ACTIVE
                self.last_error = None

                logger.info(
                    "session_connected",
                    session_id=self.session_id,
                    client_id=self.client_id,
                    attempt=self.connection_attempts,
                )

                return True

            except Exception as e:
                self.status = SessionStatus.ERROR
                self.last_error = e
                self.metrics.error_count += 1

                logger.error(
                    "session_connection_failed",
                    session_id=self.session_id,
                    attempt=self.connection_attempts,
                    error=str(e),
                    exc_info=True,
                )

                if self.connection_attempts >= self.max_connection_attempts:
                    logger.error(
                        "session_connection_exhausted",
                        session_id=self.session_id,
                        max_attempts=self.max_connection_attempts,
                    )

                return False

    def connect_background(self) -> asyncio.Task[bool]:
        """Start connection in background without blocking.

        Returns:
            Task that completes when connection is established
        """
        if self._connection_task is None or self._connection_task.done():
            self._connection_task = asyncio.create_task(self._connect_async())
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
                    logger.info("session_disconnected", session_id=self.session_id)
                except Exception as e:
                    logger.warning(
                        "session_disconnect_error",
                        session_id=self.session_id,
                        error=str(e),
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

        logger.info(
            "session_interrupting",
            session_id=self.session_id,
            status=self.status.value,
        )

        # Set up a hard timeout for the entire interrupt operation
        start_time = asyncio.get_event_loop().time()
        max_interrupt_time = 15.0  # Maximum 15 seconds for entire interrupt

        try:
            # IMPORTANT: Claude SDK interrupt requires active message consumption
            # Start draining messages BEFORE calling interrupt
            drain_task = None
            if self.has_active_stream:
                logger.debug(
                    "session_starting_drain_before_interrupt",
                    session_id=self.session_id,
                    message="Starting message drain task before interrupt",
                )

                # Start draining in background
                drain_task = asyncio.create_task(self._drain_messages_for_interrupt())

                # Give the drain task a moment to actually start consuming
                await asyncio.sleep(0.1)

            # Now attempt interrupt with the drain task running
            logger.debug(
                "session_interrupt_calling_sdk",
                session_id=self.session_id,
                message="Calling SDK interrupt method with drain task consuming",
                has_drain_task=drain_task is not None,
            )

            # Call interrupt - this should complete quickly with drain task running
            try:
                # Create interrupt task
                interrupt_task = asyncio.create_task(self.claude_client.interrupt())

                # Wait for interrupt with timeout
                done, pending = await asyncio.wait(
                    [interrupt_task], timeout=10.0, return_when=asyncio.FIRST_COMPLETED
                )

                if interrupt_task in done:
                    # Interrupt completed - get the result or exception
                    await interrupt_task
                    logger.info(
                        "session_interrupted_gracefully", session_id=self.session_id
                    )
                else:
                    # Timeout - interrupt is still pending
                    logger.warning(
                        "session_interrupt_timeout_canceling",
                        session_id=self.session_id,
                        message="Interrupt did not complete within 10s, canceling",
                    )
                    interrupt_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await interrupt_task
                    raise TimeoutError("Interrupt did not complete within timeout")

                # Wait for drain task to complete if it's still running
                if drain_task and not drain_task.done():
                    try:
                        await asyncio.wait_for(drain_task, timeout=5.0)
                        logger.debug(
                            "session_drain_task_completed",
                            session_id=self.session_id,
                        )
                    except TimeoutError:
                        logger.warning(
                            "session_drain_task_timeout_after_interrupt",
                            session_id=self.session_id,
                            message="Drain task timed out after successful interrupt",
                        )
                        drain_task.cancel()
                        with contextlib.suppress(asyncio.CancelledError):
                            await drain_task

            except TimeoutError:
                # Interrupt timed out
                logger.warning(
                    "session_interrupt_sdk_timeout",
                    session_id=self.session_id,
                    message="SDK interrupt timed out after 10 seconds",
                )
                if drain_task:
                    drain_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await drain_task
                raise TimeoutError("Interrupt timed out") from None

        except TimeoutError:
            logger.warning(
                "session_interrupt_timeout",
                session_id=self.session_id,
                message="Graceful interrupt timed out, forcing disconnect",
            )

            # Force disconnect if interrupt hangs
            await self._force_disconnect()

        except Exception as e:
            logger.warning(
                "session_interrupt_error",
                session_id=self.session_id,
                error=str(e),
                error_type=type(e).__name__,
            )

            # If interrupt fails, try force disconnect as fallback
            try:
                logger.info(
                    "session_interrupt_fallback_disconnect",
                    session_id=self.session_id,
                )
                await self._force_disconnect()
            except Exception as disconnect_error:
                logger.error(
                    "session_force_disconnect_failed",
                    session_id=self.session_id,
                    error=str(disconnect_error),
                    error_type=type(disconnect_error).__name__,
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
                # Force mark as disconnected
                self.status = SessionStatus.DISCONNECTED
                self.claude_client = None

            # Mark stream as no longer active
            self.has_active_stream = False

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
        except Exception as e:
            logger.warning(
                "session_force_disconnect_error",
                session_id=self.session_id,
                error=str(e),
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

    async def _drain_messages_for_interrupt(self) -> None:
        """Drain messages during interrupt operation to allow interrupt to process."""
        if not self.claude_client:
            return

        logger.info(
            "session_interrupt_drain_started",
            session_id=self.session_id,
            message="Starting to drain messages for interrupt",
        )

        message_count = 0
        start_time = asyncio.get_event_loop().time()
        max_drain_time = 10.0  # Maximum 10 seconds for draining during interrupt

        try:
            # Try to use receive_stream if available, otherwise fall back to receive_response
            if hasattr(self.claude_client, "receive_stream"):
                receive_method = self.claude_client.receive_stream()
            else:
                receive_method = self.claude_client.receive_response()

            async for msg in receive_method:
                message_count += 1
                logger.debug(
                    "session_interrupt_drain_message",
                    session_id=self.session_id,
                    message_count=message_count,
                    message_type=type(msg).__name__,
                )

                # Check if we've exceeded time limit
                elapsed_time = asyncio.get_event_loop().time() - start_time
                if elapsed_time > max_drain_time:
                    logger.warning(
                        "session_interrupt_drain_timeout",
                        session_id=self.session_id,
                        message_count=message_count,
                        elapsed_seconds=elapsed_time,
                        message="Interrupt drain exceeded time limit",
                    )
                    break

                # Check if we've reached ResultMessage (end of stream)
                if (
                    hasattr(msg, "__class__")
                    and msg.__class__.__name__ == "ResultMessage"
                ):
                    logger.info(
                        "session_interrupt_drain_complete",
                        session_id=self.session_id,
                        total_messages=message_count,
                        message="Reached ResultMessage during interrupt drain",
                    )
                    break

        except asyncio.CancelledError:
            logger.debug(
                "session_interrupt_drain_cancelled",
                session_id=self.session_id,
                messages_drained=message_count,
            )
            raise
        except StopAsyncIteration:
            # Stream ended - this is expected after interrupt
            logger.info(
                "session_interrupt_drain_stream_ended",
                session_id=self.session_id,
                messages_drained=message_count,
                message="Stream ended during interrupt drain (expected)",
            )
        except Exception as e:
            logger.error(
                "session_interrupt_drain_error",
                session_id=self.session_id,
                error=str(e),
                error_type=type(e).__name__,
                messages_drained=message_count,
            )

    async def drain_active_stream(self) -> None:
        """Drain any active stream to prevent stale messages on reconnection."""
        if not self.has_active_stream:
            logger.debug(
                "session_no_active_stream_to_drain",
                session_id=self.session_id,
            )
            return

        logger.info(
            "session_draining_active_stream",
            session_id=self.session_id,
            message="Draining active stream after client disconnection",
        )

        if not self.claude_client:
            logger.warning(
                "session_no_client_for_drain",
                session_id=self.session_id,
            )
            self.has_active_stream = False
            return

        try:
            # Continue receiving messages from the existing stream
            message_count = 0
            max_messages = 1000  # Safety limit
            start_time = asyncio.get_event_loop().time()
            max_drain_time = 30.0  # Maximum 30 seconds for draining

            # Get the response iterator from the Claude client
            response_iterator = self.claude_client.receive_response()

            async for msg in response_iterator:
                message_count += 1
                logger.debug(
                    "session_draining_message",
                    session_id=self.session_id,
                    message_count=message_count,
                    message_type=type(msg).__name__,
                )

                # Check if we've exceeded time limit
                elapsed_time = asyncio.get_event_loop().time() - start_time
                if elapsed_time > max_drain_time:
                    logger.warning(
                        "session_drain_time_limit_reached",
                        session_id=self.session_id,
                        message_count=message_count,
                        elapsed_seconds=elapsed_time,
                    )
                    break

                # Safety check to prevent infinite loops
                if message_count >= max_messages:
                    logger.warning(
                        "session_drain_limit_reached",
                        session_id=self.session_id,
                        message_count=message_count,
                    )
                    break

                # Check if we've reached the end (ResultMessage)
                if (
                    hasattr(msg, "__class__")
                    and msg.__class__.__name__ == "ResultMessage"
                ):
                    logger.info(
                        "session_drain_completed",
                        session_id=self.session_id,
                        total_messages=message_count,
                    )
                    break

        except StopAsyncIteration:
            # Stream ended
            logger.info(
                "session_stream_drained",
                session_id=self.session_id,
                drained_messages=message_count,
            )
        except Exception as e:
            logger.error(
                "session_stream_drain_error",
                session_id=self.session_id,
                error=str(e),
                error_type=type(e).__name__,
                drained_messages=message_count,
            )
        finally:
            self.has_active_stream = False
            self.active_stream_task = None

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

        logger.debug(
            "session_usage_updated",
            session_id=self.session_id,
            message_count=self.metrics.message_count,
            previous_message_count=old_message_count,
            age_seconds=self.metrics.age_seconds,
            idle_seconds=self.metrics.idle_seconds,
        )

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
