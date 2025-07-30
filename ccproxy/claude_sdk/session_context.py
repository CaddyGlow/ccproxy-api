"""Session context for persistent Claude SDK connections."""

from __future__ import annotations

import asyncio
import time
from enum import Enum

import structlog
from claude_code_sdk import ClaudeCodeOptions
from pydantic import BaseModel

from ccproxy.core.async_utils import patched_typing


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


class SessionContext:
    """Manages a persistent Claude SDK connection with session state."""

    def __init__(
        self, session_id: str, options: ClaudeCodeOptions, ttl_seconds: int = 3600
    ):
        self.session_id = session_id
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
                    attempt=self.connection_attempts,
                )

                self.claude_client = ImportedClaudeSDKClient(self.options)
                await self.claude_client.connect()

                self.status = SessionStatus.ACTIVE
                self.last_error = None

                logger.info(
                    "session_connected",
                    session_id=self.session_id,
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

        try:
            # First attempt: Try graceful interrupt with timeout
            await asyncio.wait_for(
                self.claude_client.interrupt(),
                timeout=5.0,  # 5 second timeout for interrupt
            )

            logger.info("session_interrupted_gracefully", session_id=self.session_id)

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

    async def _force_disconnect(self) -> None:
        """Force disconnect the session when interrupt fails or times out."""
        logger.warning(
            "session_force_disconnecting",
            session_id=self.session_id,
            message="Force disconnecting stuck session",
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
