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
        """Interrupt any ongoing operations without disconnecting."""
        if self.claude_client:
            try:
                logger.info(
                    "session_interrupting",
                    session_id=self.session_id,
                    status=self.status.value,
                )

                # Call interrupt to stop any ongoing operations
                await self.claude_client.interrupt()

                logger.info("session_interrupted", session_id=self.session_id)
            except Exception as e:
                logger.warning(
                    "session_interrupt_error",
                    session_id=self.session_id,
                    error=str(e),
                )
                # Don't change status or client state on interrupt error

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

    def should_cleanup(self, idle_threshold: int = 300) -> bool:
        """Determine if session should be cleaned up."""
        return (
            self.is_expired()
            or self.metrics.idle_seconds > idle_threshold
            or self.status in (SessionStatus.ERROR, SessionStatus.DISCONNECTED)
        )
