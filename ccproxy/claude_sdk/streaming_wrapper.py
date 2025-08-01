"""Wrapper for streaming responses with cleanup callbacks."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, TypeVar

import structlog


if TYPE_CHECKING:
    from ccproxy.claude_sdk.pool import PooledClient

T = TypeVar("T")

logger = structlog.get_logger(__name__)


class StreamingResponseWithCleanup(AsyncIterator[T]):
    """Wrapper that attaches a cleanup callback to a streaming response.

    This allows the StreamingResponseWithPoolCleanup to monitor for
    disconnection and signal when pool resources should be released.
    """

    def __init__(
        self,
        wrapped_iterator: AsyncIterator[T],
        pooled_client: PooledClient | None = None,
    ) -> None:
        """Initialize the wrapper.

        Args:
            wrapped_iterator: The original async iterator
            pooled_client: Optional pooled client for cleanup
        """
        self._wrapped_iterator = wrapped_iterator
        self._pooled_client = pooled_client

        # Create a future that can be set when cleanup is needed
        self._cleanup_callback: asyncio.Future[bool] = asyncio.Future()

    def __aiter__(self) -> AsyncIterator[T]:
        """Return self as the async iterator."""
        return self

    async def __anext__(self) -> T:
        """Get the next item from the wrapped iterator."""
        try:
            return await self._wrapped_iterator.__anext__()
        except StopAsyncIteration:
            # Stream completed normally
            raise
        except Exception:
            # Error during streaming - mark client as unhealthy
            if self._pooled_client:
                self._pooled_client.is_healthy = False
                logger.warning(
                    "streaming_wrapper_error_marking_unhealthy",
                    client_id=self._pooled_client.client_id,
                )
            raise

    async def wait_for_cleanup_signal(self) -> bool:
        """Wait for the cleanup signal.

        Returns:
            True if cleanup is needed (client disconnected)
        """
        try:
            return await self._cleanup_callback
        except asyncio.CancelledError:
            return False

    def cleanup_needed(self) -> None:
        """Signal that cleanup is needed (client disconnected)."""
        if not self._cleanup_callback.done():
            self._cleanup_callback.set_result(True)

            # Mark client as unhealthy if disconnection detected
            if self._pooled_client:
                self._pooled_client.is_healthy = False
                logger.info(
                    "streaming_wrapper_disconnection_cleanup",
                    client_id=self._pooled_client.client_id,
                )
