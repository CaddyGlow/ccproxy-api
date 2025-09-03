"""Stream handle for managing worker lifecycle and providing listeners."""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

from ccproxy.core.async_task_manager import create_managed_task
from ccproxy.core.logging import get_plugin_logger

from .config import SessionPoolSettings
from .message_queue import QueueListener
from .session_client import SessionClient
from .stream_worker import StreamWorker, WorkerStatus


logger = get_plugin_logger()


class StreamHandle:
    """Handle for a streaming response that manages worker and listeners."""

    def __init__(
        self,
        message_iterator: AsyncIterator[Any],
        session_id: str | None = None,
        request_id: str | None = None,
        session_client: SessionClient | None = None,
        session_config: SessionPoolSettings | None = None,
    ):
        """Initialize the stream handle.

        Args:
            message_iterator: The SDK message iterator
            session_id: Optional session ID
            request_id: Optional request ID
            session_client: Optional session client
            session_config: Optional session pool configuration
        """
        self.handle_id = str(uuid.uuid4())
        self._message_iterator = message_iterator
        self.session_id = session_id
        self.request_id = request_id
        self._session_client = session_client

        # Timeout configuration
        self._session_config = session_config
        self._first_chunk_timeout = (
            session_config.stream_first_chunk_timeout if session_config else 3.0
        )
        self._ongoing_timeout = (
            session_config.stream_ongoing_timeout if session_config else 60.0
        )
        self._interrupt_timeout = (
            session_config.stream_interrupt_timeout if session_config else 10.0
        )

        # Worker management
        self._worker: StreamWorker | None = None
        self._worker_lock = asyncio.Lock()
        self._listeners: dict[str, QueueListener] = {}
        self._created_at = time.time()
        self._first_listener_at: float | None = None

        # Message lifecycle tracking for stale detection
        self._first_chunk_received_at: float | None = None
        self._completed_at: float | None = None
        self._has_result_message = False
        self._last_activity_at = time.time()

    async def create_listener(self) -> AsyncIterator[Any]:
        """Create a new listener for this stream.

        This method starts the worker on first listener and returns
        an async iterator for consuming messages.

        Yields:
            Messages from the stream
        """
        # Start worker if needed
        await self._ensure_worker_started()

        if not self._worker:
            raise RuntimeError("Failed to start stream worker")

        # Create listener
        queue = self._worker.get_message_queue()
        listener = await queue.create_listener()
        self._listeners[listener.listener_id] = listener

        if self._first_listener_at is None:
            self._first_listener_at = time.time()

        logger.debug(
            "stream_handle_listener_created",
            handle_id=self.handle_id,
            listener_id=listener.listener_id,
            total_listeners=len(self._listeners),
            worker_status=self._worker.status.value,
            category="streaming",
        )

        try:
            # Yield messages from listener
            async for message in listener:
                yield message

        except GeneratorExit:
            # Client disconnected
            logger.debug(
                "stream_handle_listener_disconnected",
                handle_id=self.handle_id,
                listener_id=listener.listener_id,
            )

            # Check if this will be the last listener after removal
            remaining_listeners = len(self._listeners) - 1
            if remaining_listeners == 0 and self._session_client:
                logger.debug(
                    "stream_handle_last_listener_disconnected",
                    handle_id=self.handle_id,
                    listener_id=listener.listener_id,
                    message="Last listener disconnected, will trigger SDK interrupt in cleanup",
                )

            raise

        finally:
            # Remove listener
            await self._remove_listener(listener.listener_id)

            # Check if we should trigger cleanup
            await self._check_cleanup()

    async def _ensure_worker_started(self) -> None:
        """Ensure the worker is started, creating it if needed."""
        async with self._worker_lock:
            if self._worker is None:
                # Create worker
                worker_id = f"{self.handle_id}-worker"
                self._worker = StreamWorker(
                    worker_id=worker_id,
                    message_iterator=self._message_iterator,
                    session_id=self.session_id,
                    request_id=self.request_id,
                    session_client=self._session_client,
                    stream_handle=self,  # Pass self for message tracking
                )

                # Start worker
                await self._worker.start()

                logger.debug(
                    "stream_handle_worker_created",
                    handle_id=self.handle_id,
                    worker_id=worker_id,
                    session_id=self.session_id,
                    category="streaming",
                )

    async def _remove_listener(self, listener_id: str) -> None:
        """Remove a listener and clean it up.

        Args:
            listener_id: ID of the listener to remove
        """
        if listener_id in self._listeners:
            listener = self._listeners.pop(listener_id)
            listener.close()

            if self._worker:
                queue = self._worker.get_message_queue()
                await queue.remove_listener(listener_id)

            logger.debug(
                "stream_handle_listener_removed",
                handle_id=self.handle_id,
                listener_id=listener_id,
                remaining_listeners=len(self._listeners),
                category="streaming",
            )

    async def _check_cleanup(self) -> None:
        """Check if cleanup is needed when no listeners remain."""
        async with self._worker_lock:
            if len(self._listeners) == 0 and self._worker:
                worker_status = self._worker.status.value

                # Check if worker has already completed naturally
                if worker_status in ("completed", "error"):
                    logger.debug(
                        "stream_handle_worker_already_finished",
                        handle_id=self.handle_id,
                        worker_status=worker_status,
                        message="Worker already finished, no interrupt needed",
                    )
                    return

                # Send shutdown signal to any remaining queue listeners before interrupt
                logger.debug(
                    "stream_handle_shutting_down_queue_before_interrupt",
                    handle_id=self.handle_id,
                    message="Sending shutdown signal to queue listeners before interrupt",
                )
                queue = self._worker.get_message_queue()
                await queue.broadcast_shutdown()

                # No more listeners - trigger interrupt if session client available and worker is still running
                if self._session_client:
                    # Check if worker is already stopped/interrupted - no need to interrupt SDK
                    if self._worker and self._worker.status.value in (
                        "interrupted",
                        "completed",
                        "error",
                    ):
                        logger.debug(
                            "stream_handle_worker_already_stopped",
                            handle_id=self.handle_id,
                            worker_status=worker_status,
                            message="Worker already stopped, skipping SDK interrupt entirely",
                        )
                        # Still stop the worker to ensure cleanup
                        if self._worker:
                            logger.trace(
                                "stream_handle_stopping_worker_direct",
                                handle_id=self.handle_id,
                                message="Stopping worker directly since SDK interrupt not needed",
                            )
                            try:
                                await self._worker.stop(timeout=self._interrupt_timeout)
                            except Exception as worker_error:
                                logger.warning(
                                    "stream_handle_worker_stop_error",
                                    handle_id=self.handle_id,
                                    error=str(worker_error),
                                    message="Worker stop failed but continuing",
                                )
                    else:
                        logger.debug(
                            "stream_handle_all_listeners_disconnected",
                            handle_id=self.handle_id,
                            worker_status=worker_status,
                            message="All listeners disconnected, triggering SDK interrupt",
                        )

                        # Schedule interrupt using a background task with timeout control
                        try:
                            # Create a background task with proper timeout and error handling
                            await create_managed_task(
                                self._safe_interrupt_with_timeout(),
                                name=f"stream_interrupt_{self.handle_id}",
                                creator="StreamHandle",
                            )
                            logger.debug(
                                "stream_handle_interrupt_scheduled",
                                handle_id=self.handle_id,
                                message="SDK interrupt scheduled with timeout control",
                            )
                        except Exception as e:
                            logger.error(
                                "stream_handle_interrupt_schedule_error",
                                handle_id=self.handle_id,
                                error=str(e),
                                message="Failed to schedule SDK interrupt",
                            )
                else:
                    # No more listeners - worker continues but messages are discarded
                    logger.debug(
                        "stream_handle_no_listeners",
                        handle_id=self.handle_id,
                        worker_status=worker_status,
                        message="Worker continues without listeners",
                    )

                # Don't stop the worker - let it complete naturally
                # This ensures proper stream completion and interrupt capability

    async def _safe_interrupt_with_timeout(self) -> None:
        """Safely trigger session client interrupt with proper timeout and error handling."""
        if not self._session_client:
            return

        try:
            # Call SDK interrupt first - let it handle stream cleanup gracefully
            logger.debug(
                "stream_handle_calling_sdk_interrupt",
                handle_id=self.handle_id,
                message="Calling SDK interrupt to gracefully stop stream",
            )

            await asyncio.wait_for(
                self._session_client.interrupt(),
                timeout=self._interrupt_timeout,  # Configurable timeout for stream handle initiated interrupts
            )
            logger.debug(
                "stream_handle_interrupt_completed",
                handle_id=self.handle_id,
                message="SDK interrupt completed successfully",
            )

            # Stop our worker after SDK interrupt to ensure it's not blocking the session
            if self._worker:
                logger.trace(
                    "stream_handle_stopping_worker_after_interrupt",
                    handle_id=self.handle_id,
                    message="Stopping worker to free up session for reuse",
                )
                try:
                    await self._worker.stop(timeout=self._interrupt_timeout)
                except Exception as worker_error:
                    logger.warning(
                        "stream_handle_worker_stop_error",
                        handle_id=self.handle_id,
                        error=str(worker_error),
                        message="Worker stop failed but continuing",
                    )

        except TimeoutError:
            logger.warning(
                "stream_handle_interrupt_timeout",
                handle_id=self.handle_id,
                message=f"SDK interrupt timed out after {self._interrupt_timeout} seconds, falling back to worker stop",
            )

            # Fallback: Stop our worker manually if SDK interrupt timed out
            if self._worker:
                logger.trace(
                    "stream_handle_fallback_worker_stop",
                    handle_id=self.handle_id,
                    message="SDK interrupt timed out, stopping worker as fallback",
                )
                try:
                    await self._worker.stop(timeout=self._interrupt_timeout)
                except Exception as worker_error:
                    logger.warning(
                        "stream_handle_fallback_worker_stop_error",
                        handle_id=self.handle_id,
                        error=str(worker_error),
                        message="Fallback worker stop also failed",
                    )

        except Exception as e:
            logger.error(
                "stream_handle_interrupt_failed",
                handle_id=self.handle_id,
                error=str(e),
                error_type=type(e).__name__,
                message="SDK interrupt failed with error",
            )

            # Fallback: Stop our worker manually if SDK interrupt failed
            if self._worker:
                logger.trace(
                    "stream_handle_fallback_worker_stop_after_error",
                    handle_id=self.handle_id,
                    message="SDK interrupt failed, stopping worker as fallback",
                )
                try:
                    await self._worker.stop(timeout=self._interrupt_timeout)
                except Exception as worker_error:
                    logger.warning(
                        "stream_handle_fallback_worker_stop_error",
                        handle_id=self.handle_id,
                        error=str(worker_error),
                        message="Fallback worker stop also failed",
                    )

    async def interrupt(self) -> bool:
        """Interrupt the stream.

        Returns:
            True if interrupted successfully
        """
        if not self._worker:
            logger.warning(
                "stream_handle_interrupt_no_worker",
                handle_id=self.handle_id,
            )
            return False

        logger.debug(
            "stream_handle_interrupting",
            handle_id=self.handle_id,
            worker_status=self._worker.status.value,
            active_listeners=len(self._listeners),
        )

        try:
            # Stop the worker
            await self._worker.stop(timeout=self._interrupt_timeout)

            # Close all listeners
            for listener in self._listeners.values():
                listener.close()
            self._listeners.clear()

            logger.trace(
                "stream_handle_interrupted",
                handle_id=self.handle_id,
            )
            return True

        except Exception as e:
            logger.error(
                "stream_handle_interrupt_error",
                handle_id=self.handle_id,
                error=str(e),
            )
            return False

    async def wait_for_completion(self, timeout: float | None = None) -> bool:
        """Wait for the stream to complete.

        Args:
            timeout: Optional timeout in seconds

        Returns:
            True if completed, False if timed out
        """
        if not self._worker:
            return True

        return await self._worker.wait_for_completion(timeout)

    def get_stats(self) -> dict[str, Any]:
        """Get stream handle statistics.

        Returns:
            Dictionary of statistics
        """
        stats = {
            "handle_id": self.handle_id,
            "session_id": self.session_id,
            "request_id": self.request_id,
            "active_listeners": len(self._listeners),
            "lifetime_seconds": time.time() - self._created_at,
            "time_to_first_listener": (
                self._first_listener_at - self._created_at
                if self._first_listener_at
                else None
            ),
        }

        if self._worker:
            worker_stats = self._worker.get_stats()
            stats["worker_stats"] = worker_stats  # type: ignore[assignment]
        else:
            stats["worker_stats"] = None

        return stats

    @property
    def has_active_listeners(self) -> bool:
        """Check if there are any active listeners."""
        return len(self._listeners) > 0

    @property
    def worker_status(self) -> WorkerStatus | None:
        """Get the worker status if worker exists."""
        return self._worker.status if self._worker else None

    # Message lifecycle tracking methods for stale detection

    def on_first_chunk_received(self) -> None:
        """Called when SystemMessage(init) is received - first chunk."""
        if self._first_chunk_received_at is None:
            self._first_chunk_received_at = time.time()
            self._last_activity_at = self._first_chunk_received_at
            logger.debug(
                "stream_handle_first_chunk_received",
                handle_id=self.handle_id,
                session_id=self.session_id,
            )

    def on_message_received(self, message: Any) -> None:
        """Called when any message is received to update activity."""
        self._last_activity_at = time.time()

    def on_completion(self) -> None:
        """Called when ResultMessage is received - stream completed."""
        if not self._has_result_message:
            self._has_result_message = True
            self._completed_at = time.time()
            self._last_activity_at = self._completed_at
            logger.debug(
                "stream_handle_completed",
                handle_id=self.handle_id,
                session_id=self.session_id,
            )

    @property
    def is_completed(self) -> bool:
        """Check if stream has completed (received ResultMessage)."""
        return self._has_result_message

    @property
    def has_first_chunk(self) -> bool:
        """Check if stream has received first chunk (SystemMessage init)."""
        return self._first_chunk_received_at is not None

    @property
    def idle_seconds(self) -> float:
        """Get seconds since last activity."""
        return time.time() - self._last_activity_at

    def is_stale(self) -> bool:
        """Check if stream is stale based on configurable timeout logic.

        Returns:
            True if stream should be considered stale
        """
        if self.is_completed:
            # Completed streams are never stale
            return False

        if not self.has_first_chunk:
            # No first chunk received - configurable timeout
            return self.idle_seconds > self._first_chunk_timeout
        else:
            # First chunk received but not completed - configurable timeout
            return self.idle_seconds > self._ongoing_timeout

    def is_first_chunk_timeout(self) -> bool:
        """Check if this is specifically a first chunk timeout.

        Returns:
            True if no first chunk received and timeout exceeded
        """
        return (
            not self.has_first_chunk and self.idle_seconds > self._first_chunk_timeout
        )

    def is_ongoing_timeout(self) -> bool:
        """Check if this is an ongoing stream timeout.

        Returns:
            True if first chunk received but ongoing timeout exceeded
        """
        return (
            self.has_first_chunk
            and not self.is_completed
            and self.idle_seconds > self._ongoing_timeout
        )
