"""Stream worker for consuming Claude SDK messages and distributing via queue."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from enum import Enum
from typing import TYPE_CHECKING, Any

from ccproxy.core.async_task_manager import create_managed_task
from ccproxy.core.logging import get_plugin_logger

from . import models as sdk_models
from .exceptions import StreamTimeoutError
from .message_queue import MessageQueue


if TYPE_CHECKING:
    from .session_client import SessionClient
    from .stream_handle import StreamHandle

logger = get_plugin_logger()


class WorkerStatus(str, Enum):
    """Status of the stream worker."""

    IDLE = "idle"
    STARTING = "starting"
    RUNNING = "running"
    DRAINING = "draining"
    COMPLETED = "completed"
    ERROR = "error"
    INTERRUPTED = "interrupted"


class StreamWorker:
    """Worker that consumes messages from Claude SDK and distributes via queue."""

    def __init__(
        self,
        worker_id: str,
        message_iterator: AsyncIterator[Any],
        session_id: str | None = None,
        request_id: str | None = None,
        session_client: SessionClient | None = None,
        stream_handle: StreamHandle | None = None,
    ):
        """Initialize the stream worker.

        Args:
            worker_id: Unique identifier for this worker
            message_iterator: Async iterator of SDK messages
            session_id: Optional session ID for logging
            request_id: Optional request ID for logging
            session_client: Optional session client for state management
            stream_handle: Optional stream handle for message lifecycle tracking
        """
        self.worker_id = worker_id
        self._message_iterator = message_iterator
        self.session_id = session_id
        self.request_id = request_id
        self._session_client = session_client
        self._stream_handle = stream_handle

        # Worker state
        self.status = WorkerStatus.IDLE
        self._message_queue = MessageQueue()
        self._worker_task: asyncio.Task[None] | None = None
        self._started_at: float | None = None
        self._completed_at: float | None = None

        # Statistics
        self._total_messages = 0
        self._messages_delivered = 0
        self._messages_discarded = 0
        self._last_message_time: float | None = None

    async def start(self) -> None:
        """Start the worker task."""
        if self.status != WorkerStatus.IDLE:
            logger.warning(
                "stream_worker_already_started",
                worker_id=self.worker_id,
                status=self.status,
            )
            return

        self.status = WorkerStatus.STARTING
        self._started_at = time.time()

        # Create worker task
        self._worker_task = await create_managed_task(
            self._run_worker(),
            name=f"stream_worker_{self.worker_id}",
            creator="StreamWorker",
        )

        logger.debug(
            "stream_worker_started",
            worker_id=self.worker_id,
            session_id=self.session_id,
            request_id=self.request_id,
        )

    async def stop(self, timeout: float = 5.0) -> None:
        """Stop the worker gracefully.

        Args:
            timeout: Maximum time to wait for worker to stop
        """
        if self._worker_task and not self._worker_task.done():
            logger.debug(
                "stream_worker_stopping",
                worker_id=self.worker_id,
                timeout=timeout,
            )

            # Cancel the worker task
            self._worker_task.cancel()

            try:
                # Use asyncio.wait instead of wait_for to handle cancelled tasks properly
                done, pending = await asyncio.wait(
                    [self._worker_task],
                    timeout=timeout,
                    return_when=asyncio.ALL_COMPLETED,
                )

                if pending:
                    logger.warning(
                        "stream_worker_stop_timeout",
                        worker_id=self.worker_id,
                        timeout=timeout,
                    )
                elif done:
                    # Task completed (likely with CancelledError)
                    logger.debug(
                        "stream_worker_stopped",
                        worker_id=self.worker_id,
                        task_cancelled=self._worker_task.cancelled(),
                    )

            except Exception as e:
                logger.warning(
                    "stream_worker_stop_error",
                    worker_id=self.worker_id,
                    error=str(e),
                    error_type=type(e).__name__,
                )

    async def wait_for_completion(self, timeout: float | None = None) -> bool:
        """Wait for the worker to complete.

        Args:
            timeout: Optional timeout in seconds

        Returns:
            True if completed successfully, False if timed out
        """
        if not self._worker_task:
            return True

        try:
            if timeout:
                await asyncio.wait_for(self._worker_task, timeout=timeout)
            else:
                await self._worker_task
            return True
        except TimeoutError:
            return False

    def get_message_queue(self) -> MessageQueue:
        """Get the message queue for creating listeners.

        Returns:
            The worker's message queue
        """
        return self._message_queue

    async def _run_worker(self) -> None:
        """Main worker loop that consumes messages and distributes them."""
        try:
            self.status = WorkerStatus.RUNNING

            logger.debug(
                "stream_worker_consuming",
                worker_id=self.worker_id,
                session_id=self.session_id,
            )

            async for message in self._message_iterator:
                self._total_messages += 1
                self._last_message_time = time.time()

                # Check if we have listeners
                if await self._message_queue.has_listeners():
                    # Broadcast to all listeners
                    delivered_count = await self._message_queue.broadcast(message)
                    self._messages_delivered += delivered_count

                    logger.debug(
                        "stream_worker_message_delivered",
                        worker_id=self.worker_id,
                        message_type=type(message).__name__,
                        delivered_to=delivered_count,
                        total_messages=self._total_messages,
                    )
                else:
                    # No listeners - discard message
                    self._messages_discarded += 1

                    logger.debug(
                        "stream_worker_message_discarded",
                        worker_id=self.worker_id,
                        message_type=type(message).__name__,
                        total_messages=self._total_messages,
                        total_discarded=self._messages_discarded,
                    )

                # Update stream handle with message lifecycle tracking
                if self._stream_handle:
                    # Track all message activity
                    self._stream_handle.on_message_received(message)

                    # Track first chunk (SystemMessage with init subtype)
                    if (
                        isinstance(message, sdk_models.SystemMessage)
                        and hasattr(message, "subtype")
                        and message.subtype == "init"
                    ):
                        self._stream_handle.on_first_chunk_received()

                    # Track completion (ResultMessage)
                    elif isinstance(message, sdk_models.ResultMessage):
                        self._stream_handle.on_completion()

                # Update session client if we have one
                if self._session_client and isinstance(
                    message, sdk_models.ResultMessage
                ):
                    self._session_client.sdk_session_id = message.session_id

            # Stream completed successfully
            self.status = WorkerStatus.COMPLETED
            await self._message_queue.broadcast_complete()

            logger.debug(
                "stream_worker_completed",
                worker_id=self.worker_id,
                total_messages=self._total_messages,
                messages_delivered=self._messages_delivered,
                messages_discarded=self._messages_discarded,
                duration_seconds=time.time() - (self._started_at or 0),
            )

        except asyncio.CancelledError:
            # Worker was cancelled
            self.status = WorkerStatus.INTERRUPTED
            logger.debug(
                "stream_worker_cancelled",
                worker_id=self.worker_id,
                messages_processed=self._total_messages,
            )
            raise

        except StreamTimeoutError as e:
            # Handle timeout errors gracefully - these are expected for some commands
            self.status = WorkerStatus.ERROR
            await self._message_queue.broadcast_error(e)

            logger.debug(
                "stream_worker_timeout",
                worker_id=self.worker_id,
                timeout_message=str(e),
                messages_processed=self._total_messages,
                message="Stream worker completed due to timeout - this is expected for some commands",
            )
            # Don't re-raise StreamTimeoutError to avoid unhandled task exceptions

        except Exception as e:
            # Error during processing (other than timeout)
            self.status = WorkerStatus.ERROR
            await self._message_queue.broadcast_error(e)

            logger.error(
                "stream_worker_error",
                worker_id=self.worker_id,
                error=str(e),
                error_type=type(e).__name__,
                messages_processed=self._total_messages,
            )
            raise

        finally:
            self._completed_at = time.time()

            # Clean up
            if self._session_client:
                self._session_client.has_active_stream = False

            # Close the message queue
            await self._message_queue.close()

    async def drain_remaining(self, timeout: float = 30.0) -> int:
        """Drain remaining messages without listeners.

        This is useful for ensuring the stream completes properly
        even after all listeners have disconnected.

        Args:
            timeout: Maximum time to spend draining

        Returns:
            Number of messages drained
        """
        if self.status not in (WorkerStatus.RUNNING, WorkerStatus.STARTING):
            return 0

        self.status = WorkerStatus.DRAINING
        start_time = time.time()
        drained_count = 0

        logger.debug(
            "stream_worker_draining",
            worker_id=self.worker_id,
            timeout=timeout,
        )

        try:
            # Continue consuming but without broadcasting
            async for message in self._message_iterator:
                drained_count += 1
                self._total_messages += 1
                self._messages_discarded += 1

                logger.debug(
                    "stream_worker_draining_message",
                    worker_id=self.worker_id,
                    message_type=type(message).__name__,
                    drained_count=drained_count,
                )

                # Check timeout
                if time.time() - start_time > timeout:
                    logger.warning(
                        "stream_worker_drain_timeout",
                        worker_id=self.worker_id,
                        drained_count=drained_count,
                        timeout=timeout,
                    )
                    break

                # Check for completion message
                if isinstance(message, sdk_models.ResultMessage):
                    logger.debug(
                        "stream_worker_drain_complete",
                        worker_id=self.worker_id,
                        drained_count=drained_count,
                    )
                    break

        except Exception as e:
            logger.error(
                "stream_worker_drain_error",
                worker_id=self.worker_id,
                error=str(e),
                drained_count=drained_count,
            )

        return drained_count

    def get_stats(self) -> dict[str, Any]:
        """Get worker statistics.

        Returns:
            Dictionary of worker statistics
        """
        runtime = None
        if self._started_at:
            end_time = self._completed_at or time.time()
            runtime = end_time - self._started_at

        queue_stats = self._message_queue.get_stats()

        return {
            "worker_id": self.worker_id,
            "status": self.status.value,
            "session_id": self.session_id,
            "request_id": self.request_id,
            "total_messages": self._total_messages,
            "messages_delivered": self._messages_delivered,
            "messages_discarded": self._messages_discarded,
            "runtime_seconds": runtime,
            "queue_stats": queue_stats,
        }
