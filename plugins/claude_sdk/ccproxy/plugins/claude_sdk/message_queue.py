"""Message queue system for broadcasting SDK messages to multiple listeners."""

from __future__ import annotations

import asyncio
import contextlib
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TypeVar

from ccproxy.core.logging import get_plugin_logger


logger = get_plugin_logger()

T = TypeVar("T")


class MessageType(str, Enum):
    """Types of messages that can be sent through the queue."""

    DATA = "data"
    ERROR = "error"
    COMPLETE = "complete"
    SHUTDOWN = "shutdown"


@dataclass
class QueueMessage:
    """Message wrapper for queue communication."""

    type: MessageType
    data: Any = None
    error: Exception | None = None
    timestamp: float = field(default_factory=time.time)


class QueueListener:
    """Individual listener that consumes messages from the queue."""

    def __init__(self, listener_id: str | None = None):
        """Initialize a queue listener.

        Args:
            listener_id: Optional ID for the listener, generated if not provided
        """
        self.listener_id = listener_id or str(uuid.uuid4())
        self._queue: asyncio.Queue[QueueMessage] = asyncio.Queue()
        self._closed = False
        self._created_at = time.time()

    async def get_message(self) -> QueueMessage:
        """Get the next message from the queue.

        Returns:
            The next queued message

        Raises:
            asyncio.QueueEmpty: If queue is empty and closed
        """
        if self._closed and self._queue.empty():
            raise asyncio.QueueEmpty("Listener is closed")

        return await self._queue.get()

    async def put_message(self, message: QueueMessage) -> None:
        """Put a message into this listener's queue.

        Args:
            message: Message to queue
        """
        if not self._closed:
            await self._queue.put(message)

    def close(self) -> None:
        """Close the listener, preventing new messages."""
        self._closed = True
        # Put a shutdown message to unblock any waiting consumers
        with contextlib.suppress(asyncio.QueueFull):
            self._queue.put_nowait(QueueMessage(type=MessageType.SHUTDOWN))

    @property
    def is_closed(self) -> bool:
        """Check if the listener is closed."""
        return self._closed

    @property
    def queue_size(self) -> int:
        """Get the current queue size."""
        return self._queue.qsize()

    async def __aiter__(self) -> AsyncIterator[Any]:
        """Async iterator interface for consuming messages."""
        while True:
            try:
                message = await self.get_message()

                if message.type == MessageType.SHUTDOWN:
                    break
                elif message.type == MessageType.ERROR:
                    if message.error:
                        raise message.error
                    break
                elif message.type == MessageType.COMPLETE:
                    break
                else:
                    yield message.data
            except asyncio.QueueEmpty:
                break


class MessageQueue:
    """Message queue that broadcasts to multiple listeners with discard logic."""

    def __init__(self, max_listeners: int = 100):
        """Initialize the message queue.

        Args:
            max_listeners: Maximum number of concurrent listeners
        """
        self._listeners: dict[str, QueueListener] = {}
        self._lock = asyncio.Lock()
        self._max_listeners = max_listeners
        self._total_messages_received = 0
        self._total_messages_delivered = 0
        self._total_messages_discarded = 0
        self._created_at = time.time()

    async def create_listener(self, listener_id: str | None = None) -> QueueListener:
        """Create a new listener for this queue.

        Args:
            listener_id: Optional ID for the listener

        Returns:
            A new QueueListener instance

        Raises:
            RuntimeError: If max listeners exceeded
        """
        async with self._lock:
            if len(self._listeners) >= self._max_listeners:
                raise RuntimeError(
                    f"Maximum listeners ({self._max_listeners}) exceeded"
                )

            listener = QueueListener(listener_id)
            self._listeners[listener.listener_id] = listener

            logger.trace(
                "message_queue_listener_added",
                listener_id=listener.listener_id,
                active_listeners=len(self._listeners),
            )

            return listener

    async def remove_listener(self, listener_id: str) -> None:
        """Remove a listener from the queue.

        Args:
            listener_id: ID of the listener to remove
        """
        async with self._lock:
            if listener_id in self._listeners:
                listener = self._listeners.pop(listener_id)
                listener.close()

                logger.trace(
                    "message_queue_listener_removed",
                    listener_id=listener_id,
                    active_listeners=len(self._listeners),
                    listener_queue_size=listener.queue_size,
                )

    async def has_listeners(self) -> bool:
        """Check if any active listeners exist.

        Returns:
            True if at least one listener is registered
        """
        async with self._lock:
            return len(self._listeners) > 0

    async def get_listener_count(self) -> int:
        """Get the current number of active listeners.

        Returns:
            Number of active listeners
        """
        async with self._lock:
            return len(self._listeners)

    async def broadcast(self, message: Any) -> int:
        """Broadcast a message to all active listeners.

        Args:
            message: The message to broadcast

        Returns:
            Number of listeners that received the message
        """
        self._total_messages_received += 1

        async with self._lock:
            if not self._listeners:
                self._total_messages_discarded += 1
                logger.debug(
                    "message_queue_discard",
                    reason="no_listeners",
                    message_type=type(message).__name__,
                )
                return 0

            # Create queue message
            queue_msg = QueueMessage(type=MessageType.DATA, data=message)

            # Broadcast to all listeners
            delivered_count = 0
            for listener_id, listener in list(self._listeners.items()):
                if listener.is_closed:
                    # Remove closed listeners
                    self._listeners.pop(listener_id, None)
                    continue

                try:
                    # Use put_nowait to avoid blocking
                    listener._queue.put_nowait(queue_msg)
                    delivered_count += 1
                except asyncio.QueueFull:
                    logger.warning(
                        "message_queue_listener_full",
                        listener_id=listener_id,
                        queue_size=listener.queue_size,
                    )

            self._total_messages_delivered += delivered_count

            if delivered_count == 0:
                self._total_messages_discarded += 1

            logger.trace(
                "message_queue_broadcast",
                listeners_count=len(self._listeners),
                delivered_count=delivered_count,
                message_type=type(message).__name__,
            )

            return delivered_count

    async def broadcast_error(self, error: Exception) -> None:
        """Broadcast an error to all listeners.

        Args:
            error: The error to broadcast
        """
        async with self._lock:
            queue_msg = QueueMessage(type=MessageType.ERROR, error=error)

            for listener in self._listeners.values():
                if not listener.is_closed:
                    with contextlib.suppress(asyncio.QueueFull):
                        listener._queue.put_nowait(queue_msg)

            logger.trace(
                "message_queue_broadcast_error",
                error_type=type(error).__name__,
                listeners_count=len(self._listeners),
            )

    async def broadcast_complete(self) -> None:
        """Broadcast completion signal to all listeners."""
        async with self._lock:
            queue_msg = QueueMessage(type=MessageType.COMPLETE)

            for listener in self._listeners.values():
                if not listener.is_closed:
                    with contextlib.suppress(asyncio.QueueFull):
                        listener._queue.put_nowait(queue_msg)

            logger.trace(
                "message_queue_broadcast_complete",
                listeners_count=len(self._listeners),
            )

    async def broadcast_shutdown(self) -> None:
        """Broadcast shutdown signal to all listeners (for interrupts)."""
        async with self._lock:
            queue_msg = QueueMessage(type=MessageType.SHUTDOWN)

            for listener in self._listeners.values():
                if not listener.is_closed:
                    with contextlib.suppress(asyncio.QueueFull):
                        listener._queue.put_nowait(queue_msg)

            logger.trace(
                "message_queue_broadcast_shutdown",
                listeners_count=len(self._listeners),
                message="Shutdown signal sent to all listeners due to interrupt",
            )

    async def close(self) -> None:
        """Close the message queue and all listeners."""
        async with self._lock:
            # Send shutdown to all listeners
            queue_msg = QueueMessage(type=MessageType.SHUTDOWN)

            for listener in self._listeners.values():
                listener.close()

            self._listeners.clear()

            logger.debug(
                "message_queue_closed",
                total_messages_received=self._total_messages_received,
                total_messages_delivered=self._total_messages_delivered,
                total_messages_discarded=self._total_messages_discarded,
                lifetime_seconds=time.time() - self._created_at,
            )

    def get_stats(self) -> dict[str, Any]:
        """Get queue statistics.

        Returns:
            Dictionary of queue statistics
        """
        return {
            "active_listeners": len(self._listeners),
            "max_listeners": self._max_listeners,
            "total_messages_received": self._total_messages_received,
            "total_messages_delivered": self._total_messages_delivered,
            "total_messages_discarded": self._total_messages_discarded,
            "lifetime_seconds": time.time() - self._created_at,
            "delivery_rate": (
                self._total_messages_delivered / self._total_messages_received
                if self._total_messages_received > 0
                else 0.0
            ),
        }
