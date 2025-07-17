"""
Server-Sent Events (SSE) event manager for real-time dashboard updates.

This module provides centralized SSE connection management and event broadcasting
for real-time dashboard notifications when requests start, complete, or error.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections.abc import AsyncGenerator
from typing import Any

import structlog


logger = structlog.get_logger(__name__)


class SSEEventManager:
    """
    Centralized SSE connection management and event broadcasting.

    Manages multiple SSE connections and broadcasts events to all connected clients.
    Uses bounded queues to prevent memory issues with slow clients.
    """

    def __init__(self, max_queue_size: int = 100) -> None:
        """
        Initialize SSE event manager.

        Args:
            max_queue_size: Maximum events to queue per connection before dropping
        """
        self._connections: dict[str, asyncio.Queue[dict[str, Any]]] = {}
        self._lock = asyncio.Lock()
        self._max_queue_size = max_queue_size

    async def add_connection(
        self, connection_id: str | None = None, request_id: str | None = None
    ) -> AsyncGenerator[str, None]:
        """
        Add SSE connection and yield events as JSON strings.

        Args:
            connection_id: Unique connection identifier (generated if not provided)
            request_id: Request identifier for tracking

        Yields:
            JSON-formatted event strings for SSE
        """
        if connection_id is None:
            connection_id = str(uuid.uuid4())

        # Create bounded queue for this connection
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(
            maxsize=self._max_queue_size
        )

        async with self._lock:
            self._connections[connection_id] = queue

        logger.debug(
            "sse_connection_added", connection_id=connection_id, request_id=request_id
        )

        try:
            # Send initial connection event
            connection_event = {
                "type": "connection",
                "message": "Connected to metrics stream",
                "connection_id": connection_id,
                "timestamp": time.time(),
            }
            yield self._format_sse_event(connection_event)

            while True:
                # Wait for next event
                event = await queue.get()

                # Check for special disconnect event
                if event.get("type") == "_disconnect":
                    break

                # Yield formatted event
                yield self._format_sse_event(event)

        except asyncio.CancelledError:
            logger.debug("sse_connection_cancelled", connection_id=connection_id)
            raise
        except GeneratorExit:
            logger.debug("sse_connection_generator_exit", connection_id=connection_id)
            raise
        finally:
            # Clean up connection
            await self._cleanup_connection(connection_id)

            # Send disconnect event only if not in shutdown
            try:
                disconnect_event = {
                    "type": "disconnect",
                    "message": "Stream disconnected",
                    "connection_id": connection_id,
                    "timestamp": time.time(),
                }
                yield self._format_sse_event(disconnect_event)
            except (GeneratorExit, asyncio.CancelledError):
                # Ignore errors during cleanup
                pass

    async def emit_event(self, event_type: str, data: dict[str, Any]) -> None:
        """
        Broadcast event to all connected clients.

        Args:
            event_type: Type of event (request_start, request_complete, request_error)
            data: Event data dictionary
        """
        if not self._connections:
            return  # No connected clients

        event = {
            "type": event_type,
            "data": data,
            "timestamp": time.time(),
        }

        async with self._lock:
            # Get copy of connections to avoid modification during iteration
            connections = dict(self._connections)

        # Broadcast to all connections
        failed_connections = []

        for connection_id, queue in connections.items():
            try:
                # Try to put event in queue without blocking
                queue.put_nowait(event)
            except asyncio.QueueFull:
                # Queue is full, handle overflow
                try:
                    # Try to drop oldest event and add overflow indicator
                    queue.get_nowait()  # Remove oldest
                    overflow_event = {
                        "type": "overflow",
                        "message": "Event queue full, some events dropped",
                        "timestamp": time.time(),
                    }
                    try:
                        queue.put_nowait(overflow_event)
                        queue.put_nowait(event)
                    except asyncio.QueueFull:
                        # Still full after dropping, connection is problematic
                        failed_connections.append(connection_id)
                        continue

                    logger.warning(
                        "sse_queue_overflow",
                        connection_id=connection_id,
                        max_queue_size=self._max_queue_size,
                    )
                except asyncio.QueueEmpty:
                    # Queue became empty, try again
                    try:
                        queue.put_nowait(event)
                    except asyncio.QueueFull:
                        # Still full, connection is problematic
                        failed_connections.append(connection_id)
                except Exception as e:
                    logger.error(
                        "sse_overflow_error",
                        connection_id=connection_id,
                        error=str(e),
                    )
                    failed_connections.append(connection_id)
            except Exception as e:
                logger.error(
                    "sse_broadcast_error",
                    connection_id=connection_id,
                    error=str(e),
                )
                failed_connections.append(connection_id)

        # Clean up failed connections
        for connection_id in failed_connections:
            await self._cleanup_connection(connection_id)

        if failed_connections:
            logger.debug(
                "sse_connections_cleaned",
                failed_count=len(failed_connections),
                active_count=len(self._connections),
            )

    async def disconnect_all(self) -> None:
        """Disconnect all active connections gracefully."""
        async with self._lock:
            connections = dict(self._connections)

        for connection_id, queue in connections.items():
            try:
                # Send disconnect signal
                disconnect_signal = {"type": "_disconnect"}
                queue.put_nowait(disconnect_signal)
            except asyncio.QueueFull:
                # Queue is full, force cleanup
                await self._cleanup_connection(connection_id)
            except Exception as e:
                logger.error(
                    "sse_disconnect_error",
                    connection_id=connection_id,
                    error=str(e),
                )

        logger.debug("sse_all_connections_disconnected")

    async def _cleanup_connection(self, connection_id: str) -> None:
        """Remove connection from active connections."""
        async with self._lock:
            if connection_id in self._connections:
                del self._connections[connection_id]
                logger.debug("sse_connection_removed", connection_id=connection_id)

    def _format_sse_event(self, event: dict[str, Any]) -> str:
        """Format event as SSE data string."""
        try:
            json_data = json.dumps(event, default=self._json_serializer)
            return f"data: {json_data}\n\n"
        except (TypeError, ValueError) as e:
            logger.error("sse_format_error", error=str(e), event_type=event.get("type"))
            # Return error event instead
            error_event = {
                "type": "error",
                "message": "Failed to format event",
                "timestamp": time.time(),
            }
            json_data = json.dumps(error_event, default=self._json_serializer)
            return f"data: {json_data}\n\n"

    def _json_serializer(self, obj: Any) -> Any:
        """Custom JSON serializer for datetime and other objects."""
        from datetime import datetime

        if isinstance(obj, datetime):
            return obj.isoformat()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    async def get_connection_count(self) -> int:
        """Get number of active connections."""
        async with self._lock:
            return len(self._connections)

    async def get_connection_info(self) -> dict[str, Any]:
        """Get connection status information."""
        async with self._lock:
            return {
                "active_connections": len(self._connections),
                "max_queue_size": self._max_queue_size,
                "connection_ids": list(self._connections.keys()),
            }


# Global SSE event manager instance
_global_sse_manager: SSEEventManager | None = None


def get_sse_manager() -> SSEEventManager:
    """Get or create global SSE event manager."""
    global _global_sse_manager

    if _global_sse_manager is None:
        _global_sse_manager = SSEEventManager()

    return _global_sse_manager


async def emit_sse_event(event_type: str, data: dict[str, Any]) -> None:
    """
    Convenience function to emit SSE event using global manager.

    Args:
        event_type: Type of event (request_start, request_complete, request_error)
        data: Event data dictionary
    """
    try:
        manager = get_sse_manager()
        await manager.emit_event(event_type, data)
    except Exception as e:
        # Log error but don't fail the request
        logger.debug("sse_emit_failed", event_type=event_type, error=str(e))


async def cleanup_sse_manager() -> None:
    """Clean up global SSE manager and disconnect all clients."""
    global _global_sse_manager

    if _global_sse_manager is not None:
        await _global_sse_manager.disconnect_all()
        _global_sse_manager = None
        logger.debug("sse_manager_cleaned_up")
