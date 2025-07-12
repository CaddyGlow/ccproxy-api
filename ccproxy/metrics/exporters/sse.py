"""
Server-Sent Events (SSE) exporter for real-time metrics streaming.

This module provides SSE-based real-time streaming of metrics data
for dashboards and monitoring applications.
"""

import asyncio
import json
import logging
import time
import weakref
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from datetime import UTC, datetime, timedelta
from typing import Any, Optional
from uuid import uuid4

from ..models import MetricRecord, MetricType
from ..storage.base import MetricsStorage
from .base import BaseMetricsExporter, ExporterError


logger = logging.getLogger(__name__)


class SSEConnection:
    """Represents a single SSE connection with subscription preferences."""

    def __init__(
        self,
        connection_id: str,
        queue_size: int = 100,
        heartbeat_interval: float = 30.0,
    ):
        """
        Initialize SSE connection.

        Args:
            connection_id: Unique identifier for the connection
            queue_size: Maximum size of the event queue
            heartbeat_interval: Interval between heartbeat messages in seconds
        """
        self.connection_id = connection_id
        self.queue: asyncio.Queue[str] = asyncio.Queue(maxsize=queue_size)
        self.created_at = datetime.now(UTC)
        self.last_heartbeat = time.time()
        self.heartbeat_interval = heartbeat_interval
        self.is_active = True

        # Subscription filters
        self.metric_types: set[MetricType] | None = None
        self.user_id: str | None = None
        self.session_id: str | None = None
        self.subscription_types: set[str] = {"live"}  # live, summary, time_series

    def matches_filter(self, metric: MetricRecord) -> bool:
        """Check if a metric matches this connection's filters."""
        if self.metric_types and metric.metric_type not in self.metric_types:
            return False
        if self.user_id and metric.user_id != self.user_id:
            return False
        return not (self.session_id and metric.session_id != self.session_id)

    async def send_event(self, event_data: str) -> bool:
        """
        Send an event to this connection.

        Args:
            event_data: SSE formatted event data

        Returns:
            True if sent successfully, False if connection is full/inactive
        """
        if not self.is_active:
            return False

        try:
            self.queue.put_nowait(event_data)
            return True
        except asyncio.QueueFull:
            logger.warning(f"SSE connection {self.connection_id} queue is full")
            return False

    async def send_heartbeat(self) -> bool:
        """Send a heartbeat message."""
        current_time = time.time()
        if current_time - self.last_heartbeat >= self.heartbeat_interval:
            heartbeat_event = self._format_sse_event(
                event_type="heartbeat",
                data={
                    "timestamp": datetime.now(UTC).isoformat(),
                    "connection_id": self.connection_id,
                },
            )
            success = await self.send_event(heartbeat_event)
            if success:
                self.last_heartbeat = current_time
            return success
        return True

    def close(self) -> None:
        """Close the connection."""
        self.is_active = False

    @staticmethod
    def _format_sse_event(
        data: dict[str, Any],
        event_type: str = "data",
        event_id: str | None = None,
    ) -> str:
        """Format data as SSE event."""
        lines = []

        if event_id:
            lines.append(f"id: {event_id}")

        if event_type != "data":
            lines.append(f"event: {event_type}")

        json_data = json.dumps(data, separators=(",", ":"))
        lines.append(f"data: {json_data}")
        lines.append("")  # Empty line to end the event

        return "\n".join(lines)


class SSEMetricsExporter(BaseMetricsExporter):
    """
    SSE-based metrics exporter for real-time streaming.

    This exporter maintains active SSE connections and broadcasts
    metrics events in real-time to subscribed clients.
    """

    def __init__(
        self,
        storage: MetricsStorage,
        max_connections: int = 100,
        connection_timeout: float = 300.0,  # 5 minutes
        cleanup_interval: float = 60.0,  # 1 minute
        heartbeat_interval: float = 30.0,  # 30 seconds
    ):
        """
        Initialize the SSE metrics exporter.

        Args:
            storage: Metrics storage backend
            max_connections: Maximum number of concurrent connections
            connection_timeout: Connection timeout in seconds
            cleanup_interval: Interval for cleaning up stale connections
            heartbeat_interval: Interval between heartbeat messages
        """
        self.storage = storage
        self.max_connections = max_connections
        self.connection_timeout = connection_timeout
        self.cleanup_interval = cleanup_interval
        self.heartbeat_interval = heartbeat_interval

        # Connection management
        self._connections: dict[str, SSEConnection] = {}
        self._cleanup_task: asyncio.Task[None] | None = None
        self._is_running = False

    async def start(self) -> None:
        """Start the SSE exporter background tasks."""
        if self._is_running:
            return

        self._is_running = True
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("SSE metrics exporter started")

    async def stop(self) -> None:
        """Stop the SSE exporter and close all connections."""
        self._is_running = False

        if self._cleanup_task:
            self._cleanup_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._cleanup_task

        # Close all connections
        for connection in self._connections.values():
            connection.close()
        self._connections.clear()

        logger.info("SSE metrics exporter stopped")

    @asynccontextmanager
    async def create_connection(
        self,
        metric_types: list[MetricType] | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        subscription_types: list[str] | None = None,
    ) -> AsyncIterator[tuple[str, AsyncIterator[str]]]:
        """
        Create a new SSE connection with filtering options.

        Args:
            metric_types: List of metric types to subscribe to
            user_id: Filter by specific user ID
            session_id: Filter by specific session ID
            subscription_types: Types of subscriptions (live, summary, time_series)

        Yields:
            Tuple of (connection_id, event_stream)
        """
        if len(self._connections) >= self.max_connections:
            raise ExporterError("Maximum number of connections reached")

        connection_id = str(uuid4())
        connection = SSEConnection(
            connection_id=connection_id,
            heartbeat_interval=self.heartbeat_interval,
        )

        # Apply filters
        if metric_types:
            connection.metric_types = set(metric_types)
        if user_id:
            connection.user_id = user_id
        if session_id:
            connection.session_id = session_id
        if subscription_types:
            connection.subscription_types = set(subscription_types)

        self._connections[connection_id] = connection

        try:
            # Send initial connection event
            filters_dict = {
                "metric_types": [mt.value for mt in metric_types]
                if metric_types
                else None,
                "user_id": user_id,
                "session_id": session_id,
                "subscription_types": list(subscription_types)
                if subscription_types
                else ["live"],
            }
            # Remove None values from filters
            filters_dict = {k: v for k, v in filters_dict.items() if v is not None}

            initial_event = connection._format_sse_event(
                event_type="connected",
                data={
                    "connection_id": connection_id,
                    "timestamp": datetime.now(UTC).isoformat(),
                    "filters": filters_dict,
                },
                event_id=connection_id,
            )
            await connection.send_event(initial_event)

            # Yield the connection and event stream
            yield connection_id, self._event_stream(connection)

        finally:
            # Clean up connection
            connection.close()
            if connection_id in self._connections:
                del self._connections[connection_id]
            logger.debug(f"SSE connection {connection_id} closed")

    async def _event_stream(self, connection: SSEConnection) -> AsyncIterator[str]:
        """Generate SSE events for a connection."""
        try:
            while connection.is_active and self._is_running:
                try:
                    # Wait for events with timeout for heartbeat
                    event = await asyncio.wait_for(
                        connection.queue.get(),
                        timeout=connection.heartbeat_interval,
                    )
                    yield event
                except TimeoutError:
                    # Send heartbeat
                    if not await connection.send_heartbeat():
                        break

                    # Yield heartbeat if there's one in the queue
                    if not connection.queue.empty():
                        try:
                            heartbeat = connection.queue.get_nowait()
                            yield heartbeat
                        except asyncio.QueueEmpty:
                            pass

        except Exception as e:
            logger.error(
                f"Error in SSE event stream for {connection.connection_id}: {e}"
            )
        finally:
            connection.close()

    async def broadcast_metric(self, metric: MetricRecord) -> int:
        """
        Broadcast a new metric to all matching connections.

        Args:
            metric: Metric record to broadcast

        Returns:
            Number of connections that received the metric
        """
        if not self._is_running:
            return 0

        event_data = self._format_metric_event(metric)
        broadcast_count = 0

        for connection in list(self._connections.values()):
            if (
                connection.matches_filter(metric)
                and "live" in connection.subscription_types
            ) and await connection.send_event(event_data):
                broadcast_count += 1

        return broadcast_count

    async def broadcast_summary(
        self,
        summary_data: dict[str, Any],
        user_id: str | None = None,
        session_id: str | None = None,
    ) -> int:
        """
        Broadcast a metrics summary to matching connections.

        Args:
            summary_data: Summary data to broadcast
            user_id: Filter by user ID
            session_id: Filter by session ID

        Returns:
            Number of connections that received the summary
        """
        if not self._is_running:
            return 0

        event_data = SSEConnection._format_sse_event(
            event_type="summary",
            data=summary_data,
            event_id=str(uuid4()),
        )
        broadcast_count = 0

        for connection in list(self._connections.values()):
            if "summary" in connection.subscription_types:
                # Check user/session filters
                if user_id and connection.user_id and connection.user_id != user_id:
                    continue
                if (
                    session_id
                    and connection.session_id
                    and connection.session_id != session_id
                ):
                    continue

                if await connection.send_event(event_data):
                    broadcast_count += 1

        return broadcast_count

    def _format_metric_event(self, metric: MetricRecord) -> str:
        """Format a metric record as an SSE event."""
        # Convert metric to dictionary using Pydantic serialization
        metric_dict = metric.model_dump(mode="json", exclude_none=True)

        return SSEConnection._format_sse_event(
            event_type="metric",
            data=metric_dict,
            event_id=str(metric.id),
        )

    async def _cleanup_loop(self) -> None:
        """Background task to clean up stale connections."""
        while self._is_running:
            try:
                await asyncio.sleep(self.cleanup_interval)
                await self._cleanup_stale_connections()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in SSE cleanup loop: {e}")

    async def _cleanup_stale_connections(self) -> None:
        """Remove stale and inactive connections."""
        current_time = datetime.now(UTC)
        stale_connections = []

        for connection_id, connection in self._connections.items():
            # Check if connection is stale
            age = (current_time - connection.created_at).total_seconds()
            if age > self.connection_timeout or not connection.is_active:
                stale_connections.append(connection_id)

        # Remove stale connections
        for connection_id in stale_connections:
            if connection_id in self._connections:
                connection = self._connections.pop(connection_id)
                connection.close()
                logger.debug(f"Cleaned up stale SSE connection {connection_id}")

    async def get_connections_info(self) -> dict[str, Any]:
        """Get information about active connections."""
        return {
            "total_connections": len(self._connections),
            "max_connections": self.max_connections,
            "is_running": self._is_running,
            "connections": [
                {
                    "id": conn.connection_id,
                    "created_at": conn.created_at.isoformat(),
                    "queue_size": conn.queue.qsize(),
                    "is_active": conn.is_active,
                    "filters": {
                        k: v
                        for k, v in {
                            "metric_types": [mt.value for mt in conn.metric_types]
                            if conn.metric_types
                            else None,
                            "user_id": conn.user_id,
                            "session_id": conn.session_id,
                            "subscription_types": list(conn.subscription_types),
                        }.items()
                        if v is not None
                    },
                }
                for conn in self._connections.values()
            ],
        }

    # BaseMetricsExporter implementation
    async def export_metrics(self, metrics_data: Any) -> None:
        """Export metrics data (not used for SSE, metrics are broadcast individually)."""
        logger.debug("SSE exporter export_metrics called (not implemented)")

    async def health_check(self) -> bool:
        """Check if the SSE exporter is healthy."""
        return self._is_running and len(self._connections) <= self.max_connections
