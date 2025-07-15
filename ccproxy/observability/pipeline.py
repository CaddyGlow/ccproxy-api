"""
Log-to-storage pipeline processor for structured events.

This module provides a pipeline system that converts structured log events
from structlog into metrics for storage in the existing metrics storage backends.
This enables historical analysis while using lightweight structured logging.

Key features:
- Async pipeline processing with batch operations
- Conversion of log events to storage-compatible metrics
- Thread-safe event queuing with backpressure
- Graceful degradation when storage unavailable
- Configurable batch sizes and processing intervals
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections.abc import Callable, Mapping, MutableMapping
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field
from typing import Any, Optional, Union

import structlog


logger = structlog.get_logger(__name__)


@dataclass
class LogEvent:
    """Structured log event for pipeline processing."""

    event_type: str
    timestamp: float
    request_id: str | None = None
    duration_ms: float | None = None
    data: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_structlog_event(cls, event_dict: dict[str, Any]) -> LogEvent:
        """Create LogEvent from structlog event dictionary."""
        # Handle timestamp conversion from ISO string to Unix timestamp
        timestamp_raw = event_dict.get("timestamp", time.time())
        if isinstance(timestamp_raw, str):
            # Convert ISO timestamp string to Unix timestamp
            try:
                from datetime import datetime

                dt = datetime.fromisoformat(timestamp_raw.replace("Z", "+00:00"))
                timestamp = dt.timestamp()
            except (ValueError, AttributeError):
                # Fallback to current time if parsing fails
                timestamp = time.time()
        else:
            # Already a numeric timestamp
            timestamp = (
                float(timestamp_raw) if timestamp_raw is not None else time.time()
            )

        return cls(
            event_type=event_dict.get("event", "unknown"),
            timestamp=timestamp,
            request_id=event_dict.get("request_id"),
            duration_ms=event_dict.get("duration_ms"),
            data={
                k: v
                for k, v in event_dict.items()
                if k not in ("event", "timestamp", "request_id", "duration_ms")
            },
        )


@dataclass
class PipelineConfig:
    """Configuration for the log-to-storage pipeline."""

    enabled: bool = True
    batch_size: int = 100
    processing_interval: float = 5.0  # seconds
    max_queue_size: int = 10000
    enable_metrics_conversion: bool = True
    storage_backends: list[str] = field(default_factory=lambda: ["duckdb"])
    duckdb_enabled: bool = True
    duckdb_path: str = "data/metrics.duckdb"


class LogToStoragePipeline:
    """
    Pipeline processor that converts structured log events to storage metrics.

    This processor bridges the gap between lightweight structured logging
    and historical metrics storage by converting relevant log events into
    metrics that can be stored in the existing storage backends.
    """

    def __init__(self, config: PipelineConfig | None = None):
        self.config = config or PipelineConfig()
        self._event_queue: asyncio.Queue[LogEvent] = asyncio.Queue(
            maxsize=self.config.max_queue_size
        )
        self._running = False
        self._processor_task: asyncio.Task[None] | None = None
        self._storage_backends: list[Any] = []
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        """Start the pipeline processor."""
        if self._running:
            return

        self._running = True
        logger.debug("pipeline_start")

        # Initialize storage backends
        await self._init_storage_backends()

        # Start processor task
        self._processor_task = asyncio.create_task(self._processing_loop())

    async def stop(self) -> None:
        """Stop the pipeline processor."""
        if not self._running:
            return

        self._running = False
        logger.debug("pipeline_stop")

        # Cancel processor task
        if self._processor_task:
            self._processor_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._processor_task

        # Flush remaining events
        await self._flush_remaining_events()

    async def enqueue_event(self, event_dict: dict[str, Any]) -> bool:
        """
        Enqueue a structured log event for processing.

        Args:
            event_dict: Structured log event from structlog

        Returns:
            True if event was queued, False if queue full
        """
        if not self.config.enabled or not self._running:
            return False

        try:
            log_event = LogEvent.from_structlog_event(event_dict)
            await self._event_queue.put(log_event)
            return True
        except asyncio.QueueFull:
            logger.warning("pipeline_queue_full", queue_size=self._event_queue.qsize())
            return False
        except Exception as e:
            logger.error("pipeline_enqueue_error", error=str(e), event=event_dict)
            return False

    async def _init_storage_backends(self) -> None:
        """Initialize storage backends for metrics."""
        if not self.config.enable_metrics_conversion:
            return

        # Import storage backends dynamically to avoid circular imports
        for backend_name in self.config.storage_backends:
            try:
                if backend_name == "duckdb" and self.config.duckdb_enabled:
                    from .storage.duckdb import DuckDBStorage

                    storage = DuckDBStorage(database_path=self.config.duckdb_path)
                    await storage.initialize()
                    self._storage_backends.append(storage)
                    logger.info("pipeline_storage_init", backend=backend_name)
                elif backend_name == "duckdb" and not self.config.duckdb_enabled:
                    pass  # DuckDB disabled
                elif backend_name == "sqlite":
                    # Legacy SQLite support (removed - use DuckDB instead)
                    logger.warning("sqlite_backend_removed", use_instead="duckdb")
                else:
                    logger.warning("unknown_storage_backend", backend=backend_name)

            except ImportError as e:
                logger.warning(
                    "pipeline_storage_import_error", backend=backend_name, error=str(e)
                )

    async def _processing_loop(self) -> None:
        """Main processing loop for converting events to metrics."""
        while self._running:
            try:
                # Collect batch of events
                events = await self._collect_event_batch()

                if events:
                    # Convert to metrics and store
                    await self._process_event_batch(events)

                # Wait for next processing interval
                await asyncio.sleep(self.config.processing_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("pipeline_processing_error", error=str(e))
                await asyncio.sleep(1.0)  # Back off on error

    async def _collect_event_batch(self) -> list[LogEvent]:
        """Collect a batch of events from the queue."""
        events: list[LogEvent] = []
        batch_size = self.config.batch_size

        try:
            # Get first event (blocking)
            try:
                event = await asyncio.wait_for(
                    self._event_queue.get(), timeout=self.config.processing_interval
                )
                events.append(event)
            except TimeoutError:
                return events  # Empty batch

            # Get additional events (non-blocking)
            while len(events) < batch_size:
                try:
                    event = self._event_queue.get_nowait()
                    events.append(event)
                except asyncio.QueueEmpty:
                    break

        except Exception as e:
            logger.error("pipeline_batch_collection_error", error=str(e))

        return events

    async def _process_event_batch(self, events: list[LogEvent]) -> None:
        """Process a batch of events and convert to metrics."""
        if not self.config.enable_metrics_conversion or not self._storage_backends:
            return

        metrics = []

        for event in events:
            try:
                # Convert request events to metrics
                if event.event_type in (
                    "request_start",
                    "request_success",
                    "request_error",
                ):
                    metric = await self._convert_request_event(event)
                    if metric:
                        metrics.append(metric)

                # Convert operation events to metrics
                elif event.event_type in (
                    "operation_start",
                    "operation_success",
                    "operation_error",
                ):
                    metric = await self._convert_operation_event(event)
                    if metric:
                        metrics.append(metric)

            except Exception as e:
                logger.error(
                    "pipeline_event_conversion_error",
                    event=event.__dict__,
                    error=str(e),
                )

        # Store metrics in backends
        if metrics:
            await self._store_metrics(metrics)

    async def _convert_request_event(self, event: LogEvent) -> dict[str, Any] | None:
        """Convert request event to storage metric."""
        if event.event_type == "request_success" and event.duration_ms is not None:
            return {
                "timestamp": event.timestamp,
                "request_id": event.request_id or str(uuid.uuid4()),
                "method": event.data.get("method", "unknown"),
                "endpoint": event.data.get(
                    "endpoint", event.data.get("path", "unknown")
                ),
                "service_type": event.data.get("service_type", "unknown"),
                "model": event.data.get("model"),
                "status": "success",
                "response_time": event.duration_ms / 1000.0,  # Convert to seconds
                "tokens_input": event.data.get("tokens_input", 0),
                "tokens_output": event.data.get("tokens_output", 0),
                "cost_usd": event.data.get("cost_usd", 0.0),
                "metadata": {
                    k: v
                    for k, v in event.data.items()
                    if k
                    not in (
                        "method",
                        "endpoint",
                        "path",
                        "service_type",
                        "model",
                        "tokens_input",
                        "tokens_output",
                        "cost_usd",
                    )
                },
            }
        elif event.event_type == "request_error":
            return {
                "timestamp": event.timestamp,
                "request_id": event.request_id or str(uuid.uuid4()),
                "method": event.data.get("method", "unknown"),
                "endpoint": event.data.get(
                    "endpoint", event.data.get("path", "unknown")
                ),
                "service_type": event.data.get("service_type", "unknown"),
                "model": event.data.get("model"),
                "status": "error",
                "response_time": (event.duration_ms or 0) / 1000.0,
                "error_type": event.data.get("error_type", "unknown"),
                "error_message": event.data.get("error_message", ""),
                "metadata": {
                    k: v
                    for k, v in event.data.items()
                    if k
                    not in (
                        "method",
                        "endpoint",
                        "path",
                        "service_type",
                        "model",
                        "error_type",
                        "error_message",
                    )
                },
            }
        return None

    async def _convert_operation_event(self, event: LogEvent) -> dict[str, Any] | None:
        """Convert operation event to storage metric."""
        if event.event_type in ("operation_success", "operation_error"):
            return {
                "timestamp": event.timestamp,
                "request_id": event.request_id,
                "operation_id": event.data.get("operation_id"),
                "operation_name": event.data.get("operation_name", "unknown"),
                "duration_ms": event.duration_ms,
                "status": "success"
                if event.event_type == "operation_success"
                else "error",
                "error_type": event.data.get("error_type")
                if event.event_type == "operation_error"
                else None,
                "metadata": {
                    k: v
                    for k, v in event.data.items()
                    if k not in ("operation_id", "operation_name", "error_type")
                },
            }
        return None

    async def _store_metrics(self, metrics: list[dict[str, Any]]) -> None:
        """Store metrics in configured storage backends."""
        for backend in self._storage_backends:
            try:
                # Store metrics using backend's store method
                if hasattr(backend, "store_batch"):
                    await backend.store_batch(metrics)
                elif hasattr(backend, "store"):
                    for metric in metrics:
                        await backend.store(metric)

            except Exception as e:
                logger.error(
                    "pipeline_storage_error",
                    backend=type(backend).__name__,
                    error=str(e),
                )

    async def _flush_remaining_events(self) -> None:
        """Flush any remaining events in the queue."""
        remaining_events = []

        try:
            while True:
                event = self._event_queue.get_nowait()
                remaining_events.append(event)
        except asyncio.QueueEmpty:
            pass

        if remaining_events:
            await self._process_event_batch(remaining_events)
            logger.info("pipeline_flush_complete", event_count=len(remaining_events))


# Global pipeline instance
_global_pipeline: LogToStoragePipeline | None = None
_pipeline_lock = asyncio.Lock()


async def get_pipeline() -> LogToStoragePipeline:
    """Get or create global pipeline instance."""
    global _global_pipeline

    async with _pipeline_lock:
        if _global_pipeline is None:
            # Import here to avoid circular imports
            from ccproxy.config.settings import get_settings

            settings = get_settings()
            config = PipelineConfig(
                enabled=True,
                duckdb_enabled=settings.observability.duckdb_enabled,
                duckdb_path=settings.observability.duckdb_path,
            )
            _global_pipeline = LogToStoragePipeline(config)
            await _global_pipeline.start()
        return _global_pipeline


async def enqueue_log_event(event_dict: dict[str, Any]) -> bool:
    """
    Enqueue a structured log event for pipeline processing.

    This function should be called from structlog processors to
    feed events into the metrics storage pipeline.

    Args:
        event_dict: Structured log event from structlog

    Returns:
        True if event was queued, False if pipeline unavailable
    """
    try:
        pipeline = await get_pipeline()
        return await pipeline.enqueue_event(event_dict)
    except Exception as e:
        # Don't log errors here to avoid infinite recursion
        return False


def create_structlog_processor() -> Callable[
    [Any, str, MutableMapping[str, Any]], MutableMapping[str, Any]
]:
    """
    Create a structlog processor that feeds events to the pipeline.

    This processor should be added to structlog's processor chain
    to automatically send relevant events to the metrics storage pipeline.

    Returns:
        Structlog processor function
    """

    def pipeline_processor(
        logger: Any, method_name: str, event_dict: MutableMapping[str, Any]
    ) -> MutableMapping[str, Any]:
        """Structlog processor that sends events to pipeline."""

        # Only process specific event types to avoid noise
        event_type = event_dict.get("event", "")
        if event_type in (
            "request_start",
            "request_success",
            "request_error",
            "operation_start",
            "operation_success",
            "operation_error",
        ):
            # Send to pipeline asynchronously (fire and forget)
            with suppress(RuntimeError):
                # No event loop running, skip pipeline
                asyncio.create_task(enqueue_log_event(dict(event_dict)))

        return event_dict

    return pipeline_processor


@asynccontextmanager
async def pipeline_context(config: PipelineConfig | None = None) -> Any:
    """
    Context manager for pipeline lifecycle management.

    Args:
        config: Pipeline configuration

    Yields:
        LogToStoragePipeline instance
    """
    pipeline = LogToStoragePipeline(config)
    await pipeline.start()

    try:
        yield pipeline
    finally:
        await pipeline.stop()
