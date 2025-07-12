"""
Main metrics collector for the metrics domain.

This module provides the central metrics collection functionality,
coordinating between different metric types and storage backends.
"""

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from datetime import UTC, datetime, timedelta

# Type forward declaration to avoid circular imports
from typing import TYPE_CHECKING, Any, Optional
from uuid import UUID

from .calculator import CostCalculator
from .models import (
    AnyMetric,
    CostMetric,
    ErrorMetric,
    LatencyMetric,
    MetricRecord,
    MetricsSummary,
    MetricType,
    RequestMetric,
    ResponseMetric,
    UsageMetric,
)
from .storage.base import MetricsStorage


if TYPE_CHECKING:
    from .exporters.sse import SSEMetricsExporter


logger = logging.getLogger(__name__)


class MetricsCollector:
    """
    Central metrics collector that coordinates metric collection, storage, and analysis.
    """

    def __init__(
        self,
        storage: MetricsStorage,
        cost_calculator: CostCalculator | None = None,
        buffer_size: int = 1000,
        flush_interval: float = 30.0,
        enable_auto_flush: bool = True,
        sse_exporter: "SSEMetricsExporter | None" = None,
    ):
        """
        Initialize the metrics collector.

        Args:
            storage: Storage backend for metrics
            cost_calculator: Calculator for cost metrics
            buffer_size: Size of internal buffer before auto-flush
            flush_interval: Interval in seconds for auto-flush
            enable_auto_flush: Whether to enable automatic flushing
            sse_exporter: Optional SSE exporter for real-time broadcasting
        """
        self.storage = storage
        self.cost_calculator = cost_calculator or CostCalculator()
        self.buffer_size = buffer_size
        self.flush_interval = flush_interval
        self.enable_auto_flush = enable_auto_flush
        self.sse_exporter = sse_exporter

        # Internal state
        self._buffer: list[MetricRecord] = []
        self._buffer_lock = asyncio.Lock()
        self._flush_task: asyncio.Task[None] | None = None
        self._active_requests: dict[str, RequestMetric] = {}
        self._request_start_times: dict[str, datetime] = {}
        self._is_running = False

        # Metrics tracking
        self._total_metrics_collected = 0
        self._metrics_by_type: dict[MetricType, int] = dict.fromkeys(MetricType, 0)
        self._last_flush_time = datetime.now(UTC)

    async def start(self) -> None:
        """Start the metrics collector."""
        if self._is_running:
            return

        self._is_running = True

        # Initialize storage
        await self.storage.initialize()

        # Start auto-flush task if enabled
        if self.enable_auto_flush:
            self._flush_task = asyncio.create_task(self._auto_flush_loop())

        logger.info("Metrics collector started")

    async def stop(self) -> None:
        """Stop the metrics collector and flush remaining metrics."""
        if not self._is_running:
            return

        self._is_running = False

        # Cancel auto-flush task
        if self._flush_task:
            self._flush_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._flush_task

        # Flush remaining metrics
        await self.flush()

        # Close storage
        await self.storage.close()

        logger.info("Metrics collector stopped")

    async def collect_request_start(
        self,
        request_id: str,
        method: str,
        path: str,
        endpoint: str,
        api_version: str,
        user_id: str | None = None,
        session_id: str | None = None,
        **kwargs: Any,
    ) -> RequestMetric:
        """
        Collect metrics for a request start.

        Args:
            request_id: Unique request identifier
            method: HTTP method
            path: Request path
            endpoint: API endpoint
            api_version: API version
            user_id: User identifier
            session_id: Session identifier
            **kwargs: Additional request parameters

        Returns:
            RequestMetric object
        """
        request_metric = RequestMetric(
            request_id=request_id,
            user_id=user_id,
            session_id=session_id,
            method=method,
            path=path,
            endpoint=endpoint,
            api_version=api_version,
            **kwargs,
        )

        # Store for correlation with response
        self._active_requests[request_id] = request_metric
        self._request_start_times[request_id] = datetime.now(UTC)

        await self._add_to_buffer(request_metric)
        return request_metric

    async def collect_response(
        self,
        request_id: str,
        status_code: int,
        content_length: int | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        cache_read_tokens: int | None = None,
        cache_write_tokens: int | None = None,
        **kwargs: Any,
    ) -> ResponseMetric:
        """
        Collect metrics for a response.

        Args:
            request_id: Unique request identifier
            status_code: HTTP status code
            content_length: Response content length
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            cache_read_tokens: Number of cache read tokens
            cache_write_tokens: Number of cache write tokens
            **kwargs: Additional response parameters

        Returns:
            ResponseMetric object
        """
        # Calculate response time
        response_time_ms = 0.0
        if request_id in self._request_start_times:
            start_time = self._request_start_times[request_id]
            response_time_ms = (datetime.now(UTC) - start_time).total_seconds() * 1000

        response_metric = ResponseMetric(
            request_id=request_id,
            status_code=status_code,
            response_time_ms=response_time_ms,
            content_length=content_length,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read_tokens,
            cache_write_tokens=cache_write_tokens,
            **kwargs,
        )

        # Get user and session info from request
        if request_id in self._active_requests:
            request_metric = self._active_requests[request_id]
            response_metric.user_id = request_metric.user_id
            response_metric.session_id = request_metric.session_id

        await self._add_to_buffer(response_metric)

        # Generate cost metric if we have token information and model
        if (
            input_tokens is not None or output_tokens is not None
        ) and request_id in self._active_requests:
            request_metric = self._active_requests[request_id]
            if request_metric.model:
                await self.collect_cost(
                    request_id=request_id,
                    model=request_metric.model,
                    input_tokens=input_tokens or 0,
                    output_tokens=output_tokens or 0,
                    cache_read_tokens=cache_read_tokens or 0,
                    cache_write_tokens=cache_write_tokens or 0,
                )

        return response_metric

    async def collect_error(
        self,
        request_id: str | None,
        error_type: str,
        error_code: str | None = None,
        error_message: str | None = None,
        stack_trace: str | None = None,
        **kwargs: Any,
    ) -> ErrorMetric:
        """
        Collect metrics for an error.

        Args:
            request_id: Associated request identifier
            error_type: Type of error
            error_code: Error code
            error_message: Error message
            stack_trace: Stack trace
            **kwargs: Additional error parameters

        Returns:
            ErrorMetric object
        """
        error_metric = ErrorMetric(
            request_id=request_id,
            error_type=error_type,
            error_code=error_code,
            error_message=error_message,
            stack_trace=stack_trace,
            **kwargs,
        )

        # Get user and session info from request
        if request_id and request_id in self._active_requests:
            request_metric = self._active_requests[request_id]
            error_metric.user_id = request_metric.user_id
            error_metric.session_id = request_metric.session_id

        await self._add_to_buffer(error_metric)
        return error_metric

    async def collect_cost(
        self,
        request_id: str,
        model: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
        sdk_total_cost: float | None = None,
        sdk_input_cost: float | None = None,
        sdk_output_cost: float | None = None,
        sdk_cache_read_cost: float | None = None,
        sdk_cache_write_cost: float | None = None,
        **kwargs: Any,
    ) -> CostMetric:
        """
        Collect cost metrics for a request.

        Args:
            request_id: Unique request identifier
            model: Model name
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            cache_read_tokens: Number of cache read tokens
            cache_write_tokens: Number of cache write tokens
            sdk_total_cost: SDK-provided total cost
            sdk_input_cost: SDK-provided input cost
            sdk_output_cost: SDK-provided output cost
            sdk_cache_read_cost: SDK-provided cache read cost
            sdk_cache_write_cost: SDK-provided cache write cost
            **kwargs: Additional cost parameters

        Returns:
            CostMetric object
        """
        cost_metric = await self.cost_calculator.calculate_cost(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read_tokens,
            cache_write_tokens=cache_write_tokens,
            sdk_total_cost=sdk_total_cost,
            sdk_input_cost=sdk_input_cost,
            sdk_output_cost=sdk_output_cost,
            sdk_cache_read_cost=sdk_cache_read_cost,
            sdk_cache_write_cost=sdk_cache_write_cost,
        )

        # Set request correlation
        cost_metric.request_id = request_id

        # Get user and session info from request
        if request_id in self._active_requests:
            request_metric = self._active_requests[request_id]
            cost_metric.user_id = request_metric.user_id
            cost_metric.session_id = request_metric.session_id

        await self._add_to_buffer(cost_metric)
        return cost_metric

    async def collect_latency(self, request_id: str, **timings: float) -> LatencyMetric:
        """
        Collect latency metrics for a request.

        Args:
            request_id: Unique request identifier
            **timings: Timing measurements in milliseconds

        Returns:
            LatencyMetric object
        """
        # Build kwargs with only valid fields for LatencyMetric
        latency_kwargs: dict[str, Any] = {
            "request_id": request_id,
            "metric_type": MetricType.LATENCY,
        }
        for key, value in timings.items():
            if key in LatencyMetric.model_fields:
                latency_kwargs[key] = value

        latency_metric = LatencyMetric(**latency_kwargs)

        # Get user and session info from request
        if request_id in self._active_requests:
            request_metric = self._active_requests[request_id]
            latency_metric.user_id = request_metric.user_id
            latency_metric.session_id = request_metric.session_id

        await self._add_to_buffer(latency_metric)
        return latency_metric

    async def collect_usage(
        self,
        window_start: datetime,
        window_end: datetime,
        aggregation_level: str = "hourly",
        **counts: int,
    ) -> UsageMetric:
        """
        Collect usage metrics for a time window.

        Args:
            window_start: Start of time window
            window_end: End of time window
            aggregation_level: Level of aggregation
            **counts: Various count metrics

        Returns:
            UsageMetric object
        """
        # Build kwargs with only valid fields for UsageMetric
        usage_kwargs: dict[str, Any] = {
            "metric_type": MetricType.USAGE,
            "window_start": window_start,
            "window_end": window_end,
            "window_duration_seconds": (window_end - window_start).total_seconds(),
            "aggregation_level": aggregation_level,
        }
        for key, value in counts.items():
            if key in UsageMetric.model_fields:
                usage_kwargs[key] = value

        usage_metric = UsageMetric(**usage_kwargs)

        await self._add_to_buffer(usage_metric)
        return usage_metric

    async def finish_request(self, request_id: str) -> None:
        """
        Mark a request as finished and clean up tracking data.

        Args:
            request_id: Unique request identifier
        """
        self._active_requests.pop(request_id, None)
        self._request_start_times.pop(request_id, None)

    async def flush(self) -> int:
        """
        Flush buffered metrics to storage.

        Returns:
            Number of metrics flushed
        """
        async with self._buffer_lock:
            if not self._buffer:
                return 0

            metrics_to_flush = self._buffer.copy()
            self._buffer.clear()

        try:
            await self.storage.store_metrics(metrics_to_flush)
            self._last_flush_time = datetime.now(UTC)

            logger.debug(f"Flushed {len(metrics_to_flush)} metrics to storage")
            return len(metrics_to_flush)

        except Exception as e:
            logger.error(f"Failed to flush metrics: {e}")
            # Put metrics back in buffer
            async with self._buffer_lock:
                self._buffer.extend(metrics_to_flush)
            raise

    async def get_summary(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
    ) -> MetricsSummary:
        """
        Get a summary of metrics for a time period.

        Args:
            start_time: Start time for summary
            end_time: End time for summary
            user_id: Filter by user ID
            session_id: Filter by session ID

        Returns:
            MetricsSummary object
        """
        # Default to last 24 hours if no time range specified
        if end_time is None:
            end_time = datetime.now(UTC)
        if start_time is None:
            start_time = end_time - timedelta(hours=24)

        # Get metrics from storage
        metrics = await self.storage.get_metrics(
            start_time=start_time,
            end_time=end_time,
            user_id=user_id,
            session_id=session_id,
        )

        # Calculate summary
        return self._calculate_summary(metrics, start_time, end_time)

    def get_stats(self) -> dict[str, Any]:
        """
        Get collector statistics.

        Returns:
            Dictionary with collector statistics
        """
        return {
            "total_metrics_collected": self._total_metrics_collected,
            "metrics_by_type": dict(self._metrics_by_type),
            "buffer_size": len(self._buffer),
            "active_requests": len(self._active_requests),
            "last_flush_time": self._last_flush_time.isoformat(),
            "is_running": self._is_running,
        }

    @asynccontextmanager
    async def request_context(
        self, request_id: str, **kwargs: Any
    ) -> AsyncIterator[RequestMetric]:
        """
        Context manager for tracking a complete request lifecycle.

        Args:
            request_id: Unique request identifier
            **kwargs: Request parameters

        Yields:
            RequestMetric object
        """
        request_metric = await self.collect_request_start(
            request_id=request_id, **kwargs
        )

        try:
            yield request_metric
        finally:
            await self.finish_request(request_id)

    async def _add_to_buffer(self, metric: MetricRecord) -> None:
        """Add a metric to the internal buffer and broadcast via SSE if enabled."""
        async with self._buffer_lock:
            self._buffer.append(metric)
            self._total_metrics_collected += 1
            self._metrics_by_type[metric.metric_type] += 1

            # Broadcast to SSE connections if exporter is available
            if self.sse_exporter:
                try:
                    broadcast_count = await self.sse_exporter.broadcast_metric(metric)
                    if broadcast_count > 0:
                        logger.debug(
                            f"Broadcasted {metric.metric_type.value} metric to {broadcast_count} SSE connections"
                        )
                except Exception as e:
                    logger.warning(f"Failed to broadcast metric via SSE: {e}")

            # Auto-flush if buffer is full
            if len(self._buffer) >= self.buffer_size:
                asyncio.create_task(self.flush())

    async def _auto_flush_loop(self) -> None:
        """Background task for automatic flushing."""
        while self._is_running:
            try:
                await asyncio.sleep(self.flush_interval)
                await self.flush()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in auto-flush loop: {e}")

    def _calculate_summary(
        self, metrics: list[MetricRecord], start_time: datetime, end_time: datetime
    ) -> MetricsSummary:
        """Calculate summary statistics from metrics."""
        summary = MetricsSummary(start_time=start_time, end_time=end_time)

        # Group metrics by type
        requests = [m for m in metrics if isinstance(m, RequestMetric)]
        responses = [m for m in metrics if isinstance(m, ResponseMetric)]
        errors = [m for m in metrics if isinstance(m, ErrorMetric)]
        costs = [m for m in metrics if isinstance(m, CostMetric)]

        # Request metrics
        summary.total_requests = len(requests)
        summary.successful_requests = len(
            [r for r in responses if 200 <= r.status_code < 300]
        )
        summary.failed_requests = len([r for r in responses if r.status_code >= 400])

        if summary.total_requests > 0:
            summary.error_rate = summary.failed_requests / summary.total_requests

        # Response metrics
        if responses:
            response_times = [r.response_time_ms for r in responses]
            summary.avg_response_time_ms = sum(response_times) / len(response_times)

            # Calculate percentiles
            sorted_times = sorted(response_times)
            summary.p95_response_time_ms = sorted_times[int(len(sorted_times) * 0.95)]
            summary.p99_response_time_ms = sorted_times[int(len(sorted_times) * 0.99)]

        # Token metrics
        summary.total_input_tokens = sum(r.input_tokens or 0 for r in responses)
        summary.total_output_tokens = sum(r.output_tokens or 0 for r in responses)
        summary.total_tokens = summary.total_input_tokens + summary.total_output_tokens

        # Cost metrics
        summary.total_cost = sum(c.total_cost for c in costs)
        if summary.total_requests > 0:
            summary.avg_cost_per_request = summary.total_cost / summary.total_requests

        # Usage patterns
        unique_users = {r.user_id for r in requests if r.user_id}
        summary.unique_users = len(unique_users)

        # Model distribution
        model_usage: dict[str, int] = {}
        for request in requests:
            if request.model:
                model_usage[request.model] = model_usage.get(request.model, 0) + 1
        summary.model_usage = model_usage

        # Error breakdown
        error_types: dict[str, int] = {}
        for error in errors:
            error_types[error.error_type] = error_types.get(error.error_type, 0) + 1
        summary.error_types = error_types

        return summary
