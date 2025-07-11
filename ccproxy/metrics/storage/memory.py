"""
In-memory storage implementation for metrics.

This module provides a simple in-memory storage backend for metrics,
useful for development, testing, or lightweight deployments.
"""

import asyncio
import logging
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Optional
from uuid import UUID

from ..models import (
    CostMetric,
    ErrorMetric,
    LatencyMetric,
    MetricRecord,
    MetricsSummary,
    MetricType,
    RequestMetric,
    ResponseMetric,
)
from .base import MetricsStorage, StorageError


logger = logging.getLogger(__name__)


class InMemoryMetricsStorage(MetricsStorage):
    """
    In-memory storage implementation for metrics.

    This storage backend keeps all metrics in memory and provides
    fast access but does not persist data across restarts.
    """

    def __init__(self, max_metrics: int = 100000, auto_cleanup: bool = True):
        """
        Initialize the in-memory storage.

        Args:
            max_metrics: Maximum number of metrics to keep in memory
            auto_cleanup: Whether to automatically clean up old metrics
        """
        self.max_metrics = max_metrics
        self.auto_cleanup = auto_cleanup

        # Storage
        self._metrics: list[MetricRecord] = []
        self._metrics_by_id: dict[UUID, MetricRecord] = {}
        self._metrics_by_type: dict[MetricType, list[MetricRecord]] = defaultdict(list)
        self._metrics_by_user: dict[str, list[MetricRecord]] = defaultdict(list)
        self._metrics_by_session: dict[str, list[MetricRecord]] = defaultdict(list)
        self._metrics_by_request: dict[str, list[MetricRecord]] = defaultdict(list)

        # Synchronization
        self._lock = asyncio.Lock()

        # Statistics
        self._total_stored = 0
        self._total_cleaned = 0
        self._last_cleanup = datetime.utcnow()

    async def initialize(self) -> None:
        """Initialize the in-memory storage."""
        logger.info("Initialized in-memory metrics storage")

    async def close(self) -> None:
        """Close the in-memory storage."""
        async with self._lock:
            self._metrics.clear()
            self._metrics_by_id.clear()
            self._metrics_by_type.clear()
            self._metrics_by_user.clear()
            self._metrics_by_session.clear()
            self._metrics_by_request.clear()

        logger.info("Closed in-memory metrics storage")

    async def store_metric(self, metric: MetricRecord) -> bool:
        """Store a single metric record."""
        try:
            async with self._lock:
                # Add to main storage
                self._metrics.append(metric)
                self._metrics_by_id[metric.id] = metric

                # Add to indexes
                self._metrics_by_type[metric.metric_type].append(metric)

                if metric.user_id:
                    self._metrics_by_user[metric.user_id].append(metric)

                if metric.session_id:
                    self._metrics_by_session[metric.session_id].append(metric)

                if metric.request_id:
                    self._metrics_by_request[metric.request_id].append(metric)

                self._total_stored += 1

                # Auto-cleanup if needed
                if self.auto_cleanup and len(self._metrics) > self.max_metrics:
                    await self._cleanup_old_metrics()

            return True

        except Exception as e:
            logger.error(f"Failed to store metric: {e}")
            return False

    async def store_metrics(self, metrics: list[MetricRecord]) -> int:
        """Store multiple metric records."""
        stored_count = 0

        try:
            async with self._lock:
                for metric in metrics:
                    # Add to main storage
                    self._metrics.append(metric)
                    self._metrics_by_id[metric.id] = metric

                    # Add to indexes
                    self._metrics_by_type[metric.metric_type].append(metric)

                    if metric.user_id:
                        self._metrics_by_user[metric.user_id].append(metric)

                    if metric.session_id:
                        self._metrics_by_session[metric.session_id].append(metric)

                    if metric.request_id:
                        self._metrics_by_request[metric.request_id].append(metric)

                    stored_count += 1

                self._total_stored += stored_count

                # Auto-cleanup if needed
                if self.auto_cleanup and len(self._metrics) > self.max_metrics:
                    await self._cleanup_old_metrics()

            return stored_count

        except Exception as e:
            logger.error(f"Failed to store metrics: {e}")
            return stored_count

    async def get_metric(self, metric_id: UUID) -> MetricRecord | None:
        """Retrieve a single metric record by ID."""
        async with self._lock:
            return self._metrics_by_id.get(metric_id)

    async def get_metrics(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        metric_type: MetricType | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        request_id: str | None = None,
        limit: int | None = None,
        offset: int | None = None,
        order_by: str | None = None,
        order_desc: bool = True,
        filters: dict[str, Any] | None = None,
    ) -> list[MetricRecord]:
        """Retrieve multiple metric records with filtering."""
        async with self._lock:
            # Start with appropriate index
            if request_id:
                candidates = self._metrics_by_request.get(request_id, [])
            elif user_id:
                candidates = self._metrics_by_user.get(user_id, [])
            elif session_id:
                candidates = self._metrics_by_session.get(session_id, [])
            elif metric_type:
                candidates = self._metrics_by_type.get(metric_type, [])
            else:
                candidates = self._metrics

            # Apply filters
            filtered_metrics = []
            for metric in candidates:
                if not self._matches_filters(
                    metric,
                    start_time,
                    end_time,
                    metric_type,
                    user_id,
                    session_id,
                    request_id,
                    filters,
                ):
                    continue
                filtered_metrics.append(metric)

            # Sort
            if order_by:
                reverse = order_desc
                try:
                    filtered_metrics.sort(
                        key=lambda m: getattr(m, order_by, None) or "", reverse=reverse
                    )
                except (AttributeError, TypeError):
                    # Fall back to timestamp sorting
                    filtered_metrics.sort(key=lambda m: m.timestamp, reverse=reverse)
            else:
                filtered_metrics.sort(key=lambda m: m.timestamp, reverse=order_desc)

            # Apply pagination
            if offset:
                filtered_metrics = filtered_metrics[offset:]
            if limit:
                filtered_metrics = filtered_metrics[:limit]

            return filtered_metrics

    async def count_metrics(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        metric_type: MetricType | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        request_id: str | None = None,
        filters: dict[str, Any] | None = None,
    ) -> int:
        """Count metric records matching the criteria."""
        async with self._lock:
            # Start with appropriate index
            if request_id:
                candidates = self._metrics_by_request.get(request_id, [])
            elif user_id:
                candidates = self._metrics_by_user.get(user_id, [])
            elif session_id:
                candidates = self._metrics_by_session.get(session_id, [])
            elif metric_type:
                candidates = self._metrics_by_type.get(metric_type, [])
            else:
                candidates = self._metrics

            # Count matching metrics
            count = 0
            for metric in candidates:
                if self._matches_filters(
                    metric,
                    start_time,
                    end_time,
                    metric_type,
                    user_id,
                    session_id,
                    request_id,
                    filters,
                ):
                    count += 1

            return count

    async def delete_metrics(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        metric_type: MetricType | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        request_id: str | None = None,
        filters: dict[str, Any] | None = None,
    ) -> int:
        """Delete metric records matching the criteria."""
        async with self._lock:
            # Find metrics to delete
            to_delete = []
            for metric in self._metrics:
                if self._matches_filters(
                    metric,
                    start_time,
                    end_time,
                    metric_type,
                    user_id,
                    session_id,
                    request_id,
                    filters,
                ):
                    to_delete.append(metric)

            # Remove from all indexes
            for metric in to_delete:
                # Remove from main storage
                if metric in self._metrics:
                    self._metrics.remove(metric)

                # Remove from ID index
                self._metrics_by_id.pop(metric.id, None)

                # Remove from type index
                if metric.metric_type in self._metrics_by_type:
                    type_list = self._metrics_by_type[metric.metric_type]
                    if metric in type_list:
                        type_list.remove(metric)

                # Remove from user index
                if metric.user_id and metric.user_id in self._metrics_by_user:
                    user_list = self._metrics_by_user[metric.user_id]
                    if metric in user_list:
                        user_list.remove(metric)

                # Remove from session index
                if metric.session_id and metric.session_id in self._metrics_by_session:
                    session_list = self._metrics_by_session[metric.session_id]
                    if metric in session_list:
                        session_list.remove(metric)

                # Remove from request index
                if metric.request_id and metric.request_id in self._metrics_by_request:
                    request_list = self._metrics_by_request[metric.request_id]
                    if metric in request_list:
                        request_list.remove(metric)

            deleted_count = len(to_delete)
            self._total_cleaned += deleted_count

            return deleted_count

    async def get_metrics_summary(
        self,
        start_time: datetime,
        end_time: datetime,
        user_id: str | None = None,
        session_id: str | None = None,
        group_by: str | None = None,
    ) -> MetricsSummary:
        """Get aggregated metrics summary."""
        metrics = await self.get_metrics(
            start_time=start_time,
            end_time=end_time,
            user_id=user_id,
            session_id=session_id,
        )

        # Calculate summary (same logic as in collector)
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
            if sorted_times:
                summary.p95_response_time_ms = sorted_times[
                    int(len(sorted_times) * 0.95)
                ]
                summary.p99_response_time_ms = sorted_times[
                    int(len(sorted_times) * 0.99)
                ]

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
        model_usage = {}
        for request in requests:
            if request.model:
                model_usage[request.model] = model_usage.get(request.model, 0) + 1
        summary.model_usage = model_usage

        # Error breakdown
        error_types = {}
        for error in errors:
            error_types[error.error_type] = error_types.get(error.error_type, 0) + 1
        summary.error_types = error_types

        return summary

    async def get_time_series(
        self,
        start_time: datetime,
        end_time: datetime,
        interval: str = "1h",
        metric_type: MetricType | None = None,
        aggregation: str = "count",
        user_id: str | None = None,
        session_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get time series data for metrics."""
        metrics = await self.get_metrics(
            start_time=start_time,
            end_time=end_time,
            metric_type=metric_type,
            user_id=user_id,
            session_id=session_id,
        )

        # Parse interval
        interval_delta = self._parse_interval(interval)

        # Create time buckets
        time_buckets = []
        current_time = start_time
        while current_time < end_time:
            time_buckets.append(current_time)
            current_time += interval_delta

        # Group metrics by time bucket
        time_series = []
        for bucket_start in time_buckets:
            bucket_end = bucket_start + interval_delta

            bucket_metrics = [
                m for m in metrics if bucket_start <= m.timestamp < bucket_end
            ]

            # Calculate aggregation
            if aggregation == "count":
                value = len(bucket_metrics)
            elif aggregation == "sum" and bucket_metrics:
                # Sum numeric fields (could be enhanced)
                value = len(bucket_metrics)  # Default to count
            elif aggregation == "avg" and bucket_metrics:
                value = len(bucket_metrics)  # Default to count
            else:
                value = len(bucket_metrics)

            time_series.append(
                {
                    "timestamp": bucket_start.isoformat(),
                    "value": value,
                    "count": len(bucket_metrics),
                }
            )

        return time_series

    async def health_check(self) -> dict[str, Any]:
        """Perform health check."""
        async with self._lock:
            return {
                "status": "healthy",
                "total_metrics": len(self._metrics),
                "memory_usage": f"{len(self._metrics)} metrics",
                "total_stored": self._total_stored,
                "total_cleaned": self._total_cleaned,
                "last_cleanup": self._last_cleanup.isoformat(),
            }

    async def get_storage_info(self) -> dict[str, Any]:
        """Get storage information."""
        async with self._lock:
            return {
                "backend": "in-memory",
                "total_metrics": len(self._metrics),
                "max_metrics": self.max_metrics,
                "auto_cleanup": self.auto_cleanup,
                "metrics_by_type": {
                    str(metric_type): len(metrics)
                    for metric_type, metrics in self._metrics_by_type.items()
                },
                "unique_users": len(self._metrics_by_user),
                "unique_sessions": len(self._metrics_by_session),
                "unique_requests": len(self._metrics_by_request),
                "total_stored": self._total_stored,
                "total_cleaned": self._total_cleaned,
            }

    def _matches_filters(
        self,
        metric: MetricRecord,
        start_time: datetime | None,
        end_time: datetime | None,
        metric_type: MetricType | None,
        user_id: str | None,
        session_id: str | None,
        request_id: str | None,
        filters: dict[str, Any] | None,
    ) -> bool:
        """Check if a metric matches the given filters."""
        # Time filters
        if start_time and metric.timestamp < start_time:
            return False
        if end_time and metric.timestamp >= end_time:
            return False

        # Type filter
        if metric_type and metric.metric_type != metric_type:
            return False

        # ID filters
        if user_id and metric.user_id != user_id:
            return False
        if session_id and metric.session_id != session_id:
            return False
        if request_id and metric.request_id != request_id:
            return False

        # Additional filters
        if filters:
            for key, value in filters.items():
                if not hasattr(metric, key):
                    return False
                if getattr(metric, key) != value:
                    return False

        return True

    async def _cleanup_old_metrics(self) -> None:
        """Clean up old metrics to stay within memory limits."""
        # Sort by timestamp and keep only the most recent metrics
        self._metrics.sort(key=lambda m: m.timestamp, reverse=True)

        # Calculate how many to remove
        to_remove_count = len(self._metrics) - self.max_metrics
        if to_remove_count <= 0:
            return

        # Get metrics to remove (oldest ones)
        metrics_to_remove = self._metrics[-to_remove_count:]

        # Remove from main storage
        self._metrics = self._metrics[:-to_remove_count]

        # Remove from indexes
        for metric in metrics_to_remove:
            # Remove from ID index
            self._metrics_by_id.pop(metric.id, None)

            # Remove from type index
            if metric.metric_type in self._metrics_by_type:
                type_list = self._metrics_by_type[metric.metric_type]
                if metric in type_list:
                    type_list.remove(metric)

            # Remove from user index
            if metric.user_id and metric.user_id in self._metrics_by_user:
                user_list = self._metrics_by_user[metric.user_id]
                if metric in user_list:
                    user_list.remove(metric)

            # Remove from session index
            if metric.session_id and metric.session_id in self._metrics_by_session:
                session_list = self._metrics_by_session[metric.session_id]
                if metric in session_list:
                    session_list.remove(metric)

            # Remove from request index
            if metric.request_id and metric.request_id in self._metrics_by_request:
                request_list = self._metrics_by_request[metric.request_id]
                if metric in request_list:
                    request_list.remove(metric)

        self._total_cleaned += to_remove_count
        self._last_cleanup = datetime.utcnow()

        logger.debug(f"Cleaned up {to_remove_count} old metrics")

    def _parse_interval(self, interval: str) -> timedelta:
        """Parse interval string to timedelta."""
        # Simple parser for common intervals
        if interval.endswith("s"):
            return timedelta(seconds=int(interval[:-1]))
        elif interval.endswith("m"):
            return timedelta(minutes=int(interval[:-1]))
        elif interval.endswith("h"):
            return timedelta(hours=int(interval[:-1]))
        elif interval.endswith("d"):
            return timedelta(days=int(interval[:-1]))
        else:
            # Default to 1 hour
            return timedelta(hours=1)
