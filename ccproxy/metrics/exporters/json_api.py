"""
JSON API exporter for metrics.

This module provides a JSON-based API exporter that exposes metrics
data through REST endpoints for monitoring dashboards and analytics.
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Optional, cast

from ..models import MetricRecord, MetricsSummary, MetricType
from ..storage.base import MetricsStorage


logger = logging.getLogger(__name__)


class JsonApiExporter:
    """
    JSON API exporter for metrics data.

    This exporter provides REST API endpoints that return metrics
    data in JSON format for consumption by dashboards and analytics tools.
    """

    def __init__(
        self,
        storage: MetricsStorage,
        default_limit: int = 1000,
        max_limit: int = 10000,
        cache_ttl: int = 60,
    ):
        """
        Initialize the JSON API exporter.

        Args:
            storage: Metrics storage backend
            default_limit: Default number of records to return
            max_limit: Maximum number of records to return
            cache_ttl: Cache TTL in seconds for expensive queries
        """
        self.storage = storage
        self.default_limit = default_limit
        self.max_limit = max_limit
        self.cache_ttl = cache_ttl

        # Simple cache for expensive queries
        self._cache: dict[str, dict[str, Any]] = {}

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
    ) -> dict[str, Any]:
        """
        Get metrics with filtering and pagination.

        Args:
            start_time: Filter metrics after this time
            end_time: Filter metrics before this time
            metric_type: Filter by metric type
            user_id: Filter by user ID
            session_id: Filter by session ID
            request_id: Filter by request ID
            limit: Maximum number of records to return
            offset: Number of records to skip
            order_by: Field to order by
            order_desc: Whether to order in descending order
            filters: Additional filters

        Returns:
            Dictionary with metrics data and metadata
        """
        try:
            # Validate and set defaults
            if limit is None:
                limit = self.default_limit
            elif limit > self.max_limit:
                limit = self.max_limit

            # Default time range to last 24 hours if not specified
            if end_time is None:
                end_time = datetime.utcnow()
            if start_time is None:
                start_time = end_time - timedelta(hours=24)

            # Get metrics from storage
            metrics = await self.storage.get_metrics(
                start_time=start_time,
                end_time=end_time,
                metric_type=metric_type,
                user_id=user_id,
                session_id=session_id,
                request_id=request_id,
                limit=limit,
                offset=offset,
                order_by=order_by,
                order_desc=order_desc,
                filters=filters,
            )

            # Get total count for pagination
            total_count = await self.storage.count_metrics(
                start_time=start_time,
                end_time=end_time,
                metric_type=metric_type,
                user_id=user_id,
                session_id=session_id,
                request_id=request_id,
                filters=filters,
            )

            # Convert metrics to dictionaries
            metrics_data = [self._metric_to_dict(metric) for metric in metrics]

            return {
                "data": metrics_data,
                "metadata": {
                    "total_count": total_count,
                    "returned_count": len(metrics_data),
                    "limit": limit,
                    "offset": offset or 0,
                    "start_time": start_time.isoformat(),
                    "end_time": end_time.isoformat(),
                    "filters": {
                        k: v
                        for k, v in {
                            "metric_type": metric_type.value if metric_type else None,
                            "user_id": user_id,
                            "session_id": session_id,
                            "request_id": request_id,
                            "custom_filters": filters,
                        }.items()
                        if v is not None
                    },
                    "pagination": {
                        k: v
                        for k, v in {
                            "has_next": (offset or 0) + len(metrics_data) < total_count,
                            "has_previous": (offset or 0) > 0,
                            "next_offset": (offset or 0) + limit
                            if (offset or 0) + len(metrics_data) < total_count
                            else None,
                            "previous_offset": max(0, (offset or 0) - limit)
                            if (offset or 0) > 0
                            else None,
                        }.items()
                        if v is not None
                    },
                },
            }

        except Exception as e:
            logger.error(f"Failed to get metrics: {e}")
            return {
                "error": str(e),
                "data": [],
                "metadata": {"total_count": 0, "returned_count": 0},
            }

    async def get_summary(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        group_by: str | None = None,
    ) -> dict[str, Any]:
        """
        Get aggregated metrics summary.

        Args:
            start_time: Start of time period
            end_time: End of time period
            user_id: Filter by user ID
            session_id: Filter by session ID
            group_by: Group results by field

        Returns:
            Dictionary with summary data
        """
        try:
            # Default time range to last 24 hours if not specified
            if end_time is None:
                end_time = datetime.utcnow()
            if start_time is None:
                start_time = end_time - timedelta(hours=24)

            # Check cache first
            cache_key = f"summary_{start_time.isoformat()}_{end_time.isoformat()}_{user_id}_{session_id}_{group_by}"
            if cache_key in self._cache:
                cached_data = self._cache[cache_key]
                if (
                    datetime.utcnow() - cached_data["timestamp"]
                ).seconds < self.cache_ttl:
                    return cast(dict[str, Any], cached_data["data"])

            # Get summary from storage
            summary = await self.storage.get_metrics_summary(
                start_time=start_time,
                end_time=end_time,
                user_id=user_id,
                session_id=session_id,
                group_by=group_by,
            )

            # Convert to dictionary
            summary_data = {
                "time_period": {
                    "start_time": summary.start_time.isoformat(),
                    "end_time": summary.end_time.isoformat(),
                    "duration_hours": (
                        summary.end_time - summary.start_time
                    ).total_seconds()
                    / 3600,
                },
                "requests": {
                    "total": summary.total_requests,
                    "successful": summary.successful_requests,
                    "failed": summary.failed_requests,
                    "error_rate": summary.error_rate,
                    "success_rate": 1.0 - summary.error_rate
                    if summary.total_requests > 0
                    else 0.0,
                },
                "performance": {
                    "avg_response_time_ms": summary.avg_response_time_ms,
                    "p95_response_time_ms": summary.p95_response_time_ms,
                    "p99_response_time_ms": summary.p99_response_time_ms,
                },
                "tokens": {
                    "total_input": summary.total_input_tokens,
                    "total_output": summary.total_output_tokens,
                    "total": summary.total_tokens,
                    "avg_input_per_request": (
                        summary.total_input_tokens / summary.total_requests
                        if summary.total_requests > 0
                        else 0.0
                    ),
                    "avg_output_per_request": (
                        summary.total_output_tokens / summary.total_requests
                        if summary.total_requests > 0
                        else 0.0
                    ),
                },
                "costs": {
                    "total": summary.total_cost,
                    "avg_per_request": summary.avg_cost_per_request,
                    "currency": "USD",
                },
                "usage": {
                    "unique_users": summary.unique_users,
                    "peak_requests_per_minute": summary.peak_requests_per_minute,
                    "requests_per_hour": (
                        summary.total_requests
                        / (
                            (summary.end_time - summary.start_time).total_seconds()
                            / 3600
                        )
                        if (summary.end_time - summary.start_time).total_seconds() > 0
                        else 0.0
                    ),
                },
                "models": summary.model_usage,
                "errors": summary.error_types,
            }

            # Cache the result
            self._cache[cache_key] = {
                "data": summary_data,
                "timestamp": datetime.utcnow(),
            }

            return summary_data

        except Exception as e:
            logger.error(f"Failed to get summary: {e}")
            return {"error": str(e)}

    async def get_time_series(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        interval: str = "1h",
        metric_type: MetricType | None = None,
        aggregation: str = "count",
        user_id: str | None = None,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Get time series data for metrics.

        Args:
            start_time: Start of time period
            end_time: End of time period
            interval: Time interval for grouping
            metric_type: Filter by metric type
            aggregation: Aggregation function
            user_id: Filter by user ID
            session_id: Filter by session ID

        Returns:
            Dictionary with time series data
        """
        try:
            # Default time range to last 24 hours if not specified
            if end_time is None:
                end_time = datetime.utcnow()
            if start_time is None:
                start_time = end_time - timedelta(hours=24)

            # Get time series from storage
            time_series = await self.storage.get_time_series(
                start_time=start_time,
                end_time=end_time,
                interval=interval,
                metric_type=metric_type,
                aggregation=aggregation,
                user_id=user_id,
                session_id=session_id,
            )

            # Calculate additional statistics
            values = [point.get("value", 0) for point in time_series]

            stats = {}
            if values:
                stats = {
                    "min": min(values),
                    "max": max(values),
                    "avg": sum(values) / len(values),
                    "total": sum(values),
                }

            return {
                "data": time_series,
                "metadata": {
                    "start_time": start_time.isoformat(),
                    "end_time": end_time.isoformat(),
                    "interval": interval,
                    "metric_type": metric_type.value if metric_type else None,
                    "aggregation": aggregation,
                    "point_count": len(time_series),
                    "statistics": stats,
                    "filters": {"user_id": user_id, "session_id": session_id},
                },
            }

        except Exception as e:
            logger.error(f"Failed to get time series: {e}")
            return {"error": str(e), "data": []}

    async def get_health(self) -> dict[str, Any]:
        """
        Get health status of the metrics system.

        Returns:
            Dictionary with health information
        """
        try:
            # Get storage health
            storage_health = await self.storage.health_check()

            # Get storage info
            storage_info = await self.storage.get_storage_info()

            return {
                "status": "healthy"
                if storage_health.get("status") == "healthy"
                else "unhealthy",
                "timestamp": datetime.utcnow().isoformat(),
                "storage": storage_health,
                "info": storage_info,
                "cache": {
                    "cached_queries": len(self._cache),
                    "cache_ttl": self.cache_ttl,
                },
                "limits": {
                    "default_limit": self.default_limit,
                    "max_limit": self.max_limit,
                },
            }

        except Exception as e:
            logger.error(f"Failed to get health: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat(),
            }

    async def get_stats(self) -> dict[str, Any]:
        """
        Get statistics about the metrics system.

        Returns:
            Dictionary with statistics
        """
        try:
            # Get basic counts by type
            type_counts = {}
            for metric_type in MetricType:
                count = await self.storage.count_metrics(metric_type=metric_type)
                type_counts[metric_type.value] = count

            # Get recent activity (last hour)
            recent_start = datetime.utcnow() - timedelta(hours=1)
            recent_count = await self.storage.count_metrics(start_time=recent_start)

            # Get storage info
            storage_info = await self.storage.get_storage_info()

            return {
                "timestamp": datetime.utcnow().isoformat(),
                "totals": type_counts,
                "recent_activity": {
                    "last_hour_count": recent_count,
                    "rate_per_minute": recent_count / 60.0,
                },
                "storage": storage_info,
            }

        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {"error": str(e)}

    async def export_data(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        format_type: str = "json",
        compress: bool = False,
    ) -> dict[str, Any]:
        """
        Export metrics data for backup or analysis.

        Args:
            start_time: Start of time period
            end_time: End of time period
            format_type: Export format ('json', 'csv')
            compress: Whether to compress the output

        Returns:
            Dictionary with export data or download info
        """
        try:
            # Default time range to last 7 days if not specified
            if end_time is None:
                end_time = datetime.utcnow()
            if start_time is None:
                start_time = end_time - timedelta(days=7)

            # Get all metrics in the time range
            metrics = await self.storage.get_metrics(
                start_time=start_time,
                end_time=end_time,
                limit=None,  # Get all records
            )

            if format_type == "json":
                export_data = {
                    "export_info": {
                        "timestamp": datetime.utcnow().isoformat(),
                        "start_time": start_time.isoformat(),
                        "end_time": end_time.isoformat(),
                        "record_count": len(metrics),
                        "format": "json",
                    },
                    "metrics": [self._metric_to_dict(metric) for metric in metrics],
                }

                return export_data

            elif format_type == "csv":
                # This would need CSV conversion logic
                return {
                    "error": "CSV export not implemented yet",
                    "supported_formats": ["json"],
                }

            else:
                return {
                    "error": f"Unsupported format: {format_type}",
                    "supported_formats": ["json"],
                }

        except Exception as e:
            logger.error(f"Failed to export data: {e}")
            return {"error": str(e)}

    def _metric_to_dict(self, metric: MetricRecord) -> dict[str, Any]:
        """
        Convert a metric record to a dictionary using Pydantic serialization.

        Args:
            metric: MetricRecord to convert

        Returns:
            Dictionary representation of the metric (None values excluded)
        """
        return metric.model_dump(mode="json", exclude_none=True)

    def clear_cache(self) -> None:
        """Clear the internal cache."""
        self._cache.clear()

    def get_cache_info(self) -> dict[str, Any]:
        """Get information about the cache."""
        return {
            "cached_queries": len(self._cache),
            "cache_ttl": self.cache_ttl,
            "cache_keys": list(self._cache.keys()),
        }
