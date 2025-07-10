"""
Prometheus exporter for metrics.

This module provides a Prometheus-compatible exporter that can expose
metrics data in Prometheus format for monitoring and alerting.
"""

import logging
from collections.abc import Generator
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional


try:
    from prometheus_client import (
        CONTENT_TYPE_LATEST,
        CollectorRegistry,
        Counter,
        Gauge,
        Histogram,
        Info,
        generate_latest,
    )
    from prometheus_client.core import MetricWrapperBase

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

from ..models import MetricRecord, MetricsSummary, MetricType
from ..storage.base import MetricsStorage


logger = logging.getLogger(__name__)


class PrometheusExporter:
    """
    Prometheus exporter for metrics data.

    This exporter exposes metrics in Prometheus format and can be scraped
    by Prometheus monitoring systems.

    Requires prometheus_client to be installed: pip install prometheus-client
    """

    def __init__(
        self,
        storage: MetricsStorage,
        registry: CollectorRegistry | None = None,
        namespace: str = "ccproxy",
        include_labels: list[str] | None = None,
    ):
        """
        Initialize the Prometheus exporter.

        Args:
            storage: Metrics storage backend
            registry: Prometheus registry (uses default if None)
            namespace: Metric name namespace
            include_labels: List of labels to include in metrics
        """
        if not PROMETHEUS_AVAILABLE:
            raise ImportError(
                "prometheus_client is required for Prometheus exporter. "
                "Install it with: pip install prometheus-client"
            )

        self.storage = storage
        self.registry = registry or CollectorRegistry()
        self.namespace = namespace
        self.include_labels = include_labels or [
            "user_id",
            "model",
            "provider",
            "endpoint",
        ]

        # Initialize Prometheus metrics
        self._init_metrics()

    def _init_metrics(self) -> None:
        """Initialize Prometheus metric objects."""
        # Request metrics
        self.request_total = Counter(
            f"{self.namespace}_requests_total",
            "Total number of requests",
            ["method", "endpoint", "provider", "model"],
            registry=self.registry,
        )

        self.request_duration = Histogram(
            f"{self.namespace}_request_duration_seconds",
            "Request duration in seconds",
            ["method", "endpoint", "provider", "model"],
            buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 25.0, 50.0, 100.0],
            registry=self.registry,
        )

        # Response metrics
        self.response_total = Counter(
            f"{self.namespace}_responses_total",
            "Total number of responses",
            ["status_code", "endpoint", "provider", "model"],
            registry=self.registry,
        )

        # Token metrics
        self.tokens_processed = Counter(
            f"{self.namespace}_tokens_processed_total",
            "Total number of tokens processed",
            ["type", "model", "provider"],
            registry=self.registry,
        )

        # Cost metrics
        self.cost_total = Counter(
            f"{self.namespace}_cost_total",
            "Total cost in USD",
            ["model", "provider", "currency"],
            registry=self.registry,
        )

        # Error metrics
        self.errors_total = Counter(
            f"{self.namespace}_errors_total",
            "Total number of errors",
            ["error_type", "endpoint", "recoverable"],
            registry=self.registry,
        )

        # Gauge metrics for current state
        self.active_requests = Gauge(
            f"{self.namespace}_active_requests",
            "Number of currently active requests",
            registry=self.registry,
        )

        self.cache_hit_rate = Gauge(
            f"{self.namespace}_cache_hit_rate",
            "Cache hit rate as a percentage",
            ["model"],
            registry=self.registry,
        )

        # System info
        self.info = Info(
            f"{self.namespace}_info",
            "Information about the ccproxy instance",
            registry=self.registry,
        )

        # Set static info
        self.info.info(
            {
                "version": "1.0.0",  # Could be loaded from version file
                "storage_backend": "unknown",  # Will be updated in update_metrics
            }
        )

    async def update_metrics(
        self, time_window: timedelta = timedelta(minutes=5)
    ) -> None:
        """
        Update Prometheus metrics with recent data.

        Args:
            time_window: Time window to consider for metrics updates
        """
        try:
            end_time = datetime.utcnow()
            start_time = end_time - time_window

            # Get recent metrics
            recent_metrics = await self.storage.get_metrics(
                start_time=start_time, end_time=end_time
            )

            # Process metrics
            await self._process_metrics(recent_metrics)

            # Update summary metrics
            await self._update_summary_metrics(start_time, end_time)

            # Update system info
            storage_info = await self.storage.get_storage_info()
            self.info.info(
                {
                    "version": "1.0.0",
                    "storage_backend": storage_info.get("backend", "unknown"),
                    "total_metrics": str(storage_info.get("total_metrics", 0)),
                }
            )

        except Exception as e:
            logger.error(f"Failed to update Prometheus metrics: {e}")

    async def _process_metrics(self, metrics: list[MetricRecord]) -> None:
        """Process individual metrics and update counters."""
        for metric in metrics:
            try:
                if metric.metric_type == MetricType.REQUEST:
                    self._process_request_metric(metric)
                elif metric.metric_type == MetricType.RESPONSE:
                    self._process_response_metric(metric)
                elif metric.metric_type == MetricType.ERROR:
                    self._process_error_metric(metric)
                elif metric.metric_type == MetricType.COST:
                    self._process_cost_metric(metric)

            except Exception as e:
                logger.error(f"Failed to process metric {metric.id}: {e}")

    def _process_request_metric(self, metric: MetricRecord) -> None:
        """Process a request metric."""
        from ..models import RequestMetric

        if not isinstance(metric, RequestMetric):
            return

        labels = {
            "method": metric.method or "unknown",
            "endpoint": metric.endpoint or "unknown",
            "provider": metric.provider or "unknown",
            "model": metric.model or "unknown",
        }

        self.request_total.labels(**labels).inc()

    def _process_response_metric(self, metric: MetricRecord) -> None:
        """Process a response metric."""
        from ..models import ResponseMetric

        if not isinstance(metric, ResponseMetric):
            return

        labels = {
            "status_code": str(metric.status_code),
            "endpoint": "unknown",  # Need to correlate with request
            "provider": "unknown",  # Need to correlate with request
            "model": "unknown",  # Need to correlate with request
        }

        self.response_total.labels(**labels).inc()

        # Record duration (convert ms to seconds)
        duration_seconds = metric.response_time_ms / 1000.0
        duration_labels = {
            "method": "unknown",  # Need to correlate with request
            "endpoint": "unknown",  # Need to correlate with request
            "provider": "unknown",  # Need to correlate with request
            "model": "unknown",  # Need to correlate with request
        }

        self.request_duration.labels(**duration_labels).observe(duration_seconds)

        # Record token usage
        if metric.input_tokens:
            self.tokens_processed.labels(
                type="input", model="unknown", provider="unknown"
            ).inc(metric.input_tokens)

        if metric.output_tokens:
            self.tokens_processed.labels(
                type="output", model="unknown", provider="unknown"
            ).inc(metric.output_tokens)

        if metric.cache_read_tokens:
            self.tokens_processed.labels(
                type="cache_read", model="unknown", provider="unknown"
            ).inc(metric.cache_read_tokens)

        if metric.cache_write_tokens:
            self.tokens_processed.labels(
                type="cache_write", model="unknown", provider="unknown"
            ).inc(metric.cache_write_tokens)

    def _process_error_metric(self, metric: MetricRecord) -> None:
        """Process an error metric."""
        from ..models import ErrorMetric

        if not isinstance(metric, ErrorMetric):
            return

        labels = {
            "error_type": metric.error_type,
            "endpoint": metric.endpoint or "unknown",
            "recoverable": str(metric.recoverable).lower(),
        }

        self.errors_total.labels(**labels).inc()

    def _process_cost_metric(self, metric: MetricRecord) -> None:
        """Process a cost metric."""
        from ..models import CostMetric

        if not isinstance(metric, CostMetric):
            return

        labels = {
            "model": metric.model or "unknown",
            "provider": "anthropic",  # Assumption
            "currency": metric.currency,
        }

        self.cost_total.labels(**labels).inc(metric.total_cost)

    async def _update_summary_metrics(
        self, start_time: datetime, end_time: datetime
    ) -> None:
        """Update summary gauge metrics."""
        try:
            # Get summary for the time window
            summary = await self.storage.get_metrics_summary(
                start_time=start_time, end_time=end_time
            )

            # Update cache hit rate (calculate from token data)
            await self._update_cache_metrics(start_time, end_time)

        except Exception as e:
            logger.error(f"Failed to update summary metrics: {e}")

    async def _update_cache_metrics(
        self, start_time: datetime, end_time: datetime
    ) -> None:
        """Update cache-related metrics."""
        try:
            # Get response metrics for cache calculations
            response_metrics = await self.storage.get_metrics(
                start_time=start_time,
                end_time=end_time,
                metric_type=MetricType.RESPONSE,
            )

            # Calculate cache hit rates by model
            model_cache_stats = {}

            for metric in response_metrics:
                from ..models import ResponseMetric

                if not isinstance(metric, ResponseMetric):
                    continue

                # Get model info (would need request correlation in practice)
                model = "unknown"

                if model not in model_cache_stats:
                    model_cache_stats[model] = {
                        "total_tokens": 0,
                        "cache_read_tokens": 0,
                    }

                stats = model_cache_stats[model]
                input_tokens = metric.input_tokens or 0
                cache_read_tokens = metric.cache_read_tokens or 0

                stats["total_tokens"] += input_tokens
                stats["cache_read_tokens"] += cache_read_tokens

            # Update cache hit rate gauges
            for model, stats in model_cache_stats.items():
                if stats["total_tokens"] > 0:
                    cache_hit_rate = (
                        stats["cache_read_tokens"] / stats["total_tokens"]
                    ) * 100
                    self.cache_hit_rate.labels(model=model).set(cache_hit_rate)

        except Exception as e:
            logger.error(f"Failed to update cache metrics: {e}")

    def generate_metrics(self) -> bytes:
        """
        Generate Prometheus metrics in text format.

        Returns:
            Metrics data in Prometheus text format
        """
        return generate_latest(self.registry)

    def get_content_type(self) -> str:
        """
        Get the content type for Prometheus metrics.

        Returns:
            Content type string
        """
        return CONTENT_TYPE_LATEST

    async def collect_and_generate(self) -> bytes:
        """
        Collect current metrics and generate Prometheus output.

        Returns:
            Metrics data in Prometheus text format
        """
        await self.update_metrics()
        return self.generate_metrics()

    def reset_metrics(self) -> None:
        """Reset all metrics to zero/empty state."""
        try:
            # Clear the registry and reinitialize
            self.registry._collector_to_names.clear()
            self.registry._names_to_collectors.clear()
            self._init_metrics()

        except Exception as e:
            logger.error(f"Failed to reset metrics: {e}")

    async def get_health_metrics(self) -> dict[str, Any]:
        """
        Get health metrics for the exporter.

        Returns:
            Dictionary with health information
        """
        try:
            storage_health = await self.storage.health_check()

            return {
                "status": "healthy"
                if storage_health.get("status") == "healthy"
                else "unhealthy",
                "storage_status": storage_health.get("status", "unknown"),
                "registry_collectors": len(self.registry._names_to_collectors),
                "last_update": datetime.utcnow().isoformat(),
            }

        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "last_update": datetime.utcnow().isoformat(),
            }


class PrometheusCollector:
    """
    Custom Prometheus collector for advanced metric collection.

    This collector can be registered with Prometheus and will collect
    metrics on-demand when Prometheus scrapes the endpoint.
    """

    def __init__(self, storage: MetricsStorage, namespace: str = "ccproxy"):
        """
        Initialize the collector.

        Args:
            storage: Metrics storage backend
            namespace: Metric name namespace
        """
        self.storage = storage
        self.namespace = namespace

    def describe(self) -> Generator[Any, None, None]:
        """Describe the metrics this collector provides."""
        return []  # Return empty for now, could be enhanced

    def collect(self) -> Generator[Any, None, None]:
        """Collect metrics on-demand."""
        try:
            # This would be called by Prometheus during scraping
            # Implementation would need to be synchronous for prometheus_client
            # In practice, you'd want to cache recent metrics or use a different approach
            pass
        except Exception as e:
            logger.error(f"Failed to collect metrics: {e}")

        return []


# Utility functions for FastAPI integration


async def create_prometheus_exporter(
    storage: MetricsStorage, namespace: str = "ccproxy"
) -> PrometheusExporter:
    """
    Create and initialize a Prometheus exporter.

    Args:
        storage: Metrics storage backend
        namespace: Metric name namespace

    Returns:
        Configured PrometheusExporter instance
    """
    exporter = PrometheusExporter(storage=storage, namespace=namespace)
    await exporter.update_metrics()
    return exporter


def create_metrics_handler(exporter: PrometheusExporter):
    """
    Create a FastAPI handler function for the /metrics endpoint.

    Args:
        exporter: Configured PrometheusExporter instance

    Returns:
        FastAPI handler function
    """

    async def metrics_handler():
        """Handle Prometheus metrics scraping."""
        try:
            metrics_data = await exporter.collect_and_generate()

            # Return Response for FastAPI
            from fastapi import Response

            return Response(
                content=metrics_data, media_type=exporter.get_content_type()
            )

        except Exception as e:
            logger.error(f"Failed to generate metrics: {e}")
            from fastapi import HTTPException

            raise HTTPException(status_code=500, detail="Failed to generate metrics")

    return metrics_handler
