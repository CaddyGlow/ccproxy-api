"""Metrics service for collecting and processing metrics.

This module provides a metrics service that handles the business logic
for collecting, aggregating, and processing metrics from various sources.
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Optional, Union

from ccproxy.core.interfaces import MetricExporter


logger = logging.getLogger(__name__)


class MetricsService:
    """Service for metrics collection and processing.

    This service implements the business logic for collecting metrics
    from various sources, aggregating them, and managing their export
    to external systems.
    """

    def __init__(
        self,
        exporters: list[MetricExporter] | None = None,
        buffer_size: int = 1000,
        export_interval: int = 60,
    ):
        """Initialize the metrics service.

        Args:
            exporters: List of metric exporters to use
            buffer_size: Maximum number of metrics to buffer
            export_interval: Interval in seconds between exports
        """
        self.exporters = exporters or []
        self.buffer_size = buffer_size
        self.export_interval = export_interval

        # Internal metrics storage
        self._metrics_buffer: list[dict[str, Any]] = []
        self._aggregated_metrics: dict[str, Any] = {}
        self._last_export_time = datetime.now()

        # Metrics categories
        self._request_metrics: dict[str, Any] = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "average_response_time": 0.0,
            "requests_by_model": {},
            "requests_by_path": {},
        }

        self._service_metrics: dict[str, Any] = {
            "sdk_requests": 0,
            "proxy_requests": 0,
            "streaming_requests": 0,
            "tool_requests": 0,
            "uptime": 0,
        }

        self._error_metrics: dict[str, Any] = {
            "total_errors": 0,
            "errors_by_type": {},
            "errors_by_status_code": {},
        }

        logger.info("Metrics service initialized")

    def record_request_start(
        self,
        request_id: str,
        method: str,
        path: str,
        model: str | None = None,
        user_id: str | None = None,
    ) -> None:
        """Record the start of a request.

        Args:
            request_id: Unique identifier for the request
            method: HTTP method
            path: Request path
            model: AI model used (if applicable)
            user_id: User identifier (if available)
        """
        metric = {
            "type": "request_start",
            "request_id": request_id,
            "method": method,
            "path": path,
            "model": model,
            "user_id": user_id,
            "timestamp": datetime.now().isoformat(),
        }

        self._add_metric(metric)
        self._update_request_metrics(metric)

    def record_request_end(
        self,
        request_id: str,
        status_code: int,
        response_time: float,
        tokens_used: int | None = None,
        service_type: str | None = None,
    ) -> None:
        """Record the end of a request.

        Args:
            request_id: Unique identifier for the request
            status_code: HTTP status code
            response_time: Response time in seconds
            tokens_used: Number of tokens used
            service_type: Type of service used (sdk, proxy)
        """
        metric = {
            "type": "request_end",
            "request_id": request_id,
            "status_code": status_code,
            "response_time": response_time,
            "tokens_used": tokens_used,
            "service_type": service_type,
            "timestamp": datetime.now().isoformat(),
        }

        self._add_metric(metric)
        self._update_request_metrics(metric)

    def record_error(
        self,
        error_type: str,
        error_message: str,
        status_code: int | None = None,
        request_id: str | None = None,
    ) -> None:
        """Record an error event.

        Args:
            error_type: Type of error
            error_message: Error message
            status_code: HTTP status code (if applicable)
            request_id: Associated request ID (if applicable)
        """
        metric = {
            "type": "error",
            "error_type": error_type,
            "error_message": error_message,
            "status_code": status_code,
            "request_id": request_id,
            "timestamp": datetime.now().isoformat(),
        }

        self._add_metric(metric)
        self._update_error_metrics(metric)

    def record_service_event(
        self,
        event_type: str,
        service_type: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record a service-level event.

        Args:
            event_type: Type of event (e.g., 'startup', 'shutdown', 'health_check')
            service_type: Type of service (e.g., 'sdk', 'proxy')
            metadata: Additional metadata
        """
        metric = {
            "type": "service_event",
            "event_type": event_type,
            "service_type": service_type,
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat(),
        }

        self._add_metric(metric)
        self._update_service_metrics(metric)

    def get_metrics_summary(self) -> dict[str, Any]:
        """Get a summary of current metrics.

        Returns:
            Dictionary containing aggregated metrics
        """
        return {
            "request_metrics": self._request_metrics.copy(),
            "service_metrics": self._service_metrics.copy(),
            "error_metrics": self._error_metrics.copy(),
            "buffer_size": len(self._metrics_buffer),
            "last_export": self._last_export_time.isoformat(),
        }

    def get_metrics_by_timeframe(
        self,
        start_time: datetime,
        end_time: datetime,
    ) -> list[dict[str, Any]]:
        """Get metrics within a specific timeframe.

        Args:
            start_time: Start of the timeframe
            end_time: End of the timeframe

        Returns:
            List of metrics within the timeframe
        """
        filtered_metrics = []

        for metric in self._metrics_buffer:
            metric_time = datetime.fromisoformat(metric["timestamp"])
            if start_time <= metric_time <= end_time:
                filtered_metrics.append(metric)

        return filtered_metrics

    def calculate_request_rate(self, window_minutes: int = 5) -> float:
        """Calculate the request rate over a time window.

        Args:
            window_minutes: Time window in minutes

        Returns:
            Requests per minute
        """
        now = datetime.now()
        start_time = now - timedelta(minutes=window_minutes)

        request_count = 0
        for metric in self._metrics_buffer:
            if metric["type"] == "request_start":
                metric_time = datetime.fromisoformat(metric["timestamp"])
                if start_time <= metric_time <= now:
                    request_count += 1

        return request_count / window_minutes if window_minutes > 0 else 0.0

    def calculate_error_rate(self, window_minutes: int = 5) -> float:
        """Calculate the error rate over a time window.

        Args:
            window_minutes: Time window in minutes

        Returns:
            Error rate as a percentage
        """
        now = datetime.now()
        start_time = now - timedelta(minutes=window_minutes)

        total_requests = 0
        error_requests = 0

        for metric in self._metrics_buffer:
            if metric["type"] == "request_end":
                metric_time = datetime.fromisoformat(metric["timestamp"])
                if start_time <= metric_time <= now:
                    total_requests += 1
                    if metric.get("status_code", 200) >= 400:
                        error_requests += 1

        return (error_requests / total_requests * 100) if total_requests > 0 else 0.0

    async def export_metrics(self) -> bool:
        """Export metrics to configured exporters.

        Returns:
            True if export was successful for at least one exporter
        """
        if not self.exporters:
            logger.debug("No exporters configured")
            return True

        metrics_to_export = self.get_metrics_summary()
        success_count = 0

        for exporter in self.exporters:
            try:
                if await exporter.export_metrics(metrics_to_export):
                    success_count += 1
                    logger.debug(
                        f"Successfully exported metrics to {type(exporter).__name__}"
                    )
                else:
                    logger.warning(
                        f"Failed to export metrics to {type(exporter).__name__}"
                    )
            except Exception as e:
                logger.error(
                    f"Error exporting metrics to {type(exporter).__name__}: {e}"
                )

        if success_count > 0:
            self._last_export_time = datetime.now()
            return True

        return False

    def should_export(self) -> bool:
        """Check if metrics should be exported based on time interval.

        Returns:
            True if export should be performed
        """
        time_since_export = datetime.now() - self._last_export_time
        return time_since_export.total_seconds() >= self.export_interval

    def clear_buffer(self) -> None:
        """Clear the metrics buffer."""
        self._metrics_buffer.clear()
        logger.debug("Metrics buffer cleared")

    def _add_metric(self, metric: dict[str, Any]) -> None:
        """Add a metric to the buffer.

        Args:
            metric: Metric data to add
        """
        self._metrics_buffer.append(metric)

        # Maintain buffer size limit
        if len(self._metrics_buffer) > self.buffer_size:
            self._metrics_buffer.pop(0)

    def _update_request_metrics(self, metric: dict[str, Any]) -> None:
        """Update request-related metrics.

        Args:
            metric: Metric data to process
        """
        if metric["type"] == "request_start":
            self._request_metrics["total_requests"] += 1

            # Track by path
            path = metric.get("path", "unknown")
            if path not in self._request_metrics["requests_by_path"]:
                self._request_metrics["requests_by_path"][path] = 0
            self._request_metrics["requests_by_path"][path] += 1

            # Track by model
            model = metric.get("model")
            if model:
                if model not in self._request_metrics["requests_by_model"]:
                    self._request_metrics["requests_by_model"][model] = 0
                self._request_metrics["requests_by_model"][model] += 1

        elif metric["type"] == "request_end":
            status_code = metric.get("status_code", 200)

            if status_code < 400:
                self._request_metrics["successful_requests"] += 1
            else:
                self._request_metrics["failed_requests"] += 1

            # Update average response time
            response_time = metric.get("response_time", 0.0)
            current_avg = self._request_metrics["average_response_time"]
            total_requests = self._request_metrics["total_requests"]

            if total_requests > 0:
                self._request_metrics["average_response_time"] = (
                    current_avg * (total_requests - 1) + response_time
                ) / total_requests

    def _update_service_metrics(self, metric: dict[str, Any]) -> None:
        """Update service-related metrics.

        Args:
            metric: Metric data to process
        """
        if metric["type"] == "service_event":
            service_type = metric.get("service_type", "unknown")

            if service_type == "sdk":
                self._service_metrics["sdk_requests"] += 1
            elif service_type == "proxy":
                self._service_metrics["proxy_requests"] += 1

            # Track streaming and tool requests
            metadata = metric.get("metadata", {})
            if metadata.get("streaming", False):
                self._service_metrics["streaming_requests"] += 1
            if metadata.get("tools", False):
                self._service_metrics["tool_requests"] += 1

    def _update_error_metrics(self, metric: dict[str, Any]) -> None:
        """Update error-related metrics.

        Args:
            metric: Metric data to process
        """
        if metric["type"] == "error":
            self._error_metrics["total_errors"] += 1

            # Track by error type
            error_type = metric.get("error_type", "unknown")
            if error_type not in self._error_metrics["errors_by_type"]:
                self._error_metrics["errors_by_type"][error_type] = 0
            self._error_metrics["errors_by_type"][error_type] += 1

            # Track by status code
            status_code = metric.get("status_code")
            if status_code:
                if status_code not in self._error_metrics["errors_by_status_code"]:
                    self._error_metrics["errors_by_status_code"][status_code] = 0
                self._error_metrics["errors_by_status_code"][status_code] += 1
