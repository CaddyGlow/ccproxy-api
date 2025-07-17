"""
Prometheus metrics for operational monitoring.

This module provides direct prometheus_client integration for fast operational metrics
like request counts, response times, and resource usage. These metrics are optimized
for real-time monitoring and alerting.

Key features:
- Thread-safe metric operations using prometheus_client
- Minimal overhead for high-frequency operations
- Standard Prometheus metric types (Counter, Histogram, Gauge)
- Automatic label management and validation
- Pushgateway integration for batch metric pushing
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional, Union


try:
    from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, Info

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

    # Create dummy classes for graceful degradation
    class _DummyCounter:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def labels(self, **kwargs: Any) -> _DummyCounter:
            return self

        def inc(self, value: float = 1) -> None:
            pass

    class _DummyHistogram:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def labels(self, **kwargs: Any) -> _DummyHistogram:
            return self

        def observe(self, value: float) -> None:
            pass

        def time(self) -> _DummyHistogram:
            return self

    class _DummyGauge:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def labels(self, **kwargs: Any) -> _DummyGauge:
            return self

        def set(self, value: float) -> None:
            pass

        def inc(self, value: float = 1) -> None:
            pass

        def dec(self, value: float = 1) -> None:
            pass

    class _DummyInfo:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def info(self, labels: dict[str, str]) -> None:
            pass

    class _DummyCollectorRegistry:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

    # Assign dummy classes to the expected names
    Counter = _DummyCounter  # type: ignore[misc,assignment]
    Histogram = _DummyHistogram  # type: ignore[misc,assignment]
    Gauge = _DummyGauge  # type: ignore[misc,assignment]
    Info = _DummyInfo  # type: ignore[misc,assignment]
    CollectorRegistry = _DummyCollectorRegistry  # type: ignore[misc,assignment]


from structlog import get_logger


logger = get_logger(__name__)


class PrometheusMetrics:
    """
    Prometheus metrics collector for operational monitoring.

    Provides thread-safe, high-performance metrics collection using prometheus_client.
    Designed for minimal overhead in request processing hot paths.
    """

    def __init__(
        self,
        namespace: str = "ccproxy",
        registry: CollectorRegistry | None = None,
        pushgateway_client: Any | None = None,
    ):
        """
        Initialize Prometheus metrics.

        Args:
            namespace: Metric name prefix
            registry: Custom Prometheus registry (uses default if None)
            pushgateway_client: Optional pushgateway client for dependency injection
        """
        if not PROMETHEUS_AVAILABLE:
            logger.warning(
                "prometheus_client not available. Metrics will be disabled. "
                "Install with: pip install prometheus-client"
            )

        self.namespace = namespace
        # Use default registry if None is passed
        if registry is None and PROMETHEUS_AVAILABLE:
            from prometheus_client import REGISTRY

            self.registry: CollectorRegistry | None = REGISTRY
        else:
            self.registry = registry
        self._enabled = PROMETHEUS_AVAILABLE
        self._pushgateway_client = pushgateway_client

        if self._enabled:
            self._init_metrics()
            # Initialize pushgateway client if not provided via DI
            if self._pushgateway_client is None:
                self._init_pushgateway()

    def _init_metrics(self) -> None:
        """Initialize all Prometheus metric objects."""
        # Request metrics
        self.request_counter = Counter(
            f"{self.namespace}_requests_total",
            "Total number of requests processed",
            labelnames=["method", "endpoint", "model", "status", "service_type"],
            registry=self.registry,
        )

        self.response_time = Histogram(
            f"{self.namespace}_response_duration_seconds",
            "Response time in seconds",
            labelnames=["model", "endpoint", "service_type"],
            buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 25.0],
            registry=self.registry,
        )

        # Token metrics
        self.token_counter = Counter(
            f"{self.namespace}_tokens_total",
            "Total tokens processed",
            labelnames=[
                "type",
                "model",
                "service_type",
            ],  # _type: input, output, cache_read, cache_write
            registry=self.registry,
        )

        # Cost metrics
        self.cost_counter = Counter(
            f"{self.namespace}_cost_usd_total",
            "Total cost in USD",
            labelnames=[
                "model",
                "cost_type",
                "service_type",
            ],  # cost_type: input, output, cache, total
            registry=self.registry,
        )

        # Error metrics
        self.error_counter = Counter(
            f"{self.namespace}_errors_total",
            "Total number of errors",
            labelnames=["error_type", "endpoint", "model", "service_type"],
            registry=self.registry,
        )

        # Active requests gauge
        self.active_requests = Gauge(
            f"{self.namespace}_active_requests",
            "Number of currently active requests",
            registry=self.registry,
        )

        # System info
        self.system_info = Info(
            f"{self.namespace}_info", "System information", registry=self.registry
        )

        # Service up metric (for Grafana service health)
        self.up = Gauge(
            "up",
            "Service is up and running",
            labelnames=["job"],
            registry=self.registry,
        )

        # Set initial system info
        try:
            from ccproxy import __version__

            version = __version__
        except ImportError:
            version = "unknown"

        self.system_info.info(
            {
                "version": version,
                "metrics_enabled": "true",
            }
        )

        # Set service as up
        self.up.labels(job="ccproxy").set(1)

    def _init_pushgateway(self) -> None:
        """Initialize Pushgateway client if configured (fallback for non-DI usage)."""
        try:
            # Import here to avoid circular imports
            from ccproxy.config.settings import get_settings

            from .pushgateway import PushgatewayClient

            settings = get_settings()

            self._pushgateway_client = PushgatewayClient(settings.observability)

            if self._pushgateway_client.is_enabled():
                logger.info(
                    "pushgateway_initialized: url=%s job=%s",
                    settings.observability.pushgateway_url,
                    settings.observability.pushgateway_job,
                )
        except Exception as e:
            logger.warning("pushgateway_init_failed: error=%s", str(e))
            self._pushgateway_client = None

    def record_request(
        self,
        method: str,
        endpoint: str,
        model: str | None = None,
        status: str | int = "unknown",
        service_type: str | None = None,
    ) -> None:
        """
        Record a request event.

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint path
            model: Model name used
            status: Response status code or status string
            service_type: Service type (claude_sdk_service, proxy_service)
        """
        if not self._enabled:
            return

        self.request_counter.labels(
            method=method,
            endpoint=endpoint,
            model=model or "unknown",
            status=str(status),
            service_type=service_type or "unknown",
        ).inc()

    def record_response_time(
        self,
        duration_seconds: float,
        model: str | None = None,
        endpoint: str = "unknown",
        service_type: str | None = None,
    ) -> None:
        """
        Record response time.

        Args:
            duration_seconds: Response time in seconds
            model: Model name used
            endpoint: API endpoint
            service_type: Service type (claude_sdk_service, proxy_service)
        """
        if not self._enabled:
            return

        self.response_time.labels(
            model=model or "unknown",
            endpoint=endpoint,
            service_type=service_type or "unknown",
        ).observe(duration_seconds)

    def record_tokens(
        self,
        token_count: int,
        token_type: str,
        model: str | None = None,
        service_type: str | None = None,
    ) -> None:
        """
        Record token usage.

        Args:
            token_count: Number of tokens
            token_type: Type of tokens (input, output, cache_read, cache_write)
            model: Model name
            service_type: Service type (claude_sdk_service, proxy_service)
        """
        if not self._enabled or token_count <= 0:
            return

        self.token_counter.labels(
            type=token_type,
            model=model or "unknown",
            service_type=service_type or "unknown",
        ).inc(token_count)

    def record_cost(
        self,
        cost_usd: float,
        model: str | None = None,
        cost_type: str = "total",
        service_type: str | None = None,
    ) -> None:
        """
        Record cost.

        Args:
            cost_usd: Cost in USD
            model: Model name
            cost_type: Type of cost (input, output, cache, total)
            service_type: Service type (claude_sdk_service, proxy_service)
        """
        if not self._enabled or cost_usd <= 0:
            return

        self.cost_counter.labels(
            model=model or "unknown",
            cost_type=cost_type,
            service_type=service_type or "unknown",
        ).inc(cost_usd)

    def record_error(
        self,
        error_type: str,
        endpoint: str = "unknown",
        model: str | None = None,
        service_type: str | None = None,
    ) -> None:
        """
        Record an error event.

        Args:
            error_type: Type/name of error
            endpoint: API endpoint where error occurred
            model: Model name if applicable
            service_type: Service type (claude_sdk_service, proxy_service)
        """
        if not self._enabled:
            return

        self.error_counter.labels(
            error_type=error_type,
            endpoint=endpoint,
            model=model or "unknown",
            service_type=service_type or "unknown",
        ).inc()

    def set_active_requests(self, count: int) -> None:
        """
        Set the current number of active requests.

        Args:
            count: Number of active requests
        """
        if not self._enabled:
            return

        self.active_requests.set(count)

    def inc_active_requests(self) -> None:
        """Increment active request counter."""
        if not self._enabled:
            return

        self.active_requests.inc()

    def dec_active_requests(self) -> None:
        """Decrement active request counter."""
        if not self._enabled:
            return

        self.active_requests.dec()

    def update_system_info(self, info: dict[str, str]) -> None:
        """
        Update system information.

        Args:
            info: Dictionary of system information key-value pairs
        """
        if not self._enabled:
            return

        self.system_info.info(info)

    def is_enabled(self) -> bool:
        """Check if metrics collection is enabled."""
        return self._enabled

    def push_to_gateway(self, method: str = "push") -> bool:
        """
        Push current metrics to Pushgateway using official prometheus_client methods.

        Args:
            method: Push method - "push" (replace), "pushadd" (add), or "delete"

        Returns:
            True if push succeeded, False otherwise
        """

        if not self._enabled or not self._pushgateway_client:
            return False

        result = self._pushgateway_client.push_metrics(self.registry, method)
        return bool(result)

    def push_add_to_gateway(self) -> bool:
        """
        Add current metrics to existing job/instance in Pushgateway (pushadd operation).

        This is useful when you want to add metrics without replacing existing ones.

        Returns:
            True if push succeeded, False otherwise
        """
        return self.push_to_gateway(method="pushadd")

    def delete_from_gateway(self) -> bool:
        """
        Delete all metrics for the configured job from Pushgateway.

        This removes all metrics associated with the job, useful for cleanup.

        Returns:
            True if delete succeeded, False otherwise
        """

        if not self._enabled or not self._pushgateway_client:
            return False

        result = self._pushgateway_client.delete_metrics()
        return bool(result)

    def is_pushgateway_enabled(self) -> bool:
        """Check if Pushgateway client is enabled and configured."""
        return (
            self._pushgateway_client is not None
            and self._pushgateway_client.is_enabled()
        )


# Global metrics instance
_global_metrics: PrometheusMetrics | None = None


def get_metrics(
    namespace: str = "ccproxy",
    registry: CollectorRegistry | None = None,
    pushgateway_client: Any | None = None,
    settings: Any | None = None,
) -> PrometheusMetrics:
    """
    Get or create global metrics instance with dependency injection.

    Args:
        namespace: Metric namespace prefix
        registry: Custom Prometheus registry
        pushgateway_client: Optional pushgateway client for dependency injection
        settings: Optional settings instance to avoid circular imports

    Returns:
        PrometheusMetrics instance with full pushgateway support:
        - push_to_gateway(): Replace all metrics (default)
        - push_add_to_gateway(): Add metrics to existing job
        - delete_from_gateway(): Delete all metrics for job
    """
    global _global_metrics

    if _global_metrics is None:
        # Create pushgateway client if not provided via DI
        if pushgateway_client is None:
            from .pushgateway import get_pushgateway_client

            pushgateway_client = get_pushgateway_client()

        _global_metrics = PrometheusMetrics(
            namespace=namespace,
            registry=registry,
            pushgateway_client=pushgateway_client,
        )

    return _global_metrics


def reset_metrics() -> None:
    """Reset global metrics instance (mainly for testing)."""
    global _global_metrics
    _global_metrics = None

    # Clear Prometheus registry to avoid duplicate metrics in tests
    if PROMETHEUS_AVAILABLE:
        try:
            from prometheus_client import REGISTRY

            # Clear all collectors from the registry
            collectors = list(REGISTRY._collector_to_names.keys())
            for collector in collectors:
                REGISTRY.unregister(collector)
        except Exception:
            # If clearing the registry fails, just continue
            # This is mainly for testing and shouldn't break functionality
            pass

    # Also reset pushgateway client
    from .pushgateway import reset_pushgateway_client

    reset_pushgateway_client()
