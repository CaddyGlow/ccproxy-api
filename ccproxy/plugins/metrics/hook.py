"""Hook-based metrics collection implementation."""

import time

from ccproxy.core.logging import get_plugin_logger
from ccproxy.core.plugins.hooks import Hook
from ccproxy.core.plugins.hooks.base import HookContext
from ccproxy.core.plugins.hooks.events import HookEvent

from .collector import PrometheusMetrics
from .config import MetricsConfig
from .pushgateway import PushgatewayClient


logger = get_plugin_logger(__name__)


class MetricsHook(Hook):
    """Hook-based metrics collection implementation.

    This hook listens to request/response lifecycle events and updates
    Prometheus metrics accordingly. It provides event-driven metric
    collection without requiring direct metric calls in the code.
    """

    name = "metrics"
    events = [
        HookEvent.REQUEST_STARTED,
        HookEvent.REQUEST_COMPLETED,
        HookEvent.REQUEST_FAILED,
        HookEvent.PROVIDER_REQUEST_PREPARED,
        HookEvent.PROVIDER_RESPONSE_RECEIVED,
        HookEvent.PROVIDER_ERROR,
        HookEvent.PROVIDER_STREAM_START,
        HookEvent.PROVIDER_STREAM_CHUNK,
        HookEvent.PROVIDER_STREAM_END,
    ]
    priority = 700  # HookLayer.OBSERVATION - Metrics collection first

    def __init__(self, config: MetricsConfig | None = None) -> None:
        """Initialize the metrics hook.

        Args:
            config: Metrics configuration
        """
        self.config = config or MetricsConfig()

        # Initialize collectors based on config using an isolated registry to
        # avoid global REGISTRY collisions in multi-app/test environments.
        if self.config.enabled:
            registry = None
            try:
                from prometheus_client import (
                    CollectorRegistry as CollectorRegistry,
                )

                registry = CollectorRegistry()
            except Exception:
                registry = None

            self.collector: PrometheusMetrics | None = PrometheusMetrics(
                namespace=self.config.namespace,
                histogram_buckets=self.config.histogram_buckets,
                registry=registry,
            )
        else:
            self.collector = None

        self.pushgateway: PushgatewayClient | None = (
            PushgatewayClient(self.config)
            if self.config.pushgateway_enabled and self.config.enabled
            else None
        )

        # Track active requests and their start times
        self._request_start_times: dict[str, float] = {}

        logger.debug(
            "metrics_configured",
            enabled=self.config.enabled,
            namespace=self.config.namespace,
            pushgateway_enabled=self.config.pushgateway_enabled,
            pushgateway_url=self.config.pushgateway_url,
        )

    async def __call__(self, context: HookContext) -> None:
        """Handle hook events for metrics collection.

        Args:
            context: Hook context with event data
        """
        if not self.config.enabled or not self.collector:
            return

        # Map hook events to handler methods
        handlers = {
            HookEvent.REQUEST_STARTED: self._handle_request_start,
            HookEvent.REQUEST_COMPLETED: self._handle_request_complete,
            HookEvent.REQUEST_FAILED: self._handle_request_failed,
            HookEvent.PROVIDER_REQUEST_PREPARED: self._handle_provider_request,
            HookEvent.PROVIDER_RESPONSE_RECEIVED: self._handle_provider_response,
            HookEvent.PROVIDER_ERROR: self._handle_provider_error,
            HookEvent.PROVIDER_STREAM_START: self._handle_stream_start,
            HookEvent.PROVIDER_STREAM_CHUNK: self._handle_stream_chunk,
            HookEvent.PROVIDER_STREAM_END: self._handle_stream_end,
        }

        handler = handlers.get(context.event)
        if handler:
            try:
                await handler(context)
            except Exception as e:
                logger.error(
                    "metrics_hook_error",
                    hook_event=context.event.value if context.event else "unknown",
                    error=str(e),
                    exc_info=e,
                )

    async def _handle_request_start(self, context: HookContext) -> None:
        """Handle REQUEST_STARTED event."""
        if not self.config.collect_request_metrics or not self.collector:
            return

        request_id = context.data.get("request_id", "unknown")

        # Track request start time
        self._request_start_times[request_id] = time.time()

        # Increment active requests
        self.collector.inc_active_requests()

        logger.debug(
            "metrics_request_started",
            request_id=request_id,
            active_requests=len(self._request_start_times),
        )

    async def _handle_request_complete(self, context: HookContext) -> None:
        """Handle REQUEST_COMPLETED event."""
        if not self.config.collect_request_metrics or not self.collector:
            return

        request_id = context.data.get("request_id", "unknown")
        method = context.data.get("method", "UNKNOWN")
        endpoint = context.data.get("endpoint", context.data.get("url", "/"))
        model = context.data.get("model")
        status_code = context.data.get(
            "response_status", context.data.get("status_code", 200)
        )
        service_type = context.data.get("service_type", "unknown")

        # Calculate duration if we have start time
        duration_seconds = 0.0
        if request_id in self._request_start_times:
            start_time = self._request_start_times.pop(request_id)
            duration_seconds = time.time() - start_time
        elif "duration" in context.data:
            # Use provided duration if available
            duration_seconds = context.data["duration"]

        # Record metrics
        self.collector.record_request(
            method=method,
            endpoint=endpoint,
            model=model,
            status=status_code,
            service_type=service_type,
        )

        if duration_seconds > 0:
            self.collector.record_response_time(
                duration_seconds=duration_seconds,
                model=model,
                endpoint=endpoint,
                service_type=service_type,
            )

        # Decrement active requests
        self.collector.dec_active_requests()

        # Handle token metrics if present
        if self.config.collect_token_metrics:
            usage = context.data.get("usage", {})
            if usage:
                if input_tokens := usage.get("input_tokens"):
                    self.collector.record_tokens(
                        token_count=input_tokens,
                        token_type="input",
                        model=model,
                        service_type=service_type,
                    )
                if output_tokens := usage.get("output_tokens"):
                    self.collector.record_tokens(
                        token_count=output_tokens,
                        token_type="output",
                        model=model,
                        service_type=service_type,
                    )
                if cache_read := usage.get("cache_read_input_tokens"):
                    self.collector.record_tokens(
                        token_count=cache_read,
                        token_type="cache_read",
                        model=model,
                        service_type=service_type,
                    )
                if cache_write := usage.get("cache_creation_input_tokens"):
                    self.collector.record_tokens(
                        token_count=cache_write,
                        token_type="cache_write",
                        model=model,
                        service_type=service_type,
                    )

        # Handle cost metrics if present
        if self.config.collect_cost_metrics and (cost := context.data.get("cost_usd")):
            self.collector.record_cost(
                cost_usd=cost,
                model=model,
                cost_type="total",
                service_type=service_type,
            )

        logger.debug(
            "metrics_request_completed",
            request_id=request_id,
            duration_seconds=duration_seconds,
            status_code=status_code,
            model=model,
        )

    async def _handle_request_failed(self, context: HookContext) -> None:
        """Handle REQUEST_FAILED event."""
        if not self.config.collect_error_metrics or not self.collector:
            return

        request_id = context.data.get("request_id", "unknown")
        endpoint = context.data.get("endpoint", context.data.get("url", "/"))
        model = context.data.get("model")
        service_type = context.data.get("service_type", "unknown")
        error = context.error
        error_type = type(error).__name__ if error else "unknown"

        # Record error
        self.collector.record_error(
            error_type=error_type,
            endpoint=endpoint,
            model=model,
            service_type=service_type,
        )

        # Record as failed request
        self.collector.record_request(
            method=context.data.get("method", "UNKNOWN"),
            endpoint=endpoint,
            model=model,
            status="error",
            service_type=service_type,
        )

        # Clean up start time and decrement active requests
        self._request_start_times.pop(request_id, None)
        self.collector.dec_active_requests()

        logger.debug(
            "metrics_request_failed",
            request_id=request_id,
            error_type=error_type,
            endpoint=endpoint,
        )

    async def _handle_provider_request(self, context: HookContext) -> None:
        """Handle PROVIDER_REQUEST_PREPARED event."""
        if not self.config.collect_request_metrics:
            return

        provider = context.provider or "unknown"
        request_id = context.metadata.get("request_id", "unknown")

        logger.debug(
            "metrics_provider_request",
            request_id=request_id,
            provider=provider,
        )

    async def _handle_provider_response(self, context: HookContext) -> None:
        """Handle PROVIDER_RESPONSE_RECEIVED event."""
        if not self.config.collect_request_metrics:
            return

        provider = context.provider or "unknown"
        request_id = context.metadata.get("request_id", "unknown")
        status_code = context.data.get("status_code", 200)

        logger.debug(
            "metrics_provider_response",
            request_id=request_id,
            provider=provider,
            status_code=status_code,
        )

    async def _handle_provider_error(self, context: HookContext) -> None:
        """Handle PROVIDER_ERROR event."""
        if not self.config.collect_error_metrics or not self.collector:
            return

        provider = context.provider or "unknown"
        request_id = context.metadata.get("request_id", "unknown")
        error = context.error
        error_type = type(error).__name__ if error else "unknown"

        # Record provider error
        self.collector.record_error(
            error_type=f"provider_{error_type}",
            endpoint=context.data.get("endpoint", "/"),
            model=context.data.get("model"),
            service_type=provider,
        )

        logger.debug(
            "metrics_provider_error",
            request_id=request_id,
            provider=provider,
            error_type=error_type,
        )

    async def _handle_stream_start(self, context: HookContext) -> None:
        """Handle PROVIDER_STREAM_START event."""
        request_id = context.data.get("request_id", "unknown")
        provider = context.provider or "unknown"

        logger.debug(
            "metrics_stream_started",
            request_id=request_id,
            provider=provider,
        )

    async def _handle_stream_chunk(self, context: HookContext) -> None:
        """Handle PROVIDER_STREAM_CHUNK event."""
        # We might not want to record metrics for every chunk
        # due to performance considerations
        pass

    async def _handle_stream_end(self, context: HookContext) -> None:
        """Handle PROVIDER_STREAM_END event."""
        if not self.config.collect_token_metrics or not self.collector:
            return

        request_id = context.data.get("request_id", "unknown")
        provider = context.provider or "unknown"
        usage_metrics = context.data.get("usage_metrics", {})
        model = context.data.get("model")

        # Record streaming token metrics
        if usage_metrics:
            if input_tokens := usage_metrics.get("input_tokens"):
                self.collector.record_tokens(
                    token_count=input_tokens,
                    token_type="input",
                    model=model,
                    service_type=provider,
                )
            if output_tokens := usage_metrics.get("output_tokens"):
                self.collector.record_tokens(
                    token_count=output_tokens,
                    token_type="output",
                    model=model,
                    service_type=provider,
                )

        logger.debug(
            "metrics_stream_ended",
            request_id=request_id,
            provider=provider,
            usage_metrics=usage_metrics,
        )

    def get_collector(self) -> PrometheusMetrics | None:
        """Get the Prometheus metrics collector instance.

        Returns:
            The metrics collector or None if disabled
        """
        return self.collector

    def get_pushgateway_client(self) -> PushgatewayClient | None:
        """Get the Pushgateway client instance.

        Returns:
            The pushgateway client or None if disabled
        """
        return self.pushgateway

    async def push_metrics(self) -> bool:
        """Push current metrics to Pushgateway.

        Returns:
            True if push succeeded, False otherwise
        """
        if not self.pushgateway or not self.collector or not self.collector.registry:
            return False

        return self.pushgateway.push_metrics(self.collector.registry)
