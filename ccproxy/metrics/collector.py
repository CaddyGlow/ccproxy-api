"""Prometheus metrics collector for Claude Code Proxy API Server."""

import re
from typing import Optional

from prometheus_client import Counter, Gauge, Histogram

from ccproxy.metrics.models import (
    ErrorMetrics,
    HTTPMetrics,
    ModelMetrics,
    UserAgentCategory,
)


# HTTP Request Metrics
http_requests_total = Counter(
    "ccproxy_http_requests_total",
    "Total HTTP requests processed",
    ["method", "endpoint", "status", "api_type", "user_agent_category"],
)

http_request_duration_seconds = Histogram(
    "ccproxy_http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "endpoint", "api_type", "user_agent_category"],
    buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 25.0, 50.0, 100.0, float("inf")],
)

http_request_size_bytes = Histogram(
    "ccproxy_http_request_size_bytes",
    "HTTP request size in bytes",
    ["method", "endpoint", "api_type"],
    buckets=[
        100,
        1000,
        10000,
        50000,
        100000,
        500000,
        1000000,
        5000000,
        float("inf"),
    ],
)

# Active Requests Gauge
active_requests = Gauge(
    "ccproxy_active_requests",
    "Number of active HTTP requests",
    ["api_type"],
)

# Model Usage Metrics
model_requests_total = Counter(
    "ccproxy_model_requests_total",
    "Total model requests processed",
    ["model", "api_type", "endpoint", "streaming"],
)

model_tokens_total = Counter(
    "ccproxy_model_tokens_total",
    "Total tokens processed",
    ["model", "api_type", "endpoint", "token_type"],
)

model_cost_total = Counter(
    "ccproxy_model_cost_total",
    "Total estimated cost in USD",
    ["model", "api_type", "endpoint"],
)

# Error Metrics
error_total = Counter(
    "ccproxy_errors_total",
    "Total errors encountered",
    ["error_type", "endpoint", "status_code", "api_type", "user_agent_category"],
)


def categorize_user_agent(user_agent: str | None) -> UserAgentCategory:
    """Categorize user agent string into predefined categories.

    Args:
        user_agent: User agent string from HTTP request

    Returns:
        UserAgentCategory enum value
    """
    if not user_agent:
        return UserAgentCategory.OTHER

    user_agent_lower = user_agent.lower()

    # Check for specific SDKs and tools
    if "python" in user_agent_lower or "requests" in user_agent_lower:
        if "anthropic" in user_agent_lower:
            return UserAgentCategory.ANTHROPIC_SDK
        elif "openai" in user_agent_lower:
            return UserAgentCategory.OPENAI_SDK
        else:
            return UserAgentCategory.PYTHON_SDK

    if "node" in user_agent_lower or "javascript" in user_agent_lower:
        return UserAgentCategory.NODEJS

    if "curl" in user_agent_lower:
        return UserAgentCategory.CURL

    if "postman" in user_agent_lower:
        return UserAgentCategory.POSTMAN

    # Check for browser patterns
    browser_patterns = [
        r"mozilla",
        r"chrome",
        r"firefox",
        r"safari",
        r"edge",
        r"opera",
    ]

    if any(re.search(pattern, user_agent_lower) for pattern in browser_patterns):
        return UserAgentCategory.BROWSER

    # Check for SDK patterns
    if "anthropic" in user_agent_lower:
        return UserAgentCategory.ANTHROPIC_SDK

    if "openai" in user_agent_lower:
        return UserAgentCategory.OPENAI_SDK

    return UserAgentCategory.OTHER


class MetricsCollector:
    """Collector for application metrics using Prometheus client."""

    def __init__(self) -> None:
        """Initialize the metrics collector."""
        pass

    def record_http_request(self, metrics: HTTPMetrics) -> None:
        """Record HTTP request metrics.

        Args:
            metrics: HTTP request metrics data
        """
        labels = [
            metrics.method,
            metrics.endpoint,
            str(metrics.status_code),
            metrics.api_type,
            metrics.user_agent_category.value,
        ]

        # Record request count
        http_requests_total.labels(*labels).inc()

        # Record duration
        duration_labels = [
            metrics.method,
            metrics.endpoint,
            metrics.api_type,
            metrics.user_agent_category.value,
        ]
        http_request_duration_seconds.labels(*duration_labels).observe(
            metrics.duration_seconds
        )

        # Record request size
        size_labels = [
            metrics.method,
            metrics.endpoint,
            metrics.api_type,
        ]
        http_request_size_bytes.labels(*size_labels).observe(metrics.request_size_bytes)

    def record_model_usage(self, metrics: ModelMetrics) -> None:
        """Record model usage metrics.

        Args:
            metrics: Model usage metrics data
        """
        # Record model requests
        model_labels = [
            metrics.model,
            metrics.api_type,
            metrics.endpoint,
            str(metrics.streaming).lower(),
        ]
        model_requests_total.labels(*model_labels).inc()

        # Record tokens
        token_labels_base = [
            metrics.model,
            metrics.api_type,
            metrics.endpoint,
        ]

        if metrics.input_tokens > 0:
            model_tokens_total.labels(*token_labels_base, "input").inc(
                metrics.input_tokens
            )

        if metrics.output_tokens > 0:
            model_tokens_total.labels(*token_labels_base, "output").inc(
                metrics.output_tokens
            )

        if metrics.cache_creation_input_tokens > 0:
            model_tokens_total.labels(*token_labels_base, "cache_creation").inc(
                metrics.cache_creation_input_tokens
            )

        if metrics.cache_read_input_tokens > 0:
            model_tokens_total.labels(*token_labels_base, "cache_read").inc(
                metrics.cache_read_input_tokens
            )

        # Record estimated cost
        if metrics.estimated_cost > 0:
            cost_labels = [
                metrics.model,
                metrics.api_type,
                metrics.endpoint,
            ]
            model_cost_total.labels(*cost_labels).inc(metrics.estimated_cost)

    def record_error(self, metrics: ErrorMetrics) -> None:
        """Record error metrics.

        Args:
            metrics: Error metrics data
        """
        labels = [
            metrics.error_type,
            metrics.endpoint,
            str(metrics.status_code),
            metrics.api_type,
            metrics.user_agent_category.value,
        ]

        error_total.labels(*labels).inc()

    def increment_active_requests(self, api_type: str) -> None:
        """Increment active requests counter.

        Args:
            api_type: API type (anthropic, openai)
        """
        active_requests.labels(api_type).inc()

    def decrement_active_requests(self, api_type: str) -> None:
        """Decrement active requests counter.

        Args:
            api_type: API type (anthropic, openai)
        """
        active_requests.labels(api_type).dec()


# Global metrics collector instance
_metrics_collector: MetricsCollector | None = None


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector instance.

    Returns:
        MetricsCollector instance
    """
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector
