"""Metrics module for Claude Code Proxy API Server.

This module provides Prometheus metrics collection for monitoring HTTP requests,
model usage, errors, and other operational metrics.
"""

from ccproxy.metrics.calculator import (
    CostCalculator,
    ModelPricing,
    get_cost_calculator,
)
from ccproxy.metrics.collector import (
    MetricsCollector,
    active_requests,
    categorize_user_agent,
    error_total,
    get_metrics_collector,
    http_request_duration_seconds,
    http_request_size_bytes,
    http_requests_total,
    model_cost_total,
    model_requests_total,
    model_tokens_total,
)
from ccproxy.metrics.database import (
    Base,
    DailyAggregate,
    MetricsSnapshot,
    RequestLog,
    create_database_engine,
    create_session_factory,
    create_tables,
    deserialize_labels,
    serialize_labels,
)
from ccproxy.metrics.models import (
    ErrorMetrics,
    HTTPMetrics,
    ModelMetrics,
    UserAgentCategory,
)
from ccproxy.metrics.storage import (
    MetricsStorage,
    close_metrics_storage,
    get_metrics_storage,
)
from ccproxy.metrics.sync_storage import (
    SyncMetricsStorage,
    close_sync_metrics_storage,
    get_sync_metrics_storage,
)


__all__ = [
    # Calculator classes and functions
    "CostCalculator",
    "ModelPricing",
    "get_cost_calculator",
    # Collector classes and functions
    "MetricsCollector",
    "categorize_user_agent",
    "get_metrics_collector",
    # Prometheus metrics
    "http_request_duration_seconds",
    "http_request_size_bytes",
    "http_requests_total",
    "active_requests",
    "model_requests_total",
    "model_tokens_total",
    "model_cost_total",
    "error_total",
    # Database models
    "Base",
    "MetricsSnapshot",
    "RequestLog",
    "DailyAggregate",
    "create_database_engine",
    "create_session_factory",
    "create_tables",
    "serialize_labels",
    "deserialize_labels",
    # Storage layer
    "MetricsStorage",
    "get_metrics_storage",
    "close_metrics_storage",
    # Synchronous storage layer
    "SyncMetricsStorage",
    "get_sync_metrics_storage",
    "close_sync_metrics_storage",
    # Pydantic models
    "HTTPMetrics",
    "ModelMetrics",
    "ErrorMetrics",
    "UserAgentCategory",
]
