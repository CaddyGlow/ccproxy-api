"""
Observability module for the Claude Code Proxy API.

This module provides comprehensive observability capabilities including metrics collection,
structured logging, request context tracking, and observability pipeline management.

The observability system follows a hybrid architecture that combines:
- Real-time metrics collection and aggregation
- Structured logging with correlation IDs
- Request context propagation across service boundaries
- Pluggable pipeline for metrics export and alerting

Components:
- metrics: Core metrics collection, aggregation, and export functionality
- logging: Structured logging configuration and context-aware loggers
- context: Request context tracking and correlation across async operations
- pipeline: Observability data pipeline for metrics export and alerting
"""

from .context import (
    RequestContext,
    get_context_tracker,
    request_context,
    timed_operation,
    tracked_request_context,
)
from .metrics import PrometheusMetrics, get_metrics, reset_metrics
from .pushgateway import (
    PushgatewayClient,
    get_pushgateway_client,
    reset_pushgateway_client,
)


__all__ = [
    # Configuration
    # Context management
    "RequestContext",
    "request_context",
    "tracked_request_context",
    "timed_operation",
    "get_context_tracker",
    # Prometheus metrics
    "PrometheusMetrics",
    "get_metrics",
    "reset_metrics",
    # Pushgateway
    "PushgatewayClient",
    "get_pushgateway_client",
    "reset_pushgateway_client",
]
