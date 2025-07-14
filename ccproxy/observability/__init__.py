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

from .config import configure_observability
from .context import (
    RequestContext,
    get_context_tracker,
    request_context,
    timed_operation,
    tracked_request_context,
)
from .metrics import PrometheusMetrics, get_metrics, reset_metrics
from .pipeline import (
    LogToStoragePipeline,
    create_structlog_processor,
    enqueue_log_event,
    get_pipeline,
    pipeline_context,
)
from .pushgateway import (
    PushgatewayClient,
    get_pushgateway_client,
    reset_pushgateway_client,
)
from .scheduler import (
    ObservabilityScheduler,
    get_scheduler,
    scheduler_context,
    start_scheduler,
    stop_scheduler,
)


__all__ = [
    # Configuration
    "configure_observability",
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
    # Log-to-storage pipeline
    "LogToStoragePipeline",
    "get_pipeline",
    "enqueue_log_event",
    "create_structlog_processor",
    "pipeline_context",
    # Pushgateway
    "PushgatewayClient",
    "get_pushgateway_client",
    "reset_pushgateway_client",
    # Scheduler
    "ObservabilityScheduler",
    "get_scheduler",
    "scheduler_context",
    "start_scheduler",
    "stop_scheduler",
]
