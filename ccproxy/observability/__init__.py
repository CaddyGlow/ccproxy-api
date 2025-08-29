"""
Observability module for the CCProxy API.

This module provides core observability components that are still actively used
across the application. Most observability features have been migrated to the
plugin system (metrics, request_tracer, access_log plugins).

Remaining components:
- context: Request context tracking and correlation across async operations
- streaming_response: Streaming response with automatic logging
- storage: DuckDB storage backend for metrics and access logs
"""

from ccproxy.core.request_context import (
    RequestContext,
    get_context_tracker,
    request_context,
    timed_operation,
    tracked_request_context,
)

from .streaming_response import StreamingResponseWithLogging


__all__ = [
    # Context management
    "RequestContext",
    "request_context",
    "tracked_request_context",
    "timed_operation",
    "get_context_tracker",
    # Streaming response
    "StreamingResponseWithLogging",
]
