"""Unified access logging utilities for comprehensive request tracking.

This module provides centralized access logging functionality that can be used
across different parts of the application to generate consistent, comprehensive
access logs with complete request metadata including token usage and costs.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Optional

import structlog


if TYPE_CHECKING:
    from ccproxy.observability.context import RequestContext


logger = structlog.get_logger(__name__)


def log_request_access(
    context: RequestContext,
    status_code: int | None = None,
    client_ip: str | None = None,
    user_agent: str | None = None,
    method: str | None = None,
    path: str | None = None,
    query: str | None = None,
    error_message: str | None = None,
    **additional_metadata: Any,
) -> None:
    """Log comprehensive access information for a request.

    This function generates a unified access log entry with complete request
    metadata including timing, tokens, costs, and any additional context.

    Args:
        context: Request context with timing and metadata
        status_code: HTTP status code
        client_ip: Client IP address
        user_agent: User agent string
        method: HTTP method
        path: Request path
        query: Query parameters
        error_message: Error message if applicable
        **additional_metadata: Any additional fields to include
    """
    # Extract basic request info from context metadata if not provided
    ctx_metadata = context.metadata
    method = method or ctx_metadata.get("method")
    path = path or ctx_metadata.get("path")
    status_code = status_code or ctx_metadata.get("status_code")

    # Prepare comprehensive log data
    log_data = {
        "request_id": context.request_id,
        "method": method,
        "path": path,
        "query": query,
        "status_code": status_code,
        "client_ip": client_ip,
        "user_agent": user_agent,
        "duration_ms": context.duration_ms,
        "duration_seconds": context.duration_seconds,
        "error_message": error_message,
    }

    # Add token and cost metrics if available
    token_fields = [
        "tokens_input",
        "tokens_output",
        "cache_read_tokens",
        "cache_write_tokens",
        "cost_usd",
    ]

    for field in token_fields:
        value = ctx_metadata.get(field)
        if value is not None:
            log_data[field] = value

    # Add service and endpoint info
    service_fields = [
        "endpoint",
        "model",
        "streaming",
        "service_type",
    ]

    for field in service_fields:
        value = ctx_metadata.get(field)
        if value is not None:
            log_data[field] = value

    # Add any additional metadata provided
    log_data.update(additional_metadata)

    # Remove None values to keep log clean
    log_data = {k: v for k, v in log_data.items() if v is not None}

    # Log as access_log event
    context.logger.info("access_log", **log_data)


def log_request_start(
    request_id: str,
    method: str,
    path: str,
    client_ip: str | None = None,
    user_agent: str | None = None,
    query: str | None = None,
    **additional_metadata: Any,
) -> None:
    """Log request start event with basic information.

    This is used for early/middleware logging when full context isn't available yet.

    Args:
        request_id: Request identifier
        method: HTTP method
        path: Request path
        client_ip: Client IP address
        user_agent: User agent string
        query: Query parameters
        **additional_metadata: Any additional fields to include
    """
    log_data = {
        "request_id": request_id,
        "method": method,
        "path": path,
        "client_ip": client_ip,
        "user_agent": user_agent,
        "query": query,
        "event_type": "request_start",
        "timestamp": time.time(),
    }

    # Add any additional metadata
    log_data.update(additional_metadata)

    # Remove None values
    log_data = {k: v for k, v in log_data.items() if v is not None}

    logger.info("access_log_start", **log_data)
