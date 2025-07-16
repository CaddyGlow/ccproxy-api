"""Unified access logging utilities for comprehensive request tracking.

This module provides centralized access logging functionality that can be used
across different parts of the application to generate consistent, comprehensive
access logs with complete request metadata including token usage and costs.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any, Optional

import structlog


if TYPE_CHECKING:
    from ccproxy.observability.context import RequestContext


logger = structlog.get_logger(__name__)


async def log_request_access(
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
    Also stores the access log in DuckDB if available.

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
        "cost_sdk_usd",
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

    # Log as access_log event (structured logging)
    context.logger.info("access_log", **log_data)

    # Store in DuckDB if available
    await _store_access_log(log_data)

    # Emit SSE event for real-time dashboard updates
    await _emit_access_event("request_complete", log_data)


async def _store_access_log(log_data: dict[str, Any]) -> None:
    """Store access log in DuckDB storage if available."""
    try:
        from ccproxy.config.settings import get_settings
        from ccproxy.observability.storage.duckdb_simple import SimpleDuckDBStorage

        settings = get_settings()
        if not settings.observability.duckdb_enabled:
            return

        # Initialize storage if needed
        storage = SimpleDuckDBStorage(database_path=settings.observability.duckdb_path)

        if not storage.is_enabled():
            await storage.initialize()

        # Prepare data for DuckDB storage
        storage_data = {
            "timestamp": time.time(),
            "request_id": log_data.get("request_id"),
            "method": log_data.get("method", ""),
            "endpoint": log_data.get("endpoint", log_data.get("path", "")),
            "path": log_data.get("path", ""),
            "query": log_data.get("query", ""),
            "client_ip": log_data.get("client_ip", ""),
            "user_agent": log_data.get("user_agent", ""),
            "service_type": log_data.get("service_type", ""),
            "model": log_data.get("model", ""),
            "streaming": log_data.get("streaming", False),
            "status_code": log_data.get("status_code", 200),
            "duration_ms": log_data.get("duration_ms", 0.0),
            "duration_seconds": log_data.get("duration_seconds", 0.0),
            "tokens_input": log_data.get("tokens_input", 0),
            "tokens_output": log_data.get("tokens_output", 0),
            "cache_read_tokens": log_data.get("cache_read_tokens", 0),
            "cache_write_tokens": log_data.get("cache_write_tokens", 0),
            "cost_usd": log_data.get("cost_usd", 0.0),
            "cost_sdk_usd": log_data.get("cost_sdk_usd", 0.0),
        }

        # Store asynchronously (fire and forget)
        asyncio.create_task(_write_to_storage(storage, storage_data))

    except Exception as e:
        # Log error but don't fail the request
        logger.error(
            "access_log_duckdb_error",
            error=str(e),
            request_id=log_data.get("request_id"),
        )


async def _write_to_storage(storage: Any, data: dict[str, Any]) -> None:
    """Write data to storage asynchronously."""
    try:
        await storage.store_request(data)
    except Exception as e:
        logger.error(
            "duckdb_store_error",
            error=str(e),
            request_id=data.get("request_id"),
        )


async def _emit_access_event(event_type: str, data: dict[str, Any]) -> None:
    """Emit SSE event for real-time dashboard updates."""
    try:
        from ccproxy.observability.sse_events import emit_sse_event

        # Create event data for SSE (exclude internal fields)
        sse_data = {
            "request_id": data.get("request_id"),
            "method": data.get("method"),
            "path": data.get("path"),
            "query": data.get("query"),
            "status_code": data.get("status_code"),
            "client_ip": data.get("client_ip"),
            "user_agent": data.get("user_agent"),
            "service_type": data.get("service_type"),
            "model": data.get("model"),
            "streaming": data.get("streaming"),
            "duration_ms": data.get("duration_ms"),
            "duration_seconds": data.get("duration_seconds"),
            "tokens_input": data.get("tokens_input"),
            "tokens_output": data.get("tokens_output"),
            "cost_usd": data.get("cost_usd"),
            "endpoint": data.get("endpoint"),
        }

        # Remove None values
        sse_data = {k: v for k, v in sse_data.items() if v is not None}

        await emit_sse_event(event_type, sse_data)

    except Exception as e:
        # Log error but don't fail the request
        logger.debug(
            "sse_emit_failed",
            event_type=event_type,
            error=str(e),
            request_id=data.get("request_id"),
        )


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

    # Emit SSE event for real-time dashboard updates
    # Note: This is a synchronous function, so we schedule the async emission
    try:
        import asyncio

        from ccproxy.observability.sse_events import emit_sse_event

        # Create event data for SSE
        sse_data = {
            "request_id": request_id,
            "method": method,
            "path": path,
            "client_ip": client_ip,
            "user_agent": user_agent,
            "query": query,
        }

        # Remove None values
        sse_data = {k: v for k, v in sse_data.items() if v is not None}

        # Schedule async event emission
        asyncio.create_task(emit_sse_event("request_start", sse_data))

    except Exception as e:
        # Log error but don't fail the request
        logger.debug(
            "sse_emit_failed",
            event_type="request_start",
            error=str(e),
            request_id=request_id,
        )
