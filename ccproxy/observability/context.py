"""
Request context management with timing and correlation.

This module provides context managers and utilities for tracking request lifecycle,
timing measurements, and correlation across async operations. Uses structlog for
rich business event logging.

Key features:
- Accurate timing measurement using time.perf_counter()
- Request correlation with unique IDs
- Structured logging integration
- Async-safe context management
- Exception handling and error tracking
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import structlog


logger = structlog.get_logger(__name__)


@dataclass
class RequestContext:
    """
    Context object for tracking request state and metadata.

    Provides access to request ID, timing information, and structured logger
    with automatically injected context.
    """

    request_id: str
    start_time: float
    logger: structlog.BoundLogger
    metadata: dict[str, Any] = field(default_factory=dict)
    storage: Any | None = None  # Optional DuckDB storage instance
    log_timestamp: datetime | None = None  # Datetime for consistent logging filenames

    @property
    def duration_ms(self) -> float:
        """Get current duration in milliseconds."""
        return (time.perf_counter() - self.start_time) * 1000

    @property
    def duration_seconds(self) -> float:
        """Get current duration in seconds."""
        return time.perf_counter() - self.start_time

    def add_metadata(self, **kwargs: Any) -> None:
        """Add metadata to the request context."""
        self.metadata.update(kwargs)
        # Update logger context
        self.logger = self.logger.bind(**kwargs)

    def log_event(self, event: str, **kwargs: Any) -> None:
        """Log an event with current context and timing."""
        self.logger.info(
            event, request_id=self.request_id, duration_ms=self.duration_ms, **kwargs
        )

    def get_log_timestamp_prefix(self) -> str:
        """Get timestamp prefix for consistent log filenames.

        Returns:
            Timestamp string in YYYYMMDDhhmmss format (UTC)
        """
        if self.log_timestamp:
            return self.log_timestamp.strftime("%Y%m%d%H%M%S")
        else:
            # Fallback to current time if not set
            return datetime.now(UTC).strftime("%Y%m%d%H%M%S")


@asynccontextmanager
async def request_context(
    request_id: str | None = None,
    storage: Any | None = None,
    metrics: Any | None = None,
    log_timestamp: datetime | None = None,
    **initial_context: Any,
) -> AsyncGenerator[RequestContext, None]:
    """
    Context manager for tracking complete request lifecycle with timing.

    Automatically logs request start/success/error events with accurate timing.
    Provides structured logging with request correlation.

    Args:
        request_id: Unique request identifier (generated if not provided)
        storage: Optional storage backend for access logs
        metrics: Optional PrometheusMetrics instance for active request tracking
        **initial_context: Initial context to include in all log events

    Yields:
        RequestContext: Context object with timing and logging capabilities

    Example:
        async with request_context(method="POST", path="/v1/messages") as ctx:
            ctx.add_metadata(model="claude-3-5-sonnet")
            # Process request
            ctx.log_event("request_processed", tokens=150)
            # Context automatically logs success with timing
    """
    if request_id is None:
        request_id = str(uuid.uuid4())

    # Create logger with bound context
    request_logger = logger.bind(request_id=request_id, **initial_context)

    # Record start time
    start_time = time.perf_counter()

    # Log request start
    request_logger.debug(
        "request_start", request_id=request_id, timestamp=time.time(), **initial_context
    )

    # Emit SSE event for real-time dashboard updates
    await _emit_request_start_event(request_id, initial_context)

    # Increment active requests if metrics provided
    if metrics:
        metrics.inc_active_requests()

    # Create context object
    ctx = RequestContext(
        request_id=request_id,
        start_time=start_time,
        logger=request_logger,
        metadata=dict(initial_context),
        storage=storage,
        log_timestamp=log_timestamp,
    )

    try:
        yield ctx

        # Log successful completion with comprehensive access log
        duration_ms = ctx.duration_ms

        # Use the new unified access logger for comprehensive logging
        from ccproxy.observability.access_logger import log_request_access

        await log_request_access(
            context=ctx,
            # Extract client info from metadata if available
            client_ip=ctx.metadata.get("client_ip"),
            user_agent=ctx.metadata.get("user_agent"),
            query=ctx.metadata.get("query"),
            storage=ctx.storage,  # Pass storage from context
        )

        # Also keep the original request_success event for debugging
        request_logger.debug(
            "request_success",
            request_id=request_id,
            duration_ms=duration_ms,
            duration_seconds=ctx.duration_seconds,
            **ctx.metadata,
        )

    except Exception as e:
        # Log error with timing
        duration_ms = ctx.duration_ms
        error_type = type(e).__name__

        request_logger.error(
            "request_error",
            request_id=request_id,
            duration_ms=duration_ms,
            duration_seconds=ctx.duration_seconds,
            error_type=error_type,
            error_message=str(e),
            **ctx.metadata,
        )

        # Emit SSE event for real-time dashboard updates
        await _emit_request_error_event(request_id, error_type, str(e), ctx.metadata)

        # Re-raise the exception
        raise
    finally:
        # Decrement active requests if metrics provided
        if metrics:
            metrics.dec_active_requests()


@asynccontextmanager
async def timed_operation(
    operation_name: str, request_id: str | None = None, **context: Any
) -> AsyncGenerator[dict[str, Any], None]:
    """
    Context manager for timing individual operations within a request.

    Useful for measuring specific parts of request processing like
    API calls, database queries, or data processing steps.

    Args:
        operation_name: Name of the operation being timed
        request_id: Associated request ID for correlation
        **context: Additional context for logging

    Yields:
        Dict with timing information and logger

    Example:
        async with timed_operation("claude_api_call", request_id=ctx.request_id) as op:
            response = await api_client.call()
            op["response_size"] = len(response)
            # Automatically logs operation timing
    """
    start_time = time.perf_counter()
    operation_id = str(uuid.uuid4())

    # Create operation logger
    op_logger = logger.bind(
        operation_name=operation_name,
        operation_id=operation_id,
        request_id=request_id,
        **context,
    )

    # Log operation start (only for important operations)
    if operation_name in ("claude_api_call", "request_processing", "auth_check"):
        op_logger.debug(
            "operation_start",
            operation_name=operation_name,
            **context,
        )

    # Operation context
    op_context = {
        "operation_id": operation_id,
        "logger": op_logger,
        "start_time": start_time,
    }

    try:
        yield op_context

        # Log successful completion (only for important operations)
        duration_ms = (time.perf_counter() - start_time) * 1000
        if operation_name in ("claude_api_call", "request_processing", "auth_check"):
            op_logger.info(
                "operation_success",
                operation_name=operation_name,
                duration_ms=duration_ms,
                **{
                    k: v
                    for k, v in op_context.items()
                    if k not in ("logger", "start_time")
                },
            )

    except Exception as e:
        # Log operation error
        duration_ms = (time.perf_counter() - start_time) * 1000
        error_type = type(e).__name__

        op_logger.error(
            "operation_error",
            operation_name=operation_name,
            duration_ms=duration_ms,
            error_type=error_type,
            error_message=str(e),
            **{
                k: v for k, v in op_context.items() if k not in ("logger", "start_time")
            },
        )

        # Re-raise the exception
        raise


class ContextTracker:
    """
    Thread-safe tracker for managing active request contexts.

    Useful for tracking concurrent requests and their states,
    especially for metrics like active request counts.
    """

    def __init__(self) -> None:
        self._active_contexts: dict[str, RequestContext] = {}
        self._lock = asyncio.Lock()

    async def add_context(self, context: RequestContext) -> None:
        """Add an active request context."""
        async with self._lock:
            self._active_contexts[context.request_id] = context

    async def remove_context(self, request_id: str) -> RequestContext | None:
        """Remove and return a request context."""
        async with self._lock:
            return self._active_contexts.pop(request_id, None)

    async def get_context(self, request_id: str) -> RequestContext | None:
        """Get a request context by ID."""
        async with self._lock:
            return self._active_contexts.get(request_id)

    async def get_active_count(self) -> int:
        """Get the number of active requests."""
        async with self._lock:
            return len(self._active_contexts)

    async def get_all_contexts(self) -> dict[str, RequestContext]:
        """Get a copy of all active contexts."""
        async with self._lock:
            return self._active_contexts.copy()

    async def cleanup_stale_contexts(self, max_age_seconds: float = 300) -> int:
        """
        Remove contexts older than max_age_seconds.

        Args:
            max_age_seconds: Maximum age in seconds before considering stale

        Returns:
            Number of contexts removed
        """
        current_time = time.perf_counter()
        removed_count = 0

        async with self._lock:
            stale_ids = [
                request_id
                for request_id, ctx in self._active_contexts.items()
                if (current_time - ctx.start_time) > max_age_seconds
            ]

            for request_id in stale_ids:
                del self._active_contexts[request_id]
                removed_count += 1

        if removed_count > 0:
            logger.warning(
                "cleanup_stale_contexts",
                removed_count=removed_count,
                max_age_seconds=max_age_seconds,
            )

        return removed_count


# Global context tracker instance
_global_tracker: ContextTracker | None = None


def get_context_tracker() -> ContextTracker:
    """Get or create global context tracker."""
    global _global_tracker

    if _global_tracker is None:
        _global_tracker = ContextTracker()

    return _global_tracker


@asynccontextmanager
async def tracked_request_context(
    request_id: str | None = None, storage: Any | None = None, **initial_context: Any
) -> AsyncGenerator[RequestContext, None]:
    """
    Request context manager that also tracks active requests globally.

    Combines request_context() with automatic tracking in the global
    context tracker for monitoring active request counts.

    Args:
        request_id: Unique request identifier
        **initial_context: Initial context to include in log events

    Yields:
        RequestContext: Context object with timing and logging
    """
    tracker = get_context_tracker()

    async with request_context(request_id, storage=storage, **initial_context) as ctx:
        # Add to tracker
        await tracker.add_context(ctx)

        try:
            yield ctx
        finally:
            # Remove from tracker
            await tracker.remove_context(ctx.request_id)


async def _emit_request_start_event(
    request_id: str, initial_context: dict[str, Any]
) -> None:
    """Emit SSE event for request start."""
    try:
        from ccproxy.observability.sse_events import emit_sse_event

        # Create event data for SSE
        sse_data = {
            "request_id": request_id,
            "method": initial_context.get("method"),
            "path": initial_context.get("path"),
            "client_ip": initial_context.get("client_ip"),
            "user_agent": initial_context.get("user_agent"),
            "query": initial_context.get("query"),
        }

        # Remove None values
        sse_data = {k: v for k, v in sse_data.items() if v is not None}

        await emit_sse_event("request_start", sse_data)

    except Exception as e:
        # Log error but don't fail the request
        logger.debug(
            "sse_emit_failed",
            event_type="request_start",
            error=str(e),
            request_id=request_id,
        )


async def _emit_request_error_event(
    request_id: str, error_type: str, error_message: str, metadata: dict[str, Any]
) -> None:
    """Emit SSE event for request error."""
    try:
        from ccproxy.observability.sse_events import emit_sse_event

        # Create event data for SSE
        sse_data = {
            "request_id": request_id,
            "error_type": error_type,
            "error_message": error_message,
            "method": metadata.get("method"),
            "path": metadata.get("path"),
            "client_ip": metadata.get("client_ip"),
            "user_agent": metadata.get("user_agent"),
            "query": metadata.get("query"),
        }

        # Remove None values
        sse_data = {k: v for k, v in sse_data.items() if v is not None}

        await emit_sse_event("request_error", sse_data)

    except Exception as e:
        # Log error but don't fail the request
        logger.debug(
            "sse_emit_failed",
            event_type="request_error",
            error=str(e),
            request_id=request_id,
        )
