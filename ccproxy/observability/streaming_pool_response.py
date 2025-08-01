"""FastAPI StreamingResponse with automatic pool client cleanup on disconnection.

This module provides a StreamingResponseWithPoolCleanup class that wraps
streaming responses and monitors for client disconnections, automatically
cleaning up pool resources when clients disconnect unexpectedly.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncGenerator, AsyncIterator
from typing import TYPE_CHECKING, Any

import structlog
from fastapi import Request
from fastapi.responses import StreamingResponse

from ccproxy.observability.access_logger import log_request_access


if TYPE_CHECKING:
    from ccproxy.observability.context import RequestContext
    from ccproxy.observability.metrics import PrometheusMetrics

logger = structlog.get_logger(__name__)


class StreamingResponseWithPoolCleanup(StreamingResponse):
    """FastAPI StreamingResponse that monitors client disconnection for pool cleanup.

    This class wraps a streaming response generator to monitor for client disconnections
    during streaming. When a disconnection is detected, it signals the need for cleanup
    but doesn't directly interact with the pool (to avoid circular dependencies).
    """

    def __init__(
        self,
        content: AsyncGenerator[bytes, None] | AsyncIterator[bytes],
        request: Request,
        request_context: RequestContext,
        metrics: PrometheusMetrics | None = None,
        status_code: int = 200,
        cleanup_callback: asyncio.Future[bool] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize streaming response with disconnection monitoring.

        Args:
            content: The async generator producing streaming content
            request: FastAPI request object for disconnection detection
            request_context: The request context for access logging
            metrics: Optional PrometheusMetrics instance for recording metrics
            status_code: HTTP status code for the response
            cleanup_callback: Optional future to signal when cleanup is needed
            **kwargs: Additional arguments passed to StreamingResponse
        """
        # Wrap the content generator to add disconnection monitoring
        monitored_content = self._wrap_with_disconnection_monitoring(
            content,
            request,
            request_context,
            metrics,
            status_code,
            cleanup_callback,
        )
        super().__init__(monitored_content, status_code=status_code, **kwargs)

    async def _wrap_with_disconnection_monitoring(
        self,
        content: AsyncGenerator[bytes, None] | AsyncIterator[bytes],
        request: Request,
        context: RequestContext,
        metrics: PrometheusMetrics | None,
        status_code: int,
        cleanup_callback: asyncio.Future[bool] | None,
    ) -> AsyncGenerator[bytes, None]:
        """Wrap content generator with client disconnection monitoring.

        Args:
            content: The original content generator
            request: FastAPI request object for disconnection detection
            context: Request context for logging
            metrics: Optional metrics instance
            status_code: HTTP status code
            cleanup_callback: Optional future to signal cleanup needed

        Yields:
            bytes: Content chunks from the original generator
        """
        request_id = context.request_id
        disconnected_during_stream = False
        total_chunks_yielded = 0
        total_bytes_yielded = 0

        logger.debug(
            "streaming_pool_monitor_start",
            request_id=request_id,
        )

        # Start disconnection monitoring task
        monitor_task = asyncio.create_task(
            self._monitor_disconnection(request, request_id)
        )

        try:
            # Stream all content from the original generator
            async for chunk in content:
                # Check if monitor detected disconnection
                if monitor_task.done():
                    try:
                        disconnected = await monitor_task
                        if disconnected:
                            disconnected_during_stream = True
                            logger.warning(
                                "streaming_pool_client_disconnected",
                                request_id=request_id,
                                chunks_sent=total_chunks_yielded,
                                bytes_sent=total_bytes_yielded,
                            )
                            break
                    except Exception:
                        pass  # Monitor task failed, continue streaming

                total_chunks_yielded += 1
                total_bytes_yielded += len(chunk)
                yield chunk

        except Exception as e:
            logger.warning(
                "streaming_pool_exception",
                request_id=request_id,
                total_chunks_yielded=total_chunks_yielded,
                total_bytes_yielded=total_bytes_yielded,
                exception_type=type(e).__name__,
                exception_message=str(e),
            )
            # Check if this is a client disconnection exception
            if self._is_disconnect_exception(e):
                disconnected_during_stream = True
                logger.warning(
                    "streaming_pool_disconnect_exception",
                    request_id=request_id,
                )

        finally:
            # Cancel monitor task
            monitor_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await monitor_task

            # Signal cleanup needed if disconnected
            if (
                disconnected_during_stream
                and cleanup_callback
                and not cleanup_callback.done()
            ):
                cleanup_callback.set_result(True)
                logger.info(
                    "streaming_pool_cleanup_signaled",
                    request_id=request_id,
                )

            # Log access when stream completes
            try:
                # Add streaming completion event type to context
                context.add_metadata(
                    event_type="streaming_complete",
                    disconnected=disconnected_during_stream,
                )

                # Check if status_code was updated in context metadata
                final_status_code = context.metadata.get("status_code", status_code)

                await log_request_access(
                    context=context,
                    status_code=final_status_code,
                    metrics=metrics,
                )
            except Exception as e:
                logger.warning(
                    "streaming_access_log_failed",
                    error=str(e),
                    request_id=request_id,
                )

            logger.debug(
                "streaming_pool_monitor_complete",
                request_id=request_id,
                disconnected=disconnected_during_stream,
                total_chunks=total_chunks_yielded,
                total_bytes=total_bytes_yielded,
            )

    async def _monitor_disconnection(self, request: Request, request_id: str) -> bool:
        """Monitor for client disconnection.

        Args:
            request: FastAPI request to monitor
            request_id: Request ID for logging

        Returns:
            True if disconnection detected, False otherwise
        """
        try:
            while True:
                if await request.is_disconnected():
                    logger.info(
                        "client_disconnect_detected",
                        request_id=request_id,
                    )
                    return True
                await asyncio.sleep(0.1)  # Check every 100ms
        except asyncio.CancelledError:
            # Normal cancellation when streaming completes
            return False

    def _is_disconnect_exception(self, exception: Exception) -> bool:
        """Check if an exception indicates client disconnection.

        Args:
            exception: The exception to check

        Returns:
            True if the exception indicates disconnection
        """
        exception_str = str(exception).lower()
        exception_type = type(exception).__name__.lower()

        # Common disconnection indicators
        disconnect_indicators = [
            "disconnected",
            "connection reset",
            "broken pipe",
            "connection aborted",
            "connection closed",
            "cancelled",
            "aborted",
        ]

        return any(
            indicator in exception_str or indicator in exception_type
            for indicator in disconnect_indicators
        )
