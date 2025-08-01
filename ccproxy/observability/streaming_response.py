"""FastAPI StreamingResponse with automatic access logging on completion.

This module provides a reusable StreamingResponseWithLogging class that wraps
any async generator and handles access logging when the stream completes,
eliminating code duplication between different streaming endpoints.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, AsyncIterator
from typing import TYPE_CHECKING, Any

import structlog
from fastapi.responses import StreamingResponse

from ccproxy.observability.access_logger import log_request_access


if TYPE_CHECKING:
    from ccproxy.observability.context import RequestContext
    from ccproxy.observability.metrics import PrometheusMetrics

logger = structlog.get_logger(__name__)


class StreamingResponseWithLogging(StreamingResponse):
    """FastAPI StreamingResponse that triggers access logging on completion.

    This class wraps a streaming response generator to automatically trigger
    access logging when the stream completes (either successfully or with an error).
    This eliminates the need for manual access logging in individual stream processors.
    """

    def __init__(
        self,
        content: AsyncGenerator[bytes, None] | AsyncIterator[bytes],
        request_context: RequestContext,
        metrics: PrometheusMetrics | None = None,
        status_code: int = 200,
        **kwargs: Any,
    ) -> None:
        """Initialize streaming response with logging capability.

        Args:
            content: The async generator producing streaming content
            request_context: The request context for access logging
            metrics: Optional PrometheusMetrics instance for recording metrics
            status_code: HTTP status code for the response
            **kwargs: Additional arguments passed to StreamingResponse
        """
        # Wrap the content generator to add logging
        logged_content = self._wrap_with_logging(
            content, request_context, metrics, status_code
        )
        super().__init__(logged_content, status_code=status_code, **kwargs)

    async def _wrap_with_logging(
        self,
        content: AsyncGenerator[bytes, None] | AsyncIterator[bytes],
        context: RequestContext,
        metrics: PrometheusMetrics | None,
        status_code: int,
    ) -> AsyncGenerator[bytes, None]:
        """Wrap content generator with access logging on completion.

        Args:
            content: The original content generator
            context: Request context for logging
            metrics: Optional metrics instance
            status_code: HTTP status code

        Yields:
            bytes: Content chunks from the original generator
        """
        try:
            # Stream all content from the original generator
            async for chunk in content:
                yield chunk
        except GeneratorExit:
            # Client disconnected - log this and re-raise to propagate to underlying generators
            logger.info(
                "streaming_response_client_disconnected",
                request_id=context.request_id,
                message="Client disconnected from streaming response, propagating GeneratorExit",
            )
            # CRITICAL: Re-raise GeneratorExit to propagate disconnect to create_listener()
            raise
        finally:
            # Log access when stream completes (success or error)
            try:
                # Add streaming completion event type to context
                context.add_metadata(event_type="streaming_complete")

                # Check if status_code was updated in context metadata (e.g., due to error)
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
                    request_id=context.request_id,
                )
