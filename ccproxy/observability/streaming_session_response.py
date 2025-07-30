"""FastAPI StreamingResponse with automatic session interruption on client disconnection.

This module provides a StreamingResponseWithSessionInterrupt class that wraps
streaming responses and monitors for client disconnections, automatically
interrupting Claude sessions to free up resources when clients disconnect.
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
    from ccproxy.services.claude_sdk_service import ClaudeSDKService

logger = structlog.get_logger(__name__)


class StreamingResponseWithSessionInterrupt(StreamingResponse):
    """FastAPI StreamingResponse that monitors client disconnection and interrupts sessions.

    This class wraps a streaming response generator to monitor for client disconnections
    during streaming and automatically interrupt Claude sessions when disconnection is detected.
    It also provides access logging when the stream completes.
    """

    def __init__(
        self,
        content: AsyncGenerator[bytes, None] | AsyncIterator[bytes],
        request: Request,
        session_id: str,
        claude_service: ClaudeSDKService,
        request_context: RequestContext,
        metrics: PrometheusMetrics | None = None,
        status_code: int = 200,
        **kwargs: Any,
    ) -> None:
        """Initialize streaming response with session interruption capability.

        Args:
            content: The async generator producing streaming content
            request: FastAPI request object for disconnection detection
            session_id: Claude session ID to monitor and potentially interrupt
            claude_service: Claude service for session interruption
            request_context: The request context for access logging
            metrics: Optional PrometheusMetrics instance for recording metrics
            status_code: HTTP status code for the response
            **kwargs: Additional arguments passed to StreamingResponse
        """
        # Wrap the content generator to add session monitoring and logging
        monitored_content = self._wrap_with_session_monitoring(
            content,
            request,
            session_id,
            claude_service,
            request_context,
            metrics,
            status_code,
        )
        super().__init__(monitored_content, status_code=status_code, **kwargs)

    async def _wrap_with_session_monitoring(
        self,
        content: AsyncGenerator[bytes, None] | AsyncIterator[bytes],
        request: Request,
        session_id: str,
        claude_service: ClaudeSDKService,
        context: RequestContext,
        metrics: PrometheusMetrics | None,
        status_code: int,
    ) -> AsyncGenerator[bytes, None]:
        """Wrap content generator with client disconnection monitoring and session interruption.

        Args:
            content: The original content generator
            request: FastAPI request object for disconnection detection
            session_id: Claude session ID to monitor
            claude_service: Claude service for session interruption
            context: Request context for logging
            metrics: Optional metrics instance
            status_code: HTTP status code

        Yields:
            bytes: Content chunks from the original generator
        """
        request_id = context.request_id
        disconnected_during_stream = False
        total_chunks_yielded = 0
        total_bytes_yielded = 0

        logger.debug(
            "streaming_session_monitor_start",
            session_id=session_id,
            request_id=request_id,
        )

        try:
            # Stream all content from the original generator with concurrent disconnection monitoring
            chunk_count = 0

            # Create async iterator from content generator
            content_iter = content.__aiter__()

            # Use timeout-based chunk reading with disconnection monitoring
            async for chunk in self._monitor_stream_with_disconnection(
                content_iter, request, session_id, request_id, claude_service
            ):
                if chunk is None:  # Disconnection detected signal
                    disconnected_during_stream = True
                    break

                chunk_count += 1

                # Track yielded chunks and bytes
                total_chunks_yielded += 1
                total_bytes_yielded += len(chunk)
                yield chunk

        except Exception as e:
            logger.warning(
                "streaming_session_exception",
                session_id=session_id,
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
                    "streaming_session_disconnect_exception",
                    session_id=session_id,
                    request_id=request_id,
                    total_chunks_yielded=total_chunks_yielded,
                    total_bytes_yielded=total_bytes_yielded,
                    exception_type=type(e).__name__,
                    exception_message=str(e),
                )

                # Interrupt session due to disconnection exception
                await self._interrupt_session(
                    claude_service, session_id, request_id, "exception_disconnection"
                )

            # Re-raise the exception
            raise

        finally:
            # Log streaming completion and handle access logging
            try:
                # FIRST: Check if client is actually disconnected right now
                client_currently_disconnected = await request.is_disconnected()

                # Determine if we should interrupt the session
                should_interrupt = (
                    disconnected_during_stream or client_currently_disconnected
                )

                # Check for potential missed disconnections by analyzing chunk patterns
                potential_early_disconnect = (
                    not should_interrupt  # Only if we haven't already detected disconnection
                    and total_chunks_yielded > 0
                    and total_chunks_yielded < 3
                    and total_bytes_yielded < 1000
                )

                if potential_early_disconnect:
                    logger.warning(
                        "streaming_session_potential_early_disconnect",
                        session_id=session_id,
                        request_id=request_id,
                        total_chunks_yielded=total_chunks_yielded,
                        total_bytes_yielded=total_bytes_yielded,
                        client_currently_disconnected=client_currently_disconnected,
                        message="Stream completed with unusually few chunks - possible missed disconnection",
                    )
                    # Treat suspicious patterns as disconnection
                    should_interrupt = True
                    disconnected_during_stream = True

                # Log final status and interrupt if needed
                if should_interrupt:
                    logger.warning(
                        "streaming_session_completed_with_disconnect",
                        session_id=session_id,
                        request_id=request_id,
                        total_chunks_yielded=total_chunks_yielded,
                        total_bytes_yielded=total_bytes_yielded,
                        detected_during_stream=disconnected_during_stream,
                        detected_at_end=client_currently_disconnected,
                        message="Stream completed due to client disconnection - interrupting session",
                    )

                    # Interrupt the session since client disconnected
                    await self._interrupt_session(
                        claude_service,
                        session_id,
                        request_id,
                        "client_disconnection",
                    )
                else:
                    logger.info(
                        "streaming_session_completed_normally",
                        session_id=session_id,
                        request_id=request_id,
                        total_chunks_yielded=total_chunks_yielded,
                        total_bytes_yielded=total_bytes_yielded,
                        client_currently_disconnected=client_currently_disconnected,
                        message="Stream completed normally, session remains active",
                    )

                # Add streaming completion event type to context
                context.add_metadata(
                    event_type="streaming_complete",
                    client_disconnected=disconnected_during_stream,
                )

                # Perform access logging
                await log_request_access(
                    context=context,
                    status_code=status_code,
                    metrics=metrics,
                )

            except Exception as e:
                logger.warning(
                    "streaming_session_cleanup_failed",
                    session_id=session_id,
                    request_id=request_id,
                    error=str(e),
                    error_type=type(e).__name__,
                )

    async def _interrupt_session(
        self,
        claude_service: ClaudeSDKService,
        session_id: str,
        request_id: str,
        reason: str,
    ) -> None:
        """Interrupt a Claude session due to client disconnection.

        Args:
            claude_service: Claude service for session interruption
            session_id: The session ID to interrupt
            request_id: Request ID for logging
            reason: Reason for interruption (for logging)
        """
        try:
            interrupted = await claude_service.interrupt_session(session_id)

            logger.info(
                "streaming_session_interrupted",
                session_id=session_id,
                request_id=request_id,
                interrupted=interrupted,
                reason=reason,
            )

        except Exception as e:
            logger.error(
                "streaming_session_interrupt_failed",
                session_id=session_id,
                request_id=request_id,
                reason=reason,
                error=str(e),
                error_type=type(e).__name__,
                exc_info=True,
            )

    def _is_disconnect_exception(self, exception: Exception) -> bool:
        """Check if an exception indicates client disconnection.

        Args:
            exception: The exception to check

        Returns:
            True if this appears to be a disconnection exception
        """
        from anyio import BrokenResourceError, ClosedResourceError
        from starlette.exceptions import HTTPException

        # Check exception type
        if isinstance(
            exception,
            ConnectionError
            | BrokenResourceError
            | ClosedResourceError
            | asyncio.CancelledError,
        ):
            return True

        # Check HTTP exceptions with client disconnect status codes
        if isinstance(exception, HTTPException) and exception.status_code in (499, 400):
            return True

        # Check exception message for disconnect indicators
        message = str(exception).lower()
        disconnect_indicators = [
            "disconnect",
            "broken",
            "closed",
            "cancelled",
            "aborted",
        ]
        return any(indicator in message for indicator in disconnect_indicators)

    async def _monitor_stream_with_disconnection(
        self,
        content_iter: AsyncIterator[bytes],
        request: Request,
        session_id: str,
        request_id: str,
        claude_service: ClaudeSDKService,
    ) -> AsyncIterator[bytes | None]:
        """Monitor stream with concurrent disconnection detection.

        This method solves the issue where streams get stuck (like with /status commands)
        and disconnection detection never runs because no chunks are yielded.

        Args:
            content_iter: The async iterator for content chunks
            request: FastAPI request for disconnection detection
            session_id: Session ID for logging and interruption
            request_id: Request ID for logging
            claude_service: Claude service for session interruption

        Yields:
            bytes: Content chunks from the stream
            None: Signal that disconnection was detected
        """
        import asyncio

        disconnection_check_interval = 2.0  # Check every 2 seconds
        chunk_timeout = 30.0  # Timeout for waiting for next chunk

        try:
            while True:
                try:
                    # Wait for next chunk with timeout and concurrent disconnection monitoring
                    next_chunk_task: asyncio.Task[bytes] = asyncio.create_task(
                        self._get_next_chunk(content_iter)
                    )
                    disconnection_check_task: asyncio.Task[None] = asyncio.create_task(
                        self._periodic_disconnection_check(
                            request,
                            session_id,
                            request_id,
                            disconnection_check_interval,
                        )
                    )

                    # Race between getting next chunk and detecting disconnection
                    done, pending = await asyncio.wait(
                        [next_chunk_task, disconnection_check_task],
                        timeout=chunk_timeout,
                        return_when=asyncio.FIRST_COMPLETED,
                    )

                    # Cancel any remaining tasks
                    for task in pending:
                        task.cancel()
                        with contextlib.suppress(asyncio.CancelledError):
                            await task

                    if not done:
                        # Timeout occurred - check for disconnection and interrupt if stuck
                        logger.warning(
                            "streaming_chunk_timeout_checking_disconnection",
                            session_id=session_id,
                            request_id=request_id,
                            timeout_seconds=chunk_timeout,
                            message="No chunks received within timeout, checking for disconnection",
                        )

                        if await request.is_disconnected():
                            logger.warning(
                                "streaming_timeout_disconnection_detected",
                                session_id=session_id,
                                request_id=request_id,
                                message="Client disconnected during chunk timeout",
                            )

                            await self._interrupt_session(
                                claude_service,
                                session_id,
                                request_id,
                                "timeout_disconnection",
                            )
                            yield None  # Signal disconnection
                            return

                        # Continue waiting if still connected
                        continue

                    # Check which task completed first
                    if disconnection_check_task in done:
                        # Disconnection check task completed - this means disconnection was detected
                        # (because _periodic_disconnection_check only returns when disconnected)
                        logger.warning(
                            "streaming_periodic_disconnection_detected",
                            session_id=session_id,
                            request_id=request_id,
                            message="Client disconnection detected during periodic check",
                        )

                        await self._interrupt_session(
                            claude_service,
                            session_id,
                            request_id,
                            "periodic_disconnection",
                        )
                        yield None  # Signal disconnection
                        return

                    elif next_chunk_task in done:
                        # Got next chunk successfully
                        try:
                            chunk = await next_chunk_task
                            yield chunk
                        except StopAsyncIteration:
                            # End of stream reached normally
                            return

                except StopAsyncIteration:
                    # End of stream reached normally
                    return

        except Exception as e:
            logger.warning(
                "streaming_monitor_exception",
                session_id=session_id,
                request_id=request_id,
                exception_type=type(e).__name__,
                exception_message=str(e),
            )
            raise

    async def _periodic_disconnection_check(
        self,
        request: Request,
        session_id: str,
        request_id: str,
        check_interval: float,
    ) -> None:
        """Periodically check for client disconnection.

        Args:
            request: FastAPI request for disconnection detection
            session_id: Session ID for logging
            request_id: Request ID for logging
            check_interval: How often to check for disconnection (seconds)
        """
        while True:
            await asyncio.sleep(check_interval)

            if await request.is_disconnected():
                logger.debug(
                    "periodic_disconnection_check_detected",
                    session_id=session_id,
                    request_id=request_id,
                    check_interval=check_interval,
                )
                return  # This will complete the task and signal disconnection

    async def _get_next_chunk(self, content_iter: AsyncIterator[bytes]) -> bytes:
        """Get the next chunk from the content iterator.

        Args:
            content_iter: The async iterator for content chunks

        Returns:
            The next bytes chunk from the iterator

        Raises:
            StopAsyncIteration: When iterator is exhausted
        """
        return await anext(content_iter)
