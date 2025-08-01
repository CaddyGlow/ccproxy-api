"""Streaming response handler that triggers session interruption on client disconnect."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any, TypeVar

import structlog
from starlette.responses import StreamingResponse


if TYPE_CHECKING:
    from ccproxy.claude_sdk.client import ClaudeSDKClient

T = TypeVar("T")

logger = structlog.get_logger(__name__)


class StreamingResponseWithSessionInterrupt(StreamingResponse):
    """Streaming response that monitors for client disconnection and interrupts the session.

    This class bridges the gap between FastAPI's streaming response handling and
    our session management system. When a client disconnects, it automatically
    triggers session interruption to ensure streams are properly drained.
    """

    def __init__(
        self,
        content: AsyncIterator[Any],
        session_id: str,
        sdk_client: ClaudeSDKClient,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Initialize the streaming response with session interrupt capability.

        Args:
            content: The async iterator producing response content
            session_id: Session ID to interrupt on disconnection
            sdk_client: Claude SDK client for session management
            *args: Additional positional arguments for StreamingResponse
            **kwargs: Additional keyword arguments for StreamingResponse
        """
        self.session_id = session_id
        self.sdk_client = sdk_client
        self._interrupt_triggered = False

        # Wrap the content iterator to detect disconnection
        wrapped_content = self._wrap_content_with_disconnect_detection(content)

        super().__init__(wrapped_content, *args, **kwargs)

    async def _wrap_content_with_disconnect_detection(
        self, content: AsyncIterator[Any]
    ) -> AsyncIterator[Any]:
        """Wrap content iterator to detect and handle client disconnection.

        Args:
            content: Original content iterator

        Yields:
            Content from the original iterator
        """
        try:
            async for item in content:
                yield item
        except (asyncio.CancelledError, GeneratorExit) as e:
            # Client disconnected
            logger.warning(
                "streaming_session_client_disconnected",
                session_id=self.session_id,
                error_type=type(e).__name__,
                message="Client disconnected during streaming, triggering session interrupt",
            )

            # Trigger session interrupt if not already done
            if not self._interrupt_triggered:
                self._interrupt_triggered = True
                await self._interrupt_session()

            # Re-raise to maintain normal flow
            raise
        except Exception as e:
            # Other errors
            logger.error(
                "streaming_session_error",
                session_id=self.session_id,
                error=str(e),
                error_type=type(e).__name__,
            )
            raise
        finally:
            # Ensure interrupt is triggered even if iterator completes normally
            # but client disconnected during processing
            if not self._interrupt_triggered and self._is_disconnected():
                self._interrupt_triggered = True
                await self._interrupt_session()

    async def _interrupt_session(self) -> None:
        """Interrupt the session to trigger stream draining."""
        try:
            logger.info(
                "streaming_session_interrupting",
                session_id=self.session_id,
                message="Interrupting session due to client disconnection",
            )

            # First try to interrupt via stream handle if available
            # This is more direct and efficient than going through pool manager
            interrupted_via_handle = False
            if (
                hasattr(self.sdk_client, "_pool_manager")
                and self.sdk_client._pool_manager
                and hasattr(self.sdk_client._pool_manager, "has_session")
            ):
                try:
                    # Check if session exists without creating a new one
                    if await self.sdk_client._pool_manager.has_session(self.session_id):
                        # Get the session pool directly
                        session_pool = self.sdk_client._pool_manager._session_pool
                        if session_pool and self.session_id in session_pool.sessions:
                            session_client = session_pool.sessions[self.session_id]
                            if session_client and session_client.active_stream_handle:
                                logger.debug(
                                    "streaming_session_interrupting_via_handle",
                                    session_id=self.session_id,
                                    handle_id=session_client.active_stream_handle.handle_id,
                                )
                                await session_client.active_stream_handle.interrupt()
                                session_client.active_stream_handle = None
                                interrupted_via_handle = True
                except Exception as e:
                    logger.warning(
                        "streaming_session_handle_interrupt_failed",
                        session_id=self.session_id,
                        error=str(e),
                        message="Failed to interrupt via stream handle, falling back to session interrupt",
                    )

            # If not interrupted via handle, use regular session interrupt
            if not interrupted_via_handle:
                # Call the SDK client's interrupt_session method with timeout
                interrupted = await asyncio.wait_for(
                    self.sdk_client.interrupt_session(self.session_id),
                    timeout=5.0,  # 5 second timeout
                )

                logger.info(
                    "streaming_session_interrupted",
                    session_id=self.session_id,
                    interrupted=interrupted,
                    message="Session interrupt completed",
                )
            else:
                logger.info(
                    "streaming_session_interrupted_via_handle",
                    session_id=self.session_id,
                    message="Session interrupted via stream handle",
                )
        except TimeoutError:
            logger.error(
                "streaming_session_interrupt_timeout",
                session_id=self.session_id,
                message="Session interrupt timed out after 5 seconds",
            )
        except Exception as e:
            logger.error(
                "streaming_session_interrupt_error",
                session_id=self.session_id,
                error=str(e),
                error_type=type(e).__name__,
                message="Failed to interrupt session after disconnection",
            )

    def _is_disconnected(self) -> bool:
        """Check if the client has disconnected.

        Returns:
            True if client is disconnected, False otherwise
        """
        # In FastAPI/Starlette, we can check if the response has been sent
        # or if the connection is closed
        return hasattr(self, "_started") and self._started
