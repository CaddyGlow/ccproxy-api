"""Utility functions for monitoring client disconnection and stuck streams during streaming responses."""

import asyncio
from typing import TYPE_CHECKING

import structlog
from starlette.requests import Request


if TYPE_CHECKING:
    from ccproxy.services.claude_sdk_service import ClaudeSDKService

logger = structlog.get_logger(__name__)


async def monitor_disconnection(
    request: Request, session_id: str, claude_service: "ClaudeSDKService"
) -> None:
    """Monitor for client disconnection and interrupt session if detected.

    Args:
        request: The incoming HTTP request
        session_id: The Claude SDK session ID to interrupt if disconnected
        claude_service: The Claude SDK service instance
    """
    try:
        while True:
            await asyncio.sleep(1.0)  # Check every second
            if await request.is_disconnected():
                logger.info(
                    "client_disconnected_interrupting_session", session_id=session_id
                )
                try:
                    await claude_service.sdk_client.interrupt_session(session_id)
                except Exception as e:
                    logger.error(
                        "failed_to_interrupt_session",
                        session_id=session_id,
                        error=str(e),
                    )
                return
    except asyncio.CancelledError:
        # Task was cancelled, which is expected when streaming completes normally
        logger.debug("disconnection_monitor_cancelled", session_id=session_id)
        raise


async def monitor_stuck_stream(
    session_id: str,
    claude_service: "ClaudeSDKService",
    first_chunk_event: asyncio.Event,
    timeout: float = 10.0,
) -> None:
    """Monitor for stuck streams that don't produce a first chunk (SystemMessage).

    Args:
        session_id: The Claude SDK session ID to monitor
        claude_service: The Claude SDK service instance
        first_chunk_event: Event that will be set when first chunk is received
        timeout: Seconds to wait for first chunk before considering stream stuck
    """
    try:
        # Wait for first chunk with timeout
        await asyncio.wait_for(first_chunk_event.wait(), timeout=timeout)
        logger.debug("stuck_stream_first_chunk_received", session_id=session_id)
    except TimeoutError:
        logger.error(
            "streaming_system_message_timeout",
            session_id=session_id,
            timeout=timeout,
            message=f"No SystemMessage received within {timeout}s, interrupting session",
        )
        try:
            await claude_service.sdk_client.interrupt_session(session_id)
            logger.info("stuck_session_interrupted_successfully", session_id=session_id)
        except Exception as e:
            logger.error(
                "failed_to_interrupt_stuck_session", session_id=session_id, error=str(e)
            )
    except asyncio.CancelledError:
        # Task was cancelled, which is expected when streaming completes normally
        logger.debug("stuck_stream_monitor_cancelled", session_id=session_id)
        raise
