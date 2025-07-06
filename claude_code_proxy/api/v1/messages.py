"""Anthropic Messages API endpoint."""

import logging
import uuid
from collections.abc import AsyncGenerator
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from claude_code_proxy.config.settings import get_settings
from claude_code_proxy.exceptions import (
    ClaudeProxyError,
    ModelNotFoundError,
    ServiceUnavailableError,
    TimeoutError,
    ValidationError,
)
from claude_code_proxy.middleware.auth import get_auth_dependency
from claude_code_proxy.models.errors import create_error_response
from claude_code_proxy.models.messages import MessageRequest, MessageResponse
from claude_code_proxy.services.claude_client import ClaudeClient
from claude_code_proxy.services.pool_manager import pool_manager
from claude_code_proxy.services.streaming import stream_anthropic_message_response
from claude_code_proxy.utils import merge_claude_code_options


logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/messages", response_model=None)
async def create_message(
    request: MessageRequest,
    http_request: Request,
    _: None = Depends(get_auth_dependency()),
) -> MessageResponse | StreamingResponse:
    """
    Create a message using Claude AI models.

    This endpoint provides the official Anthropic Messages API format,
    forwarding requests to Claude using the official SDK.

    Args:
        request: Message request matching Anthropic's Messages API format
        http_request: FastAPI request object for headers/metadata

    Returns:
        Message response or streaming response

    Raises:
        HTTPException: For validation errors, API errors, or service failures
    """
    pooled_connection = None
    try:
        settings = get_settings()

        # Get Claude client from pool
        logger.info("[API] Acquiring Claude client from pool for message request")
        claude_client, pooled_connection = await pool_manager.acquire_client()

        # Prepare Claude Code options overrides from request
        overrides: dict[str, Any] = {
            "model": request.model,
        }

        if request.max_thinking_tokens:
            overrides["max_thinking_tokens"] = request.max_thinking_tokens

        # Add system message if provided - handle through system_prompt instead
        if request.system:
            if isinstance(request.system, str):
                overrides["system_prompt"] = request.system
            elif isinstance(request.system, list):
                # Handle system message blocks by converting to string
                system_text = "\n".join([block.text for block in request.system])
                overrides["system_prompt"] = system_text

        # Merge base options with request-specific overrides
        options = merge_claude_code_options(settings.claude_code_options, **overrides)

        # Convert request to messages format
        messages = [msg.model_dump() for msg in request.messages]

        # Generate unique message ID
        message_id = f"msg_{uuid.uuid4().hex[:12]}"

        # Handle streaming vs non-streaming responses
        if request.stream:
            # Return streaming response
            async def generate_stream() -> AsyncGenerator[str, None]:
                try:
                    response_iter = await claude_client.create_completion(
                        messages, options=options, stream=True
                    )

                    # Ensure we have an async iterator for streaming
                    if not hasattr(response_iter, "__aiter__"):
                        logger.error(
                            f"Expected async iterator from Claude client, got {type(response_iter)}"
                        )
                        yield "data: {'error': {'type': 'internal_server_error', 'message': 'Invalid response type from Claude client'}}\n\n"
                        yield "data: [DONE]\n\n"
                        return

                    # Use enhanced streaming formatter for Messages API
                    async for chunk in stream_anthropic_message_response(
                        response_iter,
                        message_id,
                        request.model,
                    ):
                        yield chunk

                except ClaudeProxyError as e:
                    logger.error(f"Claude proxy error in streaming: {e}")
                    error_response, _ = create_error_response(e.error_type, e.message)
                    yield f"data: {error_response}\n\n"
                except Exception as e:
                    logger.error(f"Unexpected error in streaming: {e}", exc_info=True)
                    error_response, _ = create_error_response(
                        "internal_server_error", "An unexpected error occurred"
                    )
                    yield f"data: {error_response}\n\n"

            return StreamingResponse(
                generate_stream(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "Content-Type": "text/event-stream",
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Headers": "*",
                },
            )
        else:
            # Return regular response
            response = await claude_client.create_completion(
                messages=messages,
                options=options,
                stream=False,
            )

            # Convert to Anthropic Messages API format
            if isinstance(response, dict):
                # Extract the response content and create proper MessageResponse
                message_response = MessageResponse(
                    id=message_id,
                    type="message",
                    role="assistant",
                    content=response.get("content", []),
                    model=request.model,
                    stop_reason=response.get("stop_reason"),
                    stop_sequence=response.get("stop_sequence"),
                    usage=response.get("usage", {}),
                )
                return message_response

            return MessageResponse(**response)

    except ClaudeProxyError as e:
        logger.error(f"Claude proxy error: {e}")
        error_response, _status_code = create_error_response(e.error_type, e.message)
        raise HTTPException(status_code=e.status_code, detail=error_response) from e

    except ValueError as e:
        logger.error(f"Validation error: {e}")
        error_response, status_code = create_error_response(
            "invalid_request_error", str(e)
        )
        raise HTTPException(status_code=400, detail=error_response) from e

    except Exception as e:
        logger.error(f"Unexpected error in message creation: {e}", exc_info=True)
        error_response, status_code = create_error_response(
            "internal_server_error", "An unexpected error occurred"
        )
        raise HTTPException(status_code=500, detail=error_response) from e
    finally:
        # Release connection back to pool
        if pooled_connection:
            logger.info(
                f"[API] Releasing Claude client connection {pooled_connection.id} back to pool"
            )
            await pool_manager.release_client(pooled_connection)
