"""Chat completions API endpoint."""

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
from claude_code_proxy.models.requests import ChatCompletionRequest
from claude_code_proxy.models.responses import ChatCompletionResponse
from claude_code_proxy.services.claude_client import ClaudeClient
from claude_code_proxy.services.streaming import stream_claude_response
from claude_code_proxy.utils import merge_claude_code_options


logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/chat/completions", response_model=None)
async def create_chat_completion(
    request: ChatCompletionRequest,
    http_request: Request,
    _: None = Depends(get_auth_dependency()),
) -> ChatCompletionResponse | StreamingResponse:
    """
    Create a chat completion using Claude AI models.

    This endpoint provides Anthropic API-compatible chat completions,
    forwarding requests to Claude using the official SDK.

    Args:
        request: Chat completion request matching Anthropic's API format
        http_request: FastAPI request object for headers/metadata

    Returns:
        Chat completion response or streaming response

    Raises:
        HTTPException: For validation errors, API errors, or service failures
    """
    try:
        settings = get_settings()

        # Initialize Claude client
        claude_client = ClaudeClient()

        # Prepare Claude Code options overrides from request
        overrides: dict[str, Any] = {
            "model": request.model,
        }

        if request.max_thinking_tokens:
            overrides["max_thinking_tokens"] = request.max_thinking_tokens

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

                    # Use enhanced streaming formatter
                    async for chunk in stream_claude_response(
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

            # Add message ID to response
            if isinstance(response, dict):
                response["id"] = message_id

            return ChatCompletionResponse(**response)  # type: ignore

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
        logger.error(f"Unexpected error in chat completion: {e}", exc_info=True)
        error_response, status_code = create_error_response(
            "internal_server_error", "An unexpected error occurred"
        )
        raise HTTPException(status_code=500, detail=error_response) from e


@router.get("/models")
async def list_models(_: None = Depends(get_auth_dependency())) -> dict[str, Any]:
    """
    List available Claude models.

    Returns a list of available Claude models in Anthropic API format.
    """
    try:
        claude_client = ClaudeClient()
        models = await claude_client.list_models()
        return {"object": "list", "data": models}

    except ClaudeProxyError as e:
        logger.error(f"Claude proxy error in list_models: {e}")
        error_response, _status_code = create_error_response(e.error_type, e.message)
        raise HTTPException(status_code=e.status_code, detail=error_response) from e

    except Exception as e:
        logger.error(f"Unexpected error in list_models: {e}", exc_info=True)
        error_response, status_code = create_error_response(
            "internal_server_error", "Failed to retrieve models"
        )
        raise HTTPException(status_code=500, detail=error_response) from e
