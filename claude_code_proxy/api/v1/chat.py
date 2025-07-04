"""Chat completions API endpoint."""

import logging
import uuid
from collections.abc import AsyncGenerator
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from claude_code_proxy.config.settings import get_settings
from claude_code_proxy.exceptions import (
    ClaudeProxyError,
    ModelNotFoundError,
    ServiceUnavailableError,
    TimeoutError,
    ValidationError,
)
from claude_code_proxy.models.errors import create_error_response
from claude_code_proxy.models.requests import ChatCompletionRequest
from claude_code_proxy.models.responses import ChatCompletionResponse
from claude_code_proxy.services.claude_client import ClaudeClient
from claude_code_proxy.services.streaming import stream_claude_response


logger = logging.getLogger(__name__)
router = APIRouter()


# Supported Claude models
SUPPORTED_MODELS = {
    "claude-3-opus-20240229",
    "claude-3-sonnet-20240229",
    "claude-3-haiku-20240307",
    "claude-3-5-sonnet-20241022",
    "claude-3-5-haiku-20241022",
}


async def _validate_model(model: str) -> None:
    """Validate that the requested model is supported."""
    if model not in SUPPORTED_MODELS and not model.startswith("claude-"):
        raise ModelNotFoundError(model)


@router.post("/chat/completions", response_model=None)
async def create_chat_completion(
    request: ChatCompletionRequest,
    http_request: Request,
):
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

        # Validate model
        await _validate_model(request.model)

        # Initialize Claude client
        claude_client = ClaudeClient(
            claude_cli_path=settings.claude_cli_path,
        )

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
                        messages,
                        model=request.model,
                        max_tokens=request.max_tokens,
                        temperature=request.temperature,
                        system=request.system,
                        stream=True,
                    )

                    # Use enhanced streaming formatter
                    async for chunk in stream_claude_response(
                        response_iter,  # type: ignore
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
                messages,
                model=request.model,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                system=request.system,
                stream=False,
            )

            # Add message ID to response
            if isinstance(response, dict):
                response["id"] = message_id

            return ChatCompletionResponse(**response)  # type: ignore

    except ClaudeProxyError as e:
        logger.error(f"Claude proxy error: {e}")
        error_response, _ = create_error_response(e.error_type, e.message)
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
async def list_models() -> dict[str, Any]:
    """
    List available Claude models.

    Returns a list of available Claude models in Anthropic API format.
    """
    try:
        settings = get_settings()
        claude_client = ClaudeClient(
            claude_cli_path=settings.claude_cli_path,
        )
        models = await claude_client.list_models()
        return {"object": "list", "data": models}

    except ClaudeProxyError as e:
        logger.error(f"Claude proxy error in list_models: {e}")
        error_response, _ = create_error_response(e.error_type, e.message)
        raise HTTPException(status_code=e.status_code, detail=error_response) from e

    except Exception as e:
        logger.error(f"Unexpected error in list_models: {e}", exc_info=True)
        error_response, status_code = create_error_response(
            "internal_server_error", "Failed to retrieve models"
        )
        raise HTTPException(status_code=500, detail=error_response) from e
