"""Chat completions API endpoint."""

import logging
from collections.abc import AsyncGenerator
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from claude_proxy.config.settings import get_settings
from claude_proxy.models.requests import ChatCompletionRequest
from claude_proxy.models.responses import ChatCompletionResponse
from claude_proxy.services.claude_client import ClaudeClient


logger = logging.getLogger(__name__)
router = APIRouter()


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

        # Initialize Claude client
        claude_client = ClaudeClient(
            api_key=settings.anthropic_api_key,
            claude_cli_path=settings.claude_cli_path,
        )

        # Convert request to messages format
        messages = [msg.model_dump() for msg in request.messages]

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
                    async for chunk in response_iter:  # type: ignore
                        yield f"data: {chunk}\n\n"
                    yield "data: [DONE]\n\n"
                except Exception as e:
                    logger.error(f"Error in streaming response: {e}")
                    yield f"data: {{'error': '{str(e)}'}}\n\n"

            return StreamingResponse(
                generate_stream(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "Content-Type": "text/event-stream",
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
            return ChatCompletionResponse(**response)  # type: ignore

    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "type": "invalid_request_error",
                    "message": str(e),
                }
            },
        ) from e
    except Exception as e:
        logger.error(f"Unexpected error in chat completion: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "type": "internal_server_error",
                    "message": "An unexpected error occurred",
                }
            },
        ) from e


@router.get("/models")
async def list_models() -> dict[str, Any]:
    """
    List available Claude models.

    Returns a list of available Claude models in Anthropic API format.
    """
    settings = get_settings()
    claude_client = ClaudeClient(
        claude_cli_path=settings.claude_cli_path,
    )
    models = await claude_client.list_models()
    return {"object": "list", "data": models}
