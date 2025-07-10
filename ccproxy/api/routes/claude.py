"""Claude SDK endpoints for Claude Code Proxy API Server."""

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from ccproxy.api.dependencies import get_claude_service
from ccproxy.core.logging import get_logger
from ccproxy.models.openai import (
    OpenAIChatCompletionRequest,
    OpenAIChatCompletionResponse,
)
from ccproxy.models.requests import MessageCreateParams
from ccproxy.models.responses import ChatCompletionResponse
from ccproxy.services.claude_sdk_service import ClaudeSDKService


router = APIRouter()
logger = get_logger(__name__)


@router.post("/messages", response_model=ChatCompletionResponse)
async def create_message(
    request: MessageCreateParams,
    claude_service: ClaudeSDKService = Depends(get_claude_service),
) -> ChatCompletionResponse | StreamingResponse:
    """Create a message using the Claude SDK.

    Args:
        request: Message creation parameters
        claude_service: Injected Claude SDK service dependency

    Returns:
        Chat completion response or streaming response

    Raises:
        HTTPException: If Claude SDK request fails
    """
    try:
        logger.info(
            f"Claude SDK message request: model={request.model}, max_tokens={request.max_tokens}"
        )

        if request.stream:
            # Return streaming response
            return StreamingResponse(
                claude_service.stream_message(request),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "Content-Type": "text/event-stream",
                },
            )
        else:
            # Return regular response
            response = await claude_service.create_message(request)
            return response

    except Exception as e:
        logger.error(f"Claude SDK message request failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/chat/completions", response_model=OpenAIChatCompletionResponse)
async def create_chat_completion(
    request: OpenAIChatCompletionRequest,
    claude_service: ClaudeSDKService = Depends(get_claude_service),
) -> OpenAIChatCompletionResponse | StreamingResponse:
    """Create a chat completion using the Claude SDK (OpenAI format).

    Args:
        request: OpenAI chat completion request
        claude_service: Injected Claude SDK service dependency

    Returns:
        OpenAI chat completion response or streaming response

    Raises:
        HTTPException: If Claude SDK request fails
    """
    try:
        logger.info(
            f"Claude SDK OpenAI chat completion: model={request.model}, max_tokens={request.max_tokens}"
        )

        if request.stream:
            # Return streaming response
            return StreamingResponse(
                claude_service.stream_chat_completion(request),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "Content-Type": "text/event-stream",
                },
            )
        else:
            # Return regular response
            response = await claude_service.create_chat_completion(request)
            return response

    except Exception as e:
        logger.error(f"Claude SDK OpenAI chat completion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/models")
async def list_models(
    claude_service: ClaudeSDKService = Depends(get_claude_service),
) -> dict[str, Any]:
    """List available models from the Claude SDK.

    Args:
        claude_service: Injected Claude SDK service dependency

    Returns:
        List of available models

    Raises:
        HTTPException: If Claude SDK request fails
    """
    try:
        logger.info("Claude SDK list models request")
        models = await claude_service.list_models()
        return models

    except Exception as e:
        logger.error(f"Claude SDK list models failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/models/{model_id}")
async def get_model(
    model_id: str,
    claude_service: ClaudeSDKService = Depends(get_claude_service),
) -> dict[str, Any]:
    """Get model details from the Claude SDK.

    Args:
        model_id: Model identifier
        claude_service: Injected Claude SDK service dependency

    Returns:
        Model details

    Raises:
        HTTPException: If Claude SDK request fails
    """
    try:
        logger.info(f"Claude SDK get model request: {model_id}")
        model = await claude_service.get_model(model_id)
        return model

    except Exception as e:
        logger.error(f"Claude SDK get model failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/status")
async def get_claude_status(
    claude_service: ClaudeSDKService = Depends(get_claude_service),
) -> dict[str, Any]:
    """Get Claude SDK status and health information.

    Args:
        claude_service: Injected Claude SDK service dependency

    Returns:
        Claude SDK status information

    Raises:
        HTTPException: If Claude SDK request fails
    """
    try:
        logger.info("Claude SDK status request")
        status = await claude_service.get_status()
        return status

    except Exception as e:
        logger.error(f"Claude SDK status request failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e
