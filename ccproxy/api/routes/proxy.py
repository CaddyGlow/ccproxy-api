"""Proxy endpoints for Claude Code Proxy API Server."""

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import StreamingResponse

from ccproxy.api.dependencies import get_proxy_service
from ccproxy.core.logging import get_logger
from ccproxy.models.openai import (
    OpenAIChatCompletionRequest,
    OpenAIChatCompletionResponse,
)
from ccproxy.models.requests import MessageCreateParams
from ccproxy.models.responses import (
    ChatCompletionResponse,
    StreamingChatCompletionResponse,
)
from ccproxy.services.proxy_service import ProxyService


router = APIRouter()
logger = get_logger(__name__)


@router.post("/v1/messages", response_model=ChatCompletionResponse)
async def create_message_proxy(
    request: MessageCreateParams,
    proxy_service: ProxyService = Depends(get_proxy_service),
) -> ChatCompletionResponse | StreamingResponse:
    """Create a message through the proxy service.

    Args:
        request: Message creation parameters
        proxy_service: Injected proxy service dependency

    Returns:
        Chat completion response or streaming response

    Raises:
        HTTPException: If proxy request fails
    """
    try:
        logger.info(
            f"Proxy message request: model={request.model}, max_tokens={request.max_tokens}"
        )

        if request.stream:
            # Return streaming response
            return StreamingResponse(
                proxy_service.stream_message(request),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "Content-Type": "text/event-stream",
                },
            )
        else:
            # Return regular response
            response = await proxy_service.create_message(request)
            return response

    except Exception as e:
        logger.error(f"Proxy message request failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/openai/v1/chat/completions", response_model=OpenAIChatCompletionResponse)
async def create_chat_completion_proxy(
    request: OpenAIChatCompletionRequest,
    proxy_service: ProxyService = Depends(get_proxy_service),
) -> OpenAIChatCompletionResponse | StreamingResponse:
    """Create a chat completion through the proxy service (OpenAI format).

    Args:
        request: OpenAI chat completion request
        proxy_service: Injected proxy service dependency

    Returns:
        OpenAI chat completion response or streaming response

    Raises:
        HTTPException: If proxy request fails
    """
    try:
        logger.info(
            f"Proxy OpenAI chat completion: model={request.model}, max_tokens={request.max_tokens}"
        )

        if request.stream:
            # Return streaming response
            return StreamingResponse(
                proxy_service.stream_chat_completion(request),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "Content-Type": "text/event-stream",
                },
            )
        else:
            # Return regular response
            response = await proxy_service.create_chat_completion(request)
            return response

    except Exception as e:
        logger.error(f"Proxy OpenAI chat completion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/v1/models")
async def list_models_proxy(
    proxy_service: ProxyService = Depends(get_proxy_service),
) -> dict[str, Any]:
    """List available models through the proxy service.

    Args:
        proxy_service: Injected proxy service dependency

    Returns:
        List of available models

    Raises:
        HTTPException: If proxy request fails
    """
    try:
        logger.info("Proxy list models request")
        models = await proxy_service.list_models()
        return models

    except Exception as e:
        logger.error(f"Proxy list models failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/openai/v1/models")
async def list_models_openai_proxy(
    proxy_service: ProxyService = Depends(get_proxy_service),
) -> dict[str, Any]:
    """List available models through the proxy service (OpenAI format).

    Args:
        proxy_service: Injected proxy service dependency

    Returns:
        List of available models in OpenAI format

    Raises:
        HTTPException: If proxy request fails
    """
    try:
        logger.info("Proxy list models request (OpenAI format)")
        models = await proxy_service.list_models_openai()
        return models

    except Exception as e:
        logger.error(f"Proxy list models (OpenAI format) failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.api_route(
    "/proxy/{path:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"],
)
async def proxy_passthrough(
    request: Request,
    path: str,
    proxy_service: ProxyService = Depends(get_proxy_service),
) -> Response:
    """Generic proxy passthrough for all other endpoints.

    Args:
        request: Raw HTTP request
        path: Request path
        proxy_service: Injected proxy service dependency

    Returns:
        Raw HTTP response

    Raises:
        HTTPException: If proxy request fails
    """
    try:
        logger.info(f"Proxy passthrough: {request.method} /{path}")
        response = await proxy_service.proxy_request(request, path)
        return response

    except Exception as e:
        logger.error(f"Proxy passthrough failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e
