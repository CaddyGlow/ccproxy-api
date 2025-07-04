"""Chat completion API routes."""

import logging
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from starlette.status import (
    HTTP_422_UNPROCESSABLE_ENTITY,
    HTTP_500_INTERNAL_SERVER_ERROR,
)

from claude_code_proxy.config.settings import Settings, get_settings
from claude_code_proxy.models.requests import ChatCompletionRequest
from claude_code_proxy.models.responses import (
    APIError,
    ChatCompletionResponse,
    ErrorResponse,
    InternalServerError,
    InvalidRequestError,
    RateLimitError,
    StreamingChatCompletionResponse,
)
from claude_code_proxy.services.claude_client import ClaudeClient, ClaudeClientError


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["chat"])


def get_claude_client(settings: Settings = Depends(get_settings)) -> ClaudeClient:
    """Get Claude client instance."""
    return ClaudeClient()


@router.post("/chat/completions", response_model=None)
async def create_chat_completion(
    request: ChatCompletionRequest,
    http_request: Request,
    settings: Settings = Depends(get_settings),
    claude_client: ClaudeClient = Depends(get_claude_client),
) -> ChatCompletionResponse | StreamingResponse:
    """
    Create a chat completion using Claude AI.

    This endpoint is compatible with the Anthropic API format and supports
    both streaming and non-streaming responses.

    Args:
        request: The chat completion request
        http_request: The FastAPI request object
        settings: Application settings
        claude_client: Claude client instance

    Returns:
        Chat completion response or streaming response

    Raises:
        HTTPException: For various API errors
    """
    try:
        logger.info(
            f"Chat completion request: model={request.model}, stream={request.stream}"
        )

        # Convert request to the format expected by Claude client
        messages = []
        for message in request.messages:
            msg_dict: dict[str, Any] = {"role": message.role, "content": []}

            # Convert message content to proper format
            if isinstance(message.content, str):
                msg_dict["content"] = [{"type": "text", "text": message.content}]
            elif isinstance(message.content, list):
                content_list = msg_dict["content"]
                assert isinstance(content_list, list)  # Type narrowing for mypy
                for content in message.content:
                    if hasattr(content, "type") and content.type == "text":
                        if hasattr(content, "text"):
                            content_list.append({"type": "text", "text": content.text})
                    elif (
                        hasattr(content, "type")
                        and content.type == "image"
                        and hasattr(content, "source")
                    ):
                        content_list.append(
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": content.source.media_type,
                                    "data": content.source.data,
                                },
                            }
                        )

            messages.append(msg_dict)

        # Create completion
        options = settings.claude_code_options
        if request.model:
            options.model = request.model
        if request.system:
            options.system_prompt = request.system

        response = await claude_client.create_completion(
            messages=messages,
            options=options,
            stream=request.stream or False,
        )

        if request.stream:
            # Return streaming response
            async def generate_stream() -> AsyncIterator[bytes]:
                async for chunk in response:  # type: ignore
                    chunk_data = StreamingChatCompletionResponse(**chunk)
                    yield f"data: {chunk_data.model_dump_json()}\\n\\n".encode()
                yield b"data: [DONE]\\n\\n"

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
            # Return non-streaming response
            return ChatCompletionResponse(**response)  # type: ignore

    except ClaudeClientError as e:
        logger.error(f"Claude client error: {e}")
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail=InternalServerError(
                type="internal_server_error",
                message=str(e),
            ).model_dump(),
        ) from e

    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(
            status_code=HTTP_422_UNPROCESSABLE_ENTITY,
            detail=InvalidRequestError(
                type="invalid_request_error",
                message=str(e),
            ).model_dump(),
        ) from e

    except Exception as e:
        logger.error(f"Unexpected error in chat completion: {e}")
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail=InternalServerError(
                type="internal_server_error",
                message="An unexpected error occurred",
            ).model_dump(),
        ) from e


@router.get("/models", response_model=None)
async def list_models(
    claude_client: ClaudeClient = Depends(get_claude_client),
) -> dict[str, list[dict[str, Any]]]:
    """
    List available Claude models.

    Returns:
        Dictionary containing list of available models

    Raises:
        HTTPException: For API errors
    """
    try:
        models = await claude_client.list_models()
        return {"data": models}

    except ClaudeClientError as e:
        logger.error(f"Error listing models: {e}")
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail=InternalServerError(
                type="internal_server_error",
                message=str(e),
            ).model_dump(),
        ) from e

    except Exception as e:
        logger.error(f"Unexpected error listing models: {e}")
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail=InternalServerError(
                type="internal_server_error",
                message="An unexpected error occurred",
            ).model_dump(),
        ) from e
