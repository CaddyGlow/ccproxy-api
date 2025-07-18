"""Claude SDK endpoints for Claude Code Proxy API Server."""

import json
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from ccproxy.api.dependencies import get_claude_service
from ccproxy.models.messages import MessageCreateParams, MessageResponse
from ccproxy.models.openai import (
    OpenAIChatCompletionRequest,
    OpenAIChatCompletionResponse,
    OpenAIModelInfo,
    OpenAIModelsResponse,
)
from ccproxy.services.claude_sdk_service import ClaudeSDKService


# Create the router for Claude SDK endpoints
router = APIRouter(tags=["claude-sdk"])


def _convert_openai_to_anthropic_messages(
    openai_messages: list[Any],
) -> list[dict[str, Any]]:
    """Convert OpenAI message format to Anthropic format.

    Args:
        openai_messages: List of OpenAI format messages (OpenAIMessage objects or dicts)

    Returns:
        List of Anthropic format messages
    """
    anthropic_messages = []

    for msg in openai_messages:
        # Handle both Pydantic models and dictionaries
        if hasattr(msg, "model_dump"):
            msg_dict = msg.model_dump()
        else:
            msg_dict = msg

        role = msg_dict.get("role", "user")
        content = msg_dict.get("content", "")

        # Map OpenAI roles to Anthropic roles
        if role == "system":
            # System messages in Anthropic are handled separately
            continue
        elif role == "assistant":
            anthropic_role = "assistant"
        else:  # user, function, tool
            anthropic_role = "user"

        anthropic_messages.append({"role": anthropic_role, "content": content})

    return anthropic_messages


def _convert_usage_to_openai(anthropic_usage: dict[str, Any]) -> dict[str, Any]:
    """Convert Anthropic usage format to OpenAI format.

    Args:
        anthropic_usage: Anthropic format usage data

    Returns:
        OpenAI format usage data
    """
    return {
        "prompt_tokens": anthropic_usage.get("input_tokens", 0),
        "completion_tokens": anthropic_usage.get("output_tokens", 0),
        "total_tokens": anthropic_usage.get("input_tokens", 0)
        + anthropic_usage.get("output_tokens", 0),
    }


def _convert_anthropic_to_openai_response(
    anthropic_response: dict[str, Any],
) -> dict[str, Any]:
    """Convert Anthropic response format to OpenAI format.

    Args:
        anthropic_response: Anthropic format response

    Returns:
        OpenAI format response
    """
    # Extract content from Anthropic response
    content = ""
    if "content" in anthropic_response:
        if isinstance(anthropic_response["content"], list):
            for block in anthropic_response["content"]:
                if block.get("type") == "text":
                    content += block.get("text", "")
        else:
            content = str(anthropic_response["content"])

    # Create OpenAI format response
    return {
        "id": anthropic_response.get("id", "chatcmpl-unknown"),
        "object": "chat.completion",
        "created": anthropic_response.get("created", 1234567890),
        "model": anthropic_response.get("model", "claude-3-sonnet-20240229"),
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": _convert_usage_to_openai(anthropic_response.get("usage", {})),
    }


@router.post("/v1/chat/completions", response_model=None)
async def create_openai_chat_completion(
    request: OpenAIChatCompletionRequest,
    claude_service: ClaudeSDKService = Depends(get_claude_service),
) -> StreamingResponse | OpenAIChatCompletionResponse:
    """Create a chat completion using Claude SDK with OpenAI-compatible format.

    This endpoint handles OpenAI API format requests and converts them
    to Anthropic format before using the Claude SDK directly.
    """
    try:
        # Convert OpenAI messages to Anthropic format
        anthropic_messages = _convert_openai_to_anthropic_messages(request.messages)

        # Extract parameters from OpenAI request
        model = request.model
        temperature = request.temperature
        max_tokens = request.max_tokens
        stream = request.stream or False

        # Call Claude SDK service
        response = await claude_service.create_completion(
            messages=anthropic_messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
            user_id=getattr(request, "user", None),
        )

        if stream:
            # Handle streaming response
            async def openai_stream_generator() -> AsyncIterator[bytes]:
                async for chunk in response:  # type: ignore[union-attr]
                    # Convert chunk to OpenAI format
                    openai_chunk = _convert_anthropic_to_openai_response(chunk)
                    yield f"data: {json.dumps(openai_chunk)}\n\n".encode()
                # Send final chunk
                yield b"data: [DONE]\n\n"

            return StreamingResponse(
                openai_stream_generator(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                },
            )
        else:
            # Convert non-streaming response to OpenAI format
            openai_response = _convert_anthropic_to_openai_response(response)  # type: ignore[arg-type]
            return OpenAIChatCompletionResponse.model_validate(openai_response)

    except Exception as e:
        # Re-raise specific proxy errors to be handled by the error handler
        from ccproxy.core.errors import ClaudeProxyError

        if isinstance(e, ClaudeProxyError):
            raise
        raise HTTPException(
            status_code=500, detail=f"Internal server error: {str(e)}"
        ) from e


@router.post("/v1/messages", response_model=None)
async def create_anthropic_message(
    request: MessageCreateParams,
    claude_service: ClaudeSDKService = Depends(get_claude_service),
) -> StreamingResponse | MessageResponse:
    """Create a message using Claude SDK with Anthropic format.

    This endpoint handles Anthropic API format requests directly
    using the Claude SDK without any format conversion.
    """
    try:
        # Extract parameters from Anthropic request
        messages = [msg.model_dump() for msg in request.messages]
        model = request.model
        temperature = request.temperature
        max_tokens = request.max_tokens
        stream = request.stream or False

        # Call Claude SDK service directly with Anthropic format
        response = await claude_service.create_completion(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
            user_id=getattr(request, "user_id", None),
        )

        if stream:
            # Handle streaming response
            async def anthropic_stream_generator() -> AsyncIterator[bytes]:
                async for chunk in response:  # type: ignore[union-attr]
                    if chunk:
                        yield f"data: {json.dumps(chunk)}\n\n".encode()
                # Send final chunk
                yield b"data: [DONE]\n\n"

            return StreamingResponse(
                anthropic_stream_generator(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                },
            )
        else:
            # Return Anthropic format response directly
            return MessageResponse.model_validate(response)

    except Exception as e:
        # Re-raise specific proxy errors to be handled by the error handler
        from ccproxy.core.errors import ClaudeProxyError

        if isinstance(e, ClaudeProxyError):
            raise
        raise HTTPException(
            status_code=500, detail=f"Internal server error: {str(e)}"
        ) from e


@router.get("/models", response_model=OpenAIModelsResponse)
async def list_sdk_models(
    claude_service: ClaudeSDKService = Depends(get_claude_service),
) -> OpenAIModelsResponse:
    """List available Claude models from SDK.

    Returns a list of available Claude models in OpenAI-compatible format.
    """
    try:
        models_data = await claude_service.list_models()

        # Convert to OpenAIModelInfo objects
        models = [OpenAIModelInfo.model_validate(model) for model in models_data]

        return OpenAIModelsResponse(
            object="list",
            data=models,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve models: {str(e)}"
        ) from e


@router.get("/status")
async def claude_sdk_status() -> dict[str, str]:
    """Get Claude SDK status."""
    return {"status": "claude sdk endpoint available", "service": "direct"}
