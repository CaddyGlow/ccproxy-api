"""Claude SDK endpoints for Claude Code Proxy API Server."""

import json
from collections.abc import AsyncIterator
from typing import Any

import structlog
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from ccproxy.adapters.openai.adapter import (
    OpenAIAdapter,
    OpenAIChatCompletionRequest,
    OpenAIChatCompletionResponse,
)
from ccproxy.api.dependencies import get_claude_service
from ccproxy.models.messages import MessageCreateParams, MessageResponse
from ccproxy.services.claude_sdk_service import ClaudeSDKService


# Create the router for Claude SDK endpoints
router = APIRouter(tags=["claude-sdk"])

logger = structlog.get_logger(__name__)


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
        # Create adapter instance
        adapter = OpenAIAdapter()

        # Convert entire OpenAI request to Anthropic format using adapter
        anthropic_request = adapter.adapt_request(request.model_dump())

        # Extract stream parameter
        stream = request.stream or False

        # Call Claude SDK service with adapted request
        response = await claude_service.create_completion(
            messages=anthropic_request["messages"],
            model=anthropic_request["model"],
            temperature=anthropic_request.get("temperature"),
            max_tokens=anthropic_request.get("max_tokens"),
            stream=stream,
            user_id=getattr(request, "user", None),
        )

        if stream:
            # Handle streaming response
            async def openai_stream_generator() -> AsyncIterator[bytes]:
                # Use adapt_stream for streaming responses
                async for openai_chunk in adapter.adapt_stream(response):  # type: ignore[arg-type]
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
            # Convert non-streaming response to OpenAI format using adapter
            openai_response = adapter.adapt_response(response)  # type: ignore[arg-type]
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


@router.get("/status")
async def claude_sdk_status() -> dict[str, str]:
    """Get Claude SDK status."""
    return {"status": "claude sdk endpoint available", "service": "direct"}
