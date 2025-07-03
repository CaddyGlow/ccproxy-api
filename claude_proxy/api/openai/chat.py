"""OpenAI-compatible chat completions endpoint."""

import json
import logging
import time
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from starlette.status import HTTP_500_INTERNAL_SERVER_ERROR

from claude_proxy.exceptions import ClaudeProxyError
from claude_proxy.models.openai_models import (
    OpenAIChatCompletionRequest,
    OpenAIChatCompletionResponse,
    OpenAIErrorResponse,
)
from claude_proxy.services.claude_client import ClaudeClient
from claude_proxy.services.openai_streaming import stream_claude_response_openai
from claude_proxy.services.translator import OpenAITranslator


logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/chat/completions", response_model=None)
async def create_chat_completion(
    request: OpenAIChatCompletionRequest,
    http_request: Request,
) -> OpenAIChatCompletionResponse | StreamingResponse:
    """
    Create a chat completion using OpenAI-compatible format.

    Args:
        request: OpenAI-compatible chat completion request
        http_request: FastAPI request object

    Returns:
        OpenAI-compatible chat completion response or streaming response
    """
    try:
        # Initialize services
        claude_client = ClaudeClient()
        translator = OpenAITranslator()

        # Convert OpenAI request to Anthropic format
        anthropic_request = translator.openai_to_anthropic_request(request.model_dump())

        # Generate request metadata
        request_id = f"chatcmpl-{uuid.uuid4().hex[:29]}"
        created = int(time.time())

        if request.stream:
            # Handle streaming response
            async def generate_stream():
                try:
                    # Get Claude streaming response
                    claude_stream = await claude_client.create_completion(
                        messages=anthropic_request["messages"],
                        model=anthropic_request["model"],
                        max_tokens=anthropic_request["max_tokens"],
                        temperature=anthropic_request.get("temperature"),
                        system=anthropic_request.get("system"),
                        stream=True,
                    )

                    # Convert to OpenAI format
                    async for chunk in stream_claude_response_openai(
                        claude_stream,
                        request_id,
                        request.model,
                        created,
                    ):
                        yield chunk

                except Exception as e:
                    logger.error(f"Error in OpenAI streaming: {e}")
                    error_chunk = {
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": request.model,
                        "choices": [
                            {"index": 0, "delta": {}, "finish_reason": "error"}
                        ],
                        "error": {"type": "internal_server_error", "message": str(e)},
                    }
                    yield f"data: {json.dumps(error_chunk)}\n\n"
                    yield "data: [DONE]\n\n"

            return StreamingResponse(
                generate_stream(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )
        else:
            # Handle non-streaming response
            anthropic_response = await claude_client.create_completion(
                messages=anthropic_request["messages"],
                model=anthropic_request["model"],
                max_tokens=anthropic_request["max_tokens"],
                temperature=anthropic_request.get("temperature"),
                system=anthropic_request.get("system"),
                stream=False,
            )

            # Convert to OpenAI format
            openai_response = translator.anthropic_to_openai_response(
                anthropic_response, request.model, request_id
            )

            return OpenAIChatCompletionResponse(**openai_response)

    except ClaudeProxyError as e:
        logger.error(f"Claude client error: {e}")
        error_response = OpenAIErrorResponse.create(
            message=str(e), error_type="api_error"
        )
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_response.model_dump(),
        ) from e
    except Exception as e:
        logger.error(f"Unexpected error in OpenAI chat completion: {e}")
        error_response = OpenAIErrorResponse.create(
            message="An unexpected error occurred", error_type="internal_server_error"
        )
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_response.model_dump(),
        ) from e
