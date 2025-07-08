"""OpenAI-compatible API endpoints."""

import json
import time
import uuid
from collections.abc import AsyncGenerator, AsyncIterator
from typing import Any, cast

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from starlette.status import HTTP_500_INTERNAL_SERVER_ERROR

from ccproxy.config.settings import get_settings
from ccproxy.exceptions import ClaudeProxyError
from ccproxy.middleware.auth import get_auth_dependency
from ccproxy.models.openai import (
    OpenAIChatCompletionRequest,
    OpenAIChatCompletionResponse,
    OpenAIErrorResponse,
    OpenAIModelsResponse,
)
from ccproxy.services.claude_client import ClaudeClient
from ccproxy.services.openai_streaming import stream_claude_response_openai
from ccproxy.services.translator import OpenAITranslator
from ccproxy.utils import merge_claude_code_options
from ccproxy.utils.logging import get_logger


logger = get_logger(__name__)
router = APIRouter()


@router.post("/chat/completions", response_model=None)
async def create_chat_completion(
    request: OpenAIChatCompletionRequest,
    http_request: Request,
    _: None = Depends(get_auth_dependency()),
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
        # Get settings and check tools handling configuration
        settings = get_settings()

        # Check if tools are defined in the request
        if request.tools and settings.api_tools_handling != "ignore":
            tools_message = f"Tools definition detected with {len(request.tools)} tools"

            if settings.api_tools_handling == "error":
                logger.error(f"Tools not supported: {tools_message}")
                error_response = OpenAIErrorResponse.create(
                    message="Tools definitions are not supported by this proxy",
                    error_type="unsupported_parameter",
                )
                raise HTTPException(
                    status_code=400,
                    detail=error_response.model_dump(),
                )
            elif settings.api_tools_handling == "warning":
                logger.warning(f"Tools ignored: {tools_message}")

        # Create Claude client directly
        logger.info("[API] Creating Claude client for OpenAI chat completion request")
        claude_client = ClaudeClient()
        translator = OpenAITranslator()

        # Prepare Claude Code options overrides from request
        overrides: dict[str, Any] = {
            "model": request.model,
        }

        # Check for Claude Code specific options in the request
        if hasattr(request, "max_thinking_tokens") and request.max_thinking_tokens:
            overrides["max_thinking_tokens"] = request.max_thinking_tokens
        if hasattr(request, "allowed_tools") and request.allowed_tools:
            overrides["allowed_tools"] = request.allowed_tools
        if hasattr(request, "disallowed_tools") and request.disallowed_tools:
            overrides["disallowed_tools"] = request.disallowed_tools
        if hasattr(request, "append_system_prompt") and request.append_system_prompt:
            overrides["append_system_prompt"] = request.append_system_prompt
        if hasattr(request, "permission_mode") and request.permission_mode:
            overrides["permission_mode"] = request.permission_mode
        if (
            hasattr(request, "continue_conversation")
            and request.continue_conversation is not None
        ):
            overrides["continue_conversation"] = request.continue_conversation
        if hasattr(request, "resume") and request.resume:
            overrides["resume"] = request.resume
        if hasattr(request, "max_turns") and request.max_turns:
            overrides["max_turns"] = request.max_turns
        if (
            hasattr(request, "permission_prompt_tool_name")
            and request.permission_prompt_tool_name
        ):
            overrides["permission_prompt_tool_name"] = (
                request.permission_prompt_tool_name
            )
        if hasattr(request, "cwd") and request.cwd:
            overrides["cwd"] = request.cwd
        if hasattr(request, "mcp_tools") and request.mcp_tools:
            overrides["mcp_tools"] = request.mcp_tools
        if hasattr(request, "mcp_servers") and request.mcp_servers:
            overrides["mcp_servers"] = request.mcp_servers

        # Merge base options with request-specific overrides
        options = merge_claude_code_options(settings.claude_code_options, **overrides)

        # Convert OpenAI request to Anthropic format
        anthropic_request = translator.openai_to_anthropic_request(request.model_dump())

        # Generate request metadata
        request_id = f"chatcmpl-{uuid.uuid4().hex[:29]}"
        created = int(time.time())

        if request.stream:
            # Handle streaming response
            async def generate_stream() -> AsyncGenerator[str, None]:
                try:
                    logger.debug(
                        f"Starting OpenAI streaming for request_id: {request_id}"
                    )

                    # Get Claude streaming response
                    claude_stream = await claude_client.create_completion(
                        anthropic_request["messages"],
                        options,
                        stream=True,
                    )

                    logger.debug(
                        "Claude stream created, starting conversion to OpenAI format"
                    )

                    # Convert to OpenAI format
                    chunk_count = 0
                    # Type assertion: when stream=True, create_completion returns AsyncIterator
                    claude_stream_iter = cast(
                        AsyncIterator[dict[str, Any]], claude_stream
                    )

                    async for chunk in stream_claude_response_openai(
                        claude_stream_iter,  # type: ignore[arg-type]
                        request_id,
                        request.model,
                        created,
                    ):
                        chunk_count += 1
                        logger.debug(
                            f"Yielding OpenAI chunk {chunk_count}: {chunk[:200]}..."
                        )
                        yield chunk

                    logger.debug(
                        f"OpenAI streaming completed with {chunk_count} chunks"
                    )

                except Exception as e:
                    logger.error(f"Error in OpenAI streaming: {e}", exc_info=True)
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
                anthropic_request["messages"],
                options,
                stream=False,
            )

            # Convert to OpenAI format
            # Type assertion: when stream=False, create_completion returns dict[str, Any]
            response_dict = cast(dict[str, Any], anthropic_response)
            openai_response = translator.anthropic_to_openai_response(
                response_dict, request.model, request_id
            )

            return OpenAIChatCompletionResponse(**openai_response)

    except HTTPException:
        # Re-raise HTTPExceptions (like the tools validation error)
        raise
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


@router.get("/models", response_model=OpenAIModelsResponse)
async def list_models(_: None = Depends(get_auth_dependency())) -> OpenAIModelsResponse:
    """List available OpenAI-compatible models."""
    return OpenAIModelsResponse.create_default()
