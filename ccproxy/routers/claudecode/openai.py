"""OpenAI-compatible API endpoints."""

import json
import time
import uuid
from collections.abc import AsyncGenerator, AsyncIterator
from typing import Annotated, Any, cast

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import ConfigDict, Field
from starlette.status import HTTP_500_INTERNAL_SERVER_ERROR

from ccproxy.config.settings import get_settings
from ccproxy.exceptions import ClaudeProxyError
from ccproxy.formatters.openai_streaming_formatter import stream_claude_response_openai
from ccproxy.formatters.translator import OpenAITranslator
from ccproxy.middleware.auth import get_auth_dependency
from ccproxy.models.openai import (
    OpenAIChatCompletionRequest,
    OpenAIChatCompletionResponse,
    OpenAIErrorResponse,
    OpenAIModelsResponse,
)
from ccproxy.services.claude_client import ClaudeClient
from ccproxy.utils import merge_claude_code_options
from ccproxy.utils.helper import patched_typing
from ccproxy.utils.logging import get_logger


# Import ClaudeCodeOptions with patched typing
with patched_typing():
    from claude_code_sdk import ClaudeCodeOptions


class ClaudeCodeChatCompletionRequest(OpenAIChatCompletionRequest):
    """Extended OpenAIChatCompletionRequest with ClaudeCodeOptions fields."""

    # Add ClaudeCodeOptions fields explicitly to avoid conflicts
    max_thinking_tokens: Annotated[
        int | None, Field(None, description="Maximum number of thinking tokens")
    ] = None
    max_turns: Annotated[
        int | None, Field(None, description="Maximum number of turns")
    ] = None
    cwd: Annotated[str | None, Field(None, description="Current working directory")] = (
        None
    )
    system_prompt: Annotated[str | None, Field(None, description="System prompt")] = (
        None
    )
    append_system_prompt: Annotated[
        bool | None, Field(None, description="Whether to append system prompt")
    ] = None
    permission_mode: Annotated[
        str | None, Field(None, description="Permission mode")
    ] = None
    permission_prompt_tool_name: Annotated[
        str | None, Field(None, description="Permission prompt tool name")
    ] = None
    continue_conversation: Annotated[
        bool | None, Field(None, description="Whether to continue conversation")
    ] = None
    resume: Annotated[bool | None, Field(None, description="Whether to resume")] = None
    allowed_tools: Annotated[
        list[str] | None, Field(None, description="List of allowed tools")
    ] = None
    disallowed_tools: Annotated[
        list[str] | None, Field(None, description="List of disallowed tools")
    ] = None
    mcp_servers: Annotated[
        list[str] | None, Field(None, description="List of MCP servers")
    ] = None
    mcp_tools: Annotated[
        list[str] | None, Field(None, description="List of MCP tools")
    ] = None

    # Override model config to allow extra fields (for backwards compatibility)
    model_config = ConfigDict(extra="allow")


logger = get_logger(__name__)
router = APIRouter()


@router.post("/chat/completions", response_model=None)
async def create_chat_completion(
    request: ClaudeCodeChatCompletionRequest,
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

        # Create a new ClaudeCodeOptions instance based on the request
        # Start with base options from settings
        options = merge_claude_code_options(settings.claude_code_options)

        # Map OpenAI model to Claude model for the options
        from ccproxy.formatters.translator import map_openai_model_to_claude

        mapped_model = map_openai_model_to_claude(request.model)
        options.model = mapped_model

        # Claude Code specific fields (now available directly from request)
        if request.max_thinking_tokens is not None:
            options.max_thinking_tokens = request.max_thinking_tokens
        if request.max_turns is not None:
            options.max_turns = request.max_turns
        if request.cwd is not None:
            options.cwd = request.cwd
        if request.system_prompt is not None:
            options.system_prompt = request.system_prompt
        if request.append_system_prompt is not None:
            options.append_system_prompt = request.append_system_prompt
        if request.permission_mode is not None:
            options.permission_mode = request.permission_mode
        if request.permission_prompt_tool_name is not None:
            options.permission_prompt_tool_name = request.permission_prompt_tool_name
        if request.continue_conversation is not None:
            options.continue_conversation = request.continue_conversation
        if request.resume is not None:
            options.resume = request.resume
        if request.allowed_tools is not None:
            options.allowed_tools = request.allowed_tools
        if request.disallowed_tools is not None:
            options.disallowed_tools = request.disallowed_tools
        if request.mcp_servers is not None:
            options.mcp_servers = request.mcp_servers
        if request.mcp_tools is not None:
            options.mcp_tools = request.mcp_tools

        # Convert OpenAI request to Anthropic format
        # Only pass base OpenAI fields to avoid validation errors in translator
        base_request_dict = {}
        base_model = OpenAIChatCompletionRequest

        # Copy only fields that exist in the base model
        for field_name, _field_info in base_model.model_fields.items():
            if hasattr(request, field_name):
                value = getattr(request, field_name)
                if value is not None:  # Only include non-None values
                    base_request_dict[field_name] = value

        anthropic_request = translator.openai_to_anthropic_request(base_request_dict)

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

                    # Check if usage should be included based on stream_options
                    include_usage = bool(
                        request.stream_options and request.stream_options.include_usage
                    )

                    async for chunk in stream_claude_response_openai(
                        claude_stream_iter,  # type: ignore[arg-type]
                        request_id,
                        request.model,
                        created,
                        include_usage=include_usage,
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
