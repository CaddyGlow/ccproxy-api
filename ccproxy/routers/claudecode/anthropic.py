"""Anthropic-compatible API endpoints."""

import uuid
from collections.abc import AsyncGenerator
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import ConfigDict, Field

from ccproxy.config.settings import get_settings
from ccproxy.exceptions import ClaudeProxyError
from ccproxy.formatters.anthropic_streaming import (
    stream_anthropic_message_response,
    stream_claude_response,
)
from ccproxy.middleware.auth import get_auth_dependency
from ccproxy.models.errors import create_error_response
from ccproxy.models.messages import MessageCreateParams, MessageResponse
from ccproxy.services.claude_client import ClaudeClient
from ccproxy.utils import merge_claude_code_options
from ccproxy.utils.helper import patched_typing
from ccproxy.utils.logging import get_logger


# Import ClaudeCodeOptions with patched typing
with patched_typing():
    from claude_code_sdk import ClaudeCodeOptions


class ClaudeCodeMessageCreateParams(MessageCreateParams):
    """Extended MessageCreateParams with ClaudeCodeOptions fields."""

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


@router.get("/models")
async def list_models(_: None = Depends(get_auth_dependency())) -> dict[str, Any]:
    """
    List available Claude models.

    Returns a list of available Claude models in Anthropic API format.
    """
    try:
        claude_client = ClaudeClient()
        models = await claude_client.list_models()
        return {"object": "list", "data": models}

    except ClaudeProxyError as e:
        logger.error(f"Claude proxy error in list_models: {e}")
        error_response, _status_code = create_error_response(e.error_type, e.message)
        raise HTTPException(status_code=e.status_code, detail=error_response) from e

    except Exception as e:
        logger.error(f"Unexpected error in list_models: {e}", exc_info=True)
        error_response, status_code = create_error_response(
            "internal_server_error", "Failed to retrieve models"
        )
        raise HTTPException(status_code=500, detail=error_response) from e


@router.post("/messages", response_model=None)
async def create_message(
    request: ClaudeCodeMessageCreateParams,
    http_request: Request,
    _: None = Depends(get_auth_dependency()),
) -> MessageResponse | StreamingResponse:
    """
    Create a message using Claude AI models.

    This endpoint provides the official Anthropic Messages API format,
    forwarding requests to Claude using the official SDK.

    Args:
        request: Message request matching Anthropic's Messages API format
        http_request: FastAPI request object for headers/metadata

    Returns:
        Message response or streaming response

    Raises:
        HTTPException: For validation errors, API errors, or service failures
    """
    try:
        settings = get_settings()

        # Create Claude client directly
        logger.info("[API] Creating Claude client for message request")
        claude_client = ClaudeClient()

        # Create a new ClaudeCodeOptions instance based on the request
        # Start with base options from settings
        options = merge_claude_code_options(settings.claude_code_options)

        # Override with request-specific values
        # Model from Anthropic API
        if request.model:
            options.model = request.model

        # Handle system message if provided
        if request.system:
            if isinstance(request.system, str):
                options.system_prompt = request.system
            elif isinstance(request.system, list):
                # Handle system message blocks by converting to string
                system_text = "\n".join([block.text for block in request.system])
                options.system_prompt = system_text

        # Anthropic API fields
        if request.temperature is not None:
            options.temperature = request.temperature
        if request.top_p is not None:
            options.top_p = request.top_p
        if request.top_k is not None:
            options.top_k = request.top_k
        if request.stop_sequences:
            options.stop_sequences = request.stop_sequences
        if request.tools:
            options.tools = request.tools
        if request.metadata:
            options.metadata = request.metadata
        if request.service_tier:
            options.service_tier = request.service_tier
        if request.thinking:
            # Convert ThinkingConfig to max_thinking_tokens for Claude Code SDK
            options.max_thinking_tokens = request.thinking.budget_tokens

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
                        messages, options=options, stream=True
                    )

                    # Ensure we have an async iterator for streaming
                    if not hasattr(response_iter, "__aiter__"):
                        logger.error(
                            f"Expected async iterator from Claude client, got {type(response_iter)}"
                        )
                        yield "data: {'error': {'type': 'internal_server_error', 'message': 'Invalid response type from Claude client'}}\n\n"
                        yield "data: [DONE]\n\n"
                        return

                    # Use enhanced streaming formatter for Messages API
                    async for chunk in stream_anthropic_message_response(
                        response_iter,
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
                messages=messages,
                options=options,
                stream=False,
            )

            # Convert to Anthropic Messages API format
            if isinstance(response, dict):
                # Extract the response content and create proper MessageResponse
                message_response = MessageResponse(
                    id=message_id,
                    type="message",
                    role="assistant",
                    content=response.get("content", []),
                    model=request.model,
                    stop_reason=response.get("stop_reason"),
                    stop_sequence=response.get("stop_sequence"),
                    usage=response.get("usage", {}),
                    container=response.get("container"),
                )
                return message_response
            else:
                # This should not happen in non-streaming mode, but handle it gracefully
                logger.error(
                    f"Unexpected response type from Claude client: {type(response)}"
                )
                raise ValueError(
                    f"Invalid response type: expected dict, got {type(response)}"
                )

    except ClaudeProxyError as e:
        logger.error(f"Claude proxy error: {e}")
        error_response, _status_code = create_error_response(e.error_type, e.message)
        raise HTTPException(status_code=e.status_code, detail=error_response) from e

    except ValueError as e:
        logger.error(f"Validation error: {e}")
        error_response, status_code = create_error_response(
            "invalid_request_error", str(e)
        )
        raise HTTPException(status_code=400, detail=error_response) from e

    except Exception as e:
        logger.error(f"Unexpected error in message creation: {e}", exc_info=True)
        error_response, status_code = create_error_response(
            "internal_server_error", "An unexpected error occurred"
        )
        raise HTTPException(status_code=500, detail=error_response) from e
