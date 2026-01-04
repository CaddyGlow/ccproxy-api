"""Routes for Claude SDK plugin."""

from typing import Annotated, Any

from fastapi import APIRouter, Depends, Request
from starlette.responses import Response, StreamingResponse

from ccproxy.api.decorators import with_format_chain
from ccproxy.api.dependencies import get_plugin_adapter
from ccproxy.auth.dependencies import ConditionalAuthDep
from ccproxy.core.constants import (
    FORMAT_ANTHROPIC_MESSAGES,
    FORMAT_OPENAI_CHAT,
    FORMAT_OPENAI_RESPONSES,
)
from ccproxy.plugins.claude_sdk.adapter import ClaudeSDKAdapter
from ccproxy.streaming import DeferredStreaming


ClaudeSDKAdapterDep = Annotated[Any, Depends(get_plugin_adapter("claude_sdk"))]
router = APIRouter()

ResponseType = Response | StreamingResponse | DeferredStreaming


async def _handle_claude_sdk_request(
    request: Request,
    adapter: ClaudeSDKAdapter,
) -> ResponseType:
    return await adapter.handle_request(request)


@router.post("/v1/messages", response_model=None)
@with_format_chain([FORMAT_ANTHROPIC_MESSAGES])
async def claude_sdk_messages(
    request: Request,
    auth: ConditionalAuthDep,
    adapter: ClaudeSDKAdapterDep,
) -> ResponseType:
    return await _handle_claude_sdk_request(request, adapter)


@router.post("/v1/chat/completions", response_model=None)
@with_format_chain(
    [
        FORMAT_OPENAI_CHAT,
        FORMAT_ANTHROPIC_MESSAGES,
    ]
)
async def claude_sdk_chat_completions(
    request: Request,
    auth: ConditionalAuthDep,
    adapter: ClaudeSDKAdapterDep,
) -> ResponseType:
    return await _handle_claude_sdk_request(request, adapter)


@router.post("/v1/responses", response_model=None)
@with_format_chain([FORMAT_OPENAI_RESPONSES, FORMAT_ANTHROPIC_MESSAGES])
async def claude_sdk_responses(
    request: Request,
    auth: ConditionalAuthDep,
    adapter: ClaudeSDKAdapterDep,
) -> ResponseType:
    return await _handle_claude_sdk_request(request, adapter)


@router.post("/{session_id}/v1/messages", response_model=None)
@with_format_chain([FORMAT_ANTHROPIC_MESSAGES])
async def claude_sdk_messages_with_session(
    request: Request,
    session_id: str,
    auth: ConditionalAuthDep,
    adapter: ClaudeSDKAdapterDep,
) -> ResponseType:
    request.state.session_id = session_id
    return await _handle_claude_sdk_request(request, adapter)


@router.post("/{session_id}/v1/chat/completions", response_model=None)
@with_format_chain(
    [
        FORMAT_OPENAI_CHAT,
        FORMAT_ANTHROPIC_MESSAGES,
    ]
)
async def claude_sdk_chat_completions_with_session(
    request: Request,
    session_id: str,
    auth: ConditionalAuthDep,
    adapter: ClaudeSDKAdapterDep,
) -> ResponseType:
    request.state.session_id = session_id
    return await _handle_claude_sdk_request(request, adapter)


@router.post("/{session_id}/v1/responses", response_model=None)
@with_format_chain([FORMAT_OPENAI_RESPONSES, FORMAT_ANTHROPIC_MESSAGES])
async def claude_sdk_responses_with_session(
    request: Request,
    auth: ConditionalAuthDep,
    adapter: ClaudeSDKAdapterDep,
) -> ResponseType:
    return await _handle_claude_sdk_request(request, adapter)
