"""Routes for Claude SDK plugin."""

from typing import Annotated, Any, cast

from fastapi import APIRouter, Depends, Request
from starlette.responses import Response, StreamingResponse

from ccproxy.api.dependencies import get_plugin_adapter
from ccproxy.auth.conditional import ConditionalAuthDep
from ccproxy.streaming.deferred_streaming import DeferredStreaming


ClaudeSDKAdapterDep = Annotated[Any, Depends(get_plugin_adapter("claude_sdk"))]
router = APIRouter()

ResponseType = Response | StreamingResponse | DeferredStreaming


async def _handle_claude_sdk_request(
    request: Request,
    adapter: Any,
    endpoint: str,
) -> ResponseType:
    return cast(
        ResponseType,
        await adapter.handle_request(
            request=request,
            endpoint=endpoint,
            method=request.method,
        ),
    )


@router.post("/v1/messages", response_model=None)
async def claude_sdk_messages(
    request: Request,
    auth: ConditionalAuthDep,
    adapter: ClaudeSDKAdapterDep,
) -> ResponseType:
    return await _handle_claude_sdk_request(request, adapter, "/v1/messages")


@router.post("/v1/chat/completions", response_model=None)
async def claude_sdk_chat_completions(
    request: Request,
    auth: ConditionalAuthDep,
    adapter: ClaudeSDKAdapterDep,
) -> ResponseType:
    return await _handle_claude_sdk_request(request, adapter, "/v1/chat/completions")


@router.post("/{session_id}/v1/messages", response_model=None)
async def claude_sdk_messages_with_session(
    request: Request,
    session_id: str,
    auth: ConditionalAuthDep,
    adapter: ClaudeSDKAdapterDep,
) -> ResponseType:
    request.state.session_id = session_id
    return await _handle_claude_sdk_request(
        request, adapter, f"/{session_id}/v1/messages"
    )


@router.post("/{session_id}/v1/chat/completions", response_model=None)
async def claude_sdk_chat_completions_with_session(
    request: Request,
    session_id: str,
    auth: ConditionalAuthDep,
    adapter: ClaudeSDKAdapterDep,
) -> ResponseType:
    request.state.session_id = session_id
    return await _handle_claude_sdk_request(
        request, adapter, f"/{session_id}/v1/chat/completions"
    )
