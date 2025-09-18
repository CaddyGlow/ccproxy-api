"""Codex plugin routes."""

from typing import TYPE_CHECKING, Annotated, Any, cast

from fastapi import APIRouter, Depends, Request
from starlette.responses import Response, StreamingResponse

from ccproxy.api.decorators import with_format_chain
from ccproxy.api.dependencies import get_plugin_adapter
from ccproxy.auth.conditional import ConditionalAuthDep
from ccproxy.core.constants import (
    FORMAT_ANTHROPIC_MESSAGES,
    FORMAT_OPENAI_CHAT,
    FORMAT_OPENAI_RESPONSES,
    UPSTREAM_ENDPOINT_ANTHROPIC_MESSAGES,
    UPSTREAM_ENDPOINT_OPENAI_CHAT_COMPLETIONS,
    UPSTREAM_ENDPOINT_OPENAI_RESPONSES,
)
from ccproxy.streaming import DeferredStreaming


if TYPE_CHECKING:
    pass

CodexAdapterDep = Annotated[Any, Depends(get_plugin_adapter("codex"))]
router = APIRouter()


# Helper to handle adapter requests
async def handle_codex_request(
    request: Request,
    adapter: Any,
) -> StreamingResponse | Response | DeferredStreaming:
    result = await adapter.handle_request(request)
    return cast(StreamingResponse | Response | DeferredStreaming, result)


# Route definitions
@router.post("/responses", response_model=None)
@with_format_chain(
    [FORMAT_OPENAI_RESPONSES], endpoint=UPSTREAM_ENDPOINT_OPENAI_RESPONSES
)
async def codex_responses(
    request: Request,
    auth: ConditionalAuthDep,
    adapter: CodexAdapterDep,
) -> StreamingResponse | Response | DeferredStreaming:
    return await handle_codex_request(request, adapter)


@router.post("/{session_id}/responses", response_model=None)
@with_format_chain([FORMAT_OPENAI_RESPONSES], endpoint="/{session_id}/responses")
async def codex_responses_with_session(
    session_id: str,
    request: Request,
    auth: ConditionalAuthDep,
    adapter: CodexAdapterDep,
) -> StreamingResponse | Response | DeferredStreaming:
    return await handle_codex_request(
        request,
        adapter,
    )


@router.post("/chat/completions", response_model=None)
@with_format_chain(
    [FORMAT_OPENAI_CHAT, FORMAT_OPENAI_RESPONSES],
    endpoint=UPSTREAM_ENDPOINT_OPENAI_CHAT_COMPLETIONS,
)
async def codex_chat_completions(
    request: Request,
    auth: ConditionalAuthDep,
    adapter: CodexAdapterDep,
) -> StreamingResponse | Response | DeferredStreaming:
    return await handle_codex_request(request, adapter)


@router.post("/{session_id}/chat/completions", response_model=None)
@with_format_chain(
    [FORMAT_OPENAI_CHAT, FORMAT_OPENAI_RESPONSES],
    endpoint="/{session_id}/chat/completions",
)
async def codex_chat_completions_with_session(
    session_id: str,
    request: Request,
    auth: ConditionalAuthDep,
    adapter: CodexAdapterDep,
) -> StreamingResponse | Response | DeferredStreaming:
    return await handle_codex_request(
        request,
        adapter,
    )


@router.post("/v1/chat/completions", response_model=None)
@with_format_chain(
    [FORMAT_OPENAI_CHAT, FORMAT_OPENAI_RESPONSES],
    endpoint="/v1/chat/completions",
)
async def codex_v1_chat_completions(
    request: Request,
    auth: ConditionalAuthDep,
    adapter: CodexAdapterDep,
) -> StreamingResponse | Response | DeferredStreaming:
    return await handle_codex_request(request, adapter)


@router.get("/v1/models", response_model=None)
async def list_models(
    request: Request,
    auth: ConditionalAuthDep,
) -> dict[str, Any]:
    """List available Codex models."""
    model_list = [
        "gpt-5",
        "gpt-5-2025-08-07",
        "gpt-5-mini",
        "gpt-5-mini-2025-08-07",
        "gpt-5-nano",
        "gpt-5-nano-2025-08-07",
    ]
    models: list[dict[str, Any]] = [
        {
            "id": model_id,
            "object": "model",
            "created": 1704000000,
            "owned_by": "openai",
            "permission": [],
            "root": model_id,
            "parent": None,
        }
        for model_id in model_list
    ]
    return {"object": "list", "data": models}


@router.post("/v1/messages", response_model=None)
@with_format_chain(
    [FORMAT_ANTHROPIC_MESSAGES, FORMAT_OPENAI_RESPONSES],
    endpoint=UPSTREAM_ENDPOINT_ANTHROPIC_MESSAGES,
)
async def codex_v1_messages(
    request: Request,
    auth: ConditionalAuthDep,
    adapter: CodexAdapterDep,
) -> StreamingResponse | Response | DeferredStreaming:
    return await handle_codex_request(request, adapter)


@router.post("/{session_id}/v1/messages", response_model=None)
@with_format_chain(
    [FORMAT_ANTHROPIC_MESSAGES, FORMAT_OPENAI_RESPONSES],
    endpoint="/{session_id}/v1/messages",
)
async def codex_v1_messages_with_session(
    session_id: str,
    request: Request,
    auth: ConditionalAuthDep,
    adapter: CodexAdapterDep,
) -> StreamingResponse | Response | DeferredStreaming:
    return await handle_codex_request(request, adapter)
