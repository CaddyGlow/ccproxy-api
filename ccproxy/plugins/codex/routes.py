"""Codex plugin routes."""

from typing import TYPE_CHECKING, Annotated, Any, cast

from fastapi import APIRouter, Depends, Request
from starlette.responses import Response, StreamingResponse

from ccproxy.api.decorators import with_format_chain
from ccproxy.api.dependencies import (
    get_plugin_adapter,
    get_provider_config_dependency,
)
from ccproxy.auth.dependencies import ConditionalAuthDep
from ccproxy.core.constants import (
    FORMAT_ANTHROPIC_MESSAGES,
    FORMAT_OPENAI_CHAT,
    FORMAT_OPENAI_RESPONSES,
    UPSTREAM_ENDPOINT_ANTHROPIC_MESSAGES,
    UPSTREAM_ENDPOINT_OPENAI_CHAT_COMPLETIONS,
    UPSTREAM_ENDPOINT_OPENAI_RESPONSES,
)
from ccproxy.streaming import DeferredStreaming

from .config import CodexSettings


if TYPE_CHECKING:
    pass

CodexAdapterDep = Annotated[Any, Depends(get_plugin_adapter("codex"))]
CodexConfigDep = Annotated[
    CodexSettings,
    Depends(get_provider_config_dependency("codex", CodexSettings)),
]
router = APIRouter()


# Helper to handle adapter requests
async def handle_codex_request(
    request: Request,
    adapter: Any,
) -> StreamingResponse | Response | DeferredStreaming:
    result = await adapter.handle_request(request)
    return cast(StreamingResponse | Response | DeferredStreaming, result)


# Route definitions
async def _codex_responses_handler(
    request: Request,
    adapter: CodexAdapterDep,
) -> StreamingResponse | Response | DeferredStreaming:
    """Shared handler for Codex responses endpoints."""

    return await handle_codex_request(request, adapter)


@router.post("/v1/responses", response_model=None)
@with_format_chain(
    [FORMAT_OPENAI_RESPONSES], endpoint=UPSTREAM_ENDPOINT_OPENAI_RESPONSES
)
async def codex_responses(
    request: Request,
    auth: ConditionalAuthDep,
    adapter: CodexAdapterDep,
) -> StreamingResponse | Response | DeferredStreaming:
    return await _codex_responses_handler(request, adapter)


@router.post("/responses", response_model=None, include_in_schema=False)
@with_format_chain(
    [FORMAT_OPENAI_RESPONSES], endpoint=UPSTREAM_ENDPOINT_OPENAI_RESPONSES
)
async def codex_responses_legacy(
    request: Request,
    auth: ConditionalAuthDep,
    adapter: CodexAdapterDep,
) -> StreamingResponse | Response | DeferredStreaming:
    return await _codex_responses_handler(request, adapter)


@router.post("/v1/chat/completions", response_model=None)
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


@router.get("/v1/models", response_model=None)
async def list_models(
    request: Request,
    auth: ConditionalAuthDep,
    config: CodexConfigDep,
) -> dict[str, Any]:
    """List available Codex models."""
    models = [card.model_dump(mode="json") for card in config.models_endpoint]
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
