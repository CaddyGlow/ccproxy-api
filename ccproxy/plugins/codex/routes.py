"""Codex plugin routes."""

from typing import TYPE_CHECKING, Annotated, Any

from fastapi import APIRouter, Depends, Request
from starlette.responses import Response, StreamingResponse

from ccproxy.api.dependencies import get_plugin_adapter
from ccproxy.auth.conditional import ConditionalAuthDep
from ccproxy.streaming import DeferredStreaming


if TYPE_CHECKING:
    pass

CodexAdapterDep = Annotated[Any, Depends(get_plugin_adapter("codex"))]
router = APIRouter()


def codex_path_transformer(path: str) -> str:
    """Transform stripped paths for Codex API."""
    if (
        path.endswith("/chat/completions")
        or path.endswith("/completions")
        or path.endswith("/messages")
    ):
        return "/responses"
    return path


# Helper to handle adapter requests
async def handle_codex_request(
    request: Request,
    adapter: Any,
    endpoint: str,
    session_id: str | None = None,
) -> StreamingResponse | Response | DeferredStreaming:
    from typing import cast as _cast

    result = await adapter.handle_request(
        request=request,
        endpoint=endpoint,
        method=request.method,
        session_id=session_id,
    )
    return _cast(StreamingResponse | Response | DeferredStreaming, result)


# Route definitions
@router.post("/responses", response_model=None)
async def codex_responses(
    request: Request,
    auth: ConditionalAuthDep,
    adapter: CodexAdapterDep,
) -> StreamingResponse | Response | DeferredStreaming:
    return await handle_codex_request(request, adapter, "/responses")


@router.post("/{session_id}/responses", response_model=None)
async def codex_responses_with_session(
    session_id: str,
    request: Request,
    auth: ConditionalAuthDep,
    adapter: CodexAdapterDep,
) -> StreamingResponse | Response | DeferredStreaming:
    return await handle_codex_request(
        request, adapter, "/{session_id}/responses", session_id
    )


@router.post("/chat/completions", response_model=None)
async def codex_chat_completions(
    request: Request,
    auth: ConditionalAuthDep,
    adapter: CodexAdapterDep,
) -> StreamingResponse | Response | DeferredStreaming:
    return await handle_codex_request(request, adapter, "/chat/completions")


@router.post("/{session_id}/chat/completions", response_model=None)
async def codex_chat_completions_with_session(
    session_id: str,
    request: Request,
    auth: ConditionalAuthDep,
    adapter: CodexAdapterDep,
) -> StreamingResponse | Response | DeferredStreaming:
    return await handle_codex_request(
        request, adapter, "/{session_id}/chat/completions", session_id
    )


@router.post("/v1/chat/completions", response_model=None)
async def codex_v1_chat_completions(
    request: Request,
    auth: ConditionalAuthDep,
    adapter: CodexAdapterDep,
) -> StreamingResponse | Response | DeferredStreaming:
    return await codex_chat_completions(request, auth, adapter)


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
async def codex_v1_messages(
    request: Request,
    auth: ConditionalAuthDep,
    adapter: CodexAdapterDep,
) -> StreamingResponse | Response | DeferredStreaming:
    return await handle_codex_request(request, adapter, "/v1/messages")


@router.post("/{session_id}/v1/messages", response_model=None)
async def codex_v1_messages_with_session(
    session_id: str,
    request: Request,
    auth: ConditionalAuthDep,
    adapter: CodexAdapterDep,
) -> StreamingResponse | Response | DeferredStreaming:
    return await handle_codex_request(
        request, adapter, "/{session_id}/v1/messages", session_id
    )
