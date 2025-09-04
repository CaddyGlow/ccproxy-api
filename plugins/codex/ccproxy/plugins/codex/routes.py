"""Codex plugin routes."""

import uuid
from typing import TYPE_CHECKING, Annotated, Any

from fastapi import APIRouter, Depends, Request
from starlette.responses import Response, StreamingResponse

from ccproxy.api.dependencies import get_plugin_adapter
from ccproxy.auth.conditional import ConditionalAuthDep
from ccproxy.streaming.deferred_streaming import DeferredStreaming


if TYPE_CHECKING:
    pass

# Create plugin-specific adapter dependency
CodexAdapterDep = Annotated[Any, Depends(get_plugin_adapter("codex"))]

router = APIRouter()


def codex_path_transformer(path: str) -> str:
    """Transform stripped paths for Codex API.

    The path comes in already stripped of the /codex prefix.
    Maps various endpoint patterns to the Codex /responses endpoint.
    """
    # Map chat completions to Codex responses
    if path == "/chat/completions" or path == "/v1/chat/completions":
        return "/responses"

    # Map OpenAI-style completions to Codex responses
    if path == "/completions" or path == "/v1/completions":
        return "/responses"

    # Map Anthropic-style messages to Codex responses
    if path == "/v1/messages" or path == "/messages":
        return "/responses"

    # For everything else, just return as-is
    return path


@router.post("/responses", response_model=None)
async def codex_responses(
    request: Request,
    auth: ConditionalAuthDep,
    adapter: CodexAdapterDep,
) -> StreamingResponse | Response | DeferredStreaming:
    """Create Codex completion with auto-generated session_id.

    Delegates to the adapter which will handle the request properly.
    """
    # Get session_id from header if provided
    header_session_id = request.headers.get("session_id")
    session_id = header_session_id or str(uuid.uuid4())

    # Call adapter directly - hooks are now handled by HooksMiddleware
    result = await adapter.handle_request(
        request=request,
        endpoint="/responses",
        method=request.method,
        session_id=session_id,
    )
    from typing import cast as _cast

    return _cast(StreamingResponse | Response | DeferredStreaming, result)


@router.post("/{session_id}/responses", response_model=None)
async def codex_responses_with_session(
    session_id: str,
    request: Request,
    auth: ConditionalAuthDep,
    adapter: CodexAdapterDep,
) -> StreamingResponse | Response | DeferredStreaming:
    """Create Codex completion with specific session_id.

    Delegates to the adapter which will handle the request properly.
    """
    # Call adapter directly - hooks are now handled by HooksMiddleware
    result = await adapter.handle_request(
        request=request,
        endpoint="/{session_id}/responses",
        method=request.method,
        session_id=session_id,
    )
    from typing import cast as _cast

    return _cast(StreamingResponse | Response | DeferredStreaming, result)


@router.post("/chat/completions", response_model=None)
async def codex_chat_completions(
    request: Request,
    auth: ConditionalAuthDep,
    adapter: CodexAdapterDep,
) -> StreamingResponse | Response | DeferredStreaming:
    """Create a chat completion using Codex with OpenAI-compatible format.

    This endpoint handles OpenAI format requests and converts them
    to/from Codex Response API format transparently.
    """

    # Get session_id from header if provided
    header_session_id = request.headers.get("session_id")
    session_id = header_session_id or str(uuid.uuid4())

    # Call adapter directly - hooks are now handled by HooksMiddleware
    result = await adapter.handle_request(
        request=request,
        endpoint="/chat/completions",
        method=request.method,
        session_id=session_id,
    )
    from typing import cast as _cast

    return _cast(StreamingResponse | Response | DeferredStreaming, result)


@router.post("/{session_id}/chat/completions", response_model=None)
async def codex_chat_completions_with_session(
    session_id: str,
    request: Request,
    auth: ConditionalAuthDep,
    adapter: CodexAdapterDep,
) -> StreamingResponse | Response | DeferredStreaming:
    """Create a chat completion with specific session_id using OpenAI format.

    This endpoint handles OpenAI format requests with a specific session_id.
    """
    # Call adapter directly - hooks are now handled by HooksMiddleware
    result = await adapter.handle_request(
        request=request,
        endpoint="/{session_id}/chat/completions",
        method=request.method,
        session_id=session_id,
    )
    from typing import cast as _cast

    return _cast(StreamingResponse | Response | DeferredStreaming, result)


@router.post("/v1/chat/completions", response_model=None)
async def codex_v1_chat_completions(
    request: Request,
    auth: ConditionalAuthDep,
    adapter: CodexAdapterDep,
) -> StreamingResponse | Response | DeferredStreaming:
    """OpenAI v1 compatible chat completions endpoint.

    Maps to the standard chat completions handler.
    """
    return await codex_chat_completions(request, auth, adapter)


@router.get("/v1/models", response_model=None)
async def list_models(
    request: Request,
    auth: ConditionalAuthDep,
) -> dict[str, Any]:
    """List available Codex models.

    Returns a list of available models in OpenAI-compatible format.
    """
    # Build OpenAI-compatible model list
    models = []
    model_list = [
        "gpt-5",
        "gpt-5-2025-08-07",
        "gpt-5-mini",
        "gpt-5-mini-2025-08-07",
        "gpt-5-nano",
        "gpt-5-nano-2025-08-07",
    ]

    for model_id in model_list:
        models.append(
            {
                "id": model_id,
                "object": "model",
                "created": 1704000000,  # Placeholder timestamp
                "owned_by": "openai",
                "permission": [],
                "root": model_id,
                "parent": None,
            }
        )

    return {
        "object": "list",
        "data": models,
    }


@router.post("/v1/messages", response_model=None)
async def codex_v1_messages(
    request: Request,
    auth: ConditionalAuthDep,
    adapter: CodexAdapterDep,
) -> StreamingResponse | Response | DeferredStreaming:
    """Anthropic Messages API compatible endpoint using Codex backend.

    This endpoint handles Anthropic Messages API format requests and converts them
    to/from Codex Response API format transparently, with full function calling support.
    """
    # Get session_id from header if provided
    header_session_id = request.headers.get("session_id")
    session_id = header_session_id or str(uuid.uuid4())

    # Call adapter directly - hooks are now handled by HooksMiddleware
    result = await adapter.handle_request(
        request=request,
        endpoint="/v1/messages",
        method=request.method,
        session_id=session_id,
    )
    from typing import cast as _cast

    return _cast(StreamingResponse | Response | DeferredStreaming, result)


@router.post("/{session_id}/v1/messages", response_model=None)
async def codex_v1_messages_with_session(
    session_id: str,
    request: Request,
    auth: ConditionalAuthDep,
    adapter: CodexAdapterDep,
) -> StreamingResponse | Response | DeferredStreaming:
    """Anthropic Messages API with session_id using Codex backend.

    This endpoint handles Anthropic Messages API format requests with a specific session_id.
    Includes full function calling support through Response API transformation.
    """
    # Call adapter directly - hooks are now handled by HooksMiddleware
    result = await adapter.handle_request(
        request=request,
        endpoint="/{session_id}/v1/messages",
        method=request.method,
        session_id=session_id,
    )
    from typing import cast as _cast

    return _cast(StreamingResponse | Response | DeferredStreaming, result)
