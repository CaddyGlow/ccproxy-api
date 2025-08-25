"""Codex plugin routes."""

import uuid
from typing import TYPE_CHECKING, Annotated, Any

from fastapi import APIRouter, Depends, Request
from starlette.responses import Response, StreamingResponse

from ccproxy.api.dependencies import ProxyServiceDep, get_plugin_adapter
from ccproxy.auth.conditional import ConditionalAuthDep


if TYPE_CHECKING:
    pass

# Create plugin-specific adapter dependency
CodexAdapterDep = Annotated[Any, Depends(get_plugin_adapter("codex"))]

router = APIRouter(tags=["plugin-codex"])


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

    # For everything else, just return as-is
    return path


@router.post("/responses", response_model=None)
async def codex_responses(
    request: Request,
    proxy_service: ProxyServiceDep,
    auth: ConditionalAuthDep,
    adapter: CodexAdapterDep,
) -> StreamingResponse | Response:
    """Create Codex completion with auto-generated session_id.

    Delegates to the adapter which will handle the request properly.
    """
    # Get session_id from header if provided
    header_session_id = request.headers.get("session_id")
    session_id = header_session_id or str(uuid.uuid4())

    # Use ProxyService.handle_request to enable hook emissions
    return await proxy_service.handle_request(
        request=request,
        endpoint="/responses",
        method=request.method,
        provider="codex",
        plugin_name="codex",
        adapter_handler=adapter.handle_request,
        session_id=session_id,
    )


@router.post("/{session_id}/responses", response_model=None)
async def codex_responses_with_session(
    session_id: str,
    request: Request,
    proxy_service: ProxyServiceDep,
    auth: ConditionalAuthDep,
    adapter: CodexAdapterDep,
) -> StreamingResponse | Response:
    """Create Codex completion with specific session_id.

    Delegates to the adapter which will handle the request properly.
    """
    # Use ProxyService.handle_request to enable hook emissions
    return await proxy_service.handle_request(
        request=request,
        endpoint="/{session_id}/responses",
        method=request.method,
        provider="codex",
        plugin_name="codex",
        adapter_handler=adapter.handle_request,
        session_id=session_id,
    )


@router.post("/chat/completions", response_model=None)
async def codex_chat_completions(
    request: Request,
    proxy_service: ProxyServiceDep,
    auth: ConditionalAuthDep,
    adapter: CodexAdapterDep,
) -> StreamingResponse | Response:
    """Create a chat completion using Codex with OpenAI-compatible format.

    This endpoint handles OpenAI format requests and converts them
    to/from Codex Response API format transparently.
    """

    # Get session_id from header if provided
    header_session_id = request.headers.get("session_id")
    session_id = header_session_id or str(uuid.uuid4())

    # Use ProxyService.handle_request to enable hook emissions
    return await proxy_service.handle_request(
        request=request,
        endpoint="/chat/completions",
        method=request.method,
        provider="codex",
        plugin_name="codex",
        adapter_handler=adapter.handle_request,
        session_id=session_id,
    )


@router.post("/{session_id}/chat/completions", response_model=None)
async def codex_chat_completions_with_session(
    session_id: str,
    request: Request,
    proxy_service: ProxyServiceDep,
    auth: ConditionalAuthDep,
    adapter: CodexAdapterDep,
) -> StreamingResponse | Response:
    """Create a chat completion with specific session_id using OpenAI format.

    This endpoint handles OpenAI format requests with a specific session_id.
    """
    # Use ProxyService.handle_request to enable hook emissions
    return await proxy_service.handle_request(
        request=request,
        endpoint="/{session_id}/chat/completions",
        method=request.method,
        provider="codex",
        plugin_name="codex",
        adapter_handler=adapter.handle_request,
        session_id=session_id,
    )


@router.post("/v1/chat/completions", response_model=None)
async def codex_v1_chat_completions(
    request: Request,
    proxy_service: ProxyServiceDep,
    auth: ConditionalAuthDep,
    adapter: CodexAdapterDep,
) -> StreamingResponse | Response:
    """OpenAI v1 compatible chat completions endpoint.

    Maps to the standard chat completions handler.
    """
    return await codex_chat_completions(request, proxy_service, auth, adapter)


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
