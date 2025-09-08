"""Claude API plugin routes."""

import uuid
from typing import TYPE_CHECKING, Annotated, Any

from fastapi import APIRouter, Depends, Request
from starlette.responses import Response, StreamingResponse

from ccproxy.api.dependencies import get_plugin_adapter
from ccproxy.auth.conditional import ConditionalAuthDep
from ccproxy.streaming.deferred import DeferredStreaming


if TYPE_CHECKING:
    pass

ClaudeAPIAdapterDep = Annotated[Any, Depends(get_plugin_adapter("claude_api"))]
router = APIRouter()


def claude_api_path_transformer(path: str) -> str:
    """Transform stripped paths for Claude API."""
    if path in ("/v1/chat/completions", "/v1/responses", "/responses"):
        return "/v1/messages"
    return path


def _cast_result(result: object) -> Response | StreamingResponse | DeferredStreaming:
    from typing import cast as _cast

    return _cast(Response | StreamingResponse | DeferredStreaming, result)


async def _handle_adapter_request(
    request: Request, adapter: Any, endpoint: str, **kwargs: Any
) -> Response | StreamingResponse | DeferredStreaming:
    result = await adapter.handle_request(
        request=request,
        endpoint=endpoint,
        method=request.method,
        **kwargs,
    )
    return _cast_result(result)


@router.post("/v1/messages", response_model=None)
async def create_anthropic_message(
    request: Request,
    auth: ConditionalAuthDep,
    adapter: ClaudeAPIAdapterDep,
) -> Response | StreamingResponse | DeferredStreaming:
    """Create a message using Claude AI with native Anthropic format."""
    return await _handle_adapter_request(request, adapter, "/v1/messages")


@router.post("/v1/chat/completions", response_model=None)
async def create_openai_chat_completion(
    request: Request,
    auth: ConditionalAuthDep,
    adapter: ClaudeAPIAdapterDep,
) -> Response | StreamingResponse | DeferredStreaming:
    """Create a chat completion using Claude AI with OpenAI-compatible format."""
    return await _handle_adapter_request(request, adapter, "/v1/chat/completions")


@router.get("/v1/models", response_model=None)
async def list_models(
    request: Request,
    auth: ConditionalAuthDep,
) -> dict[str, Any]:
    """List available Claude models."""
    model_list = [
        "claude-3-5-sonnet-20241022",
        "claude-3-5-haiku-20241022",
        "claude-3-opus-20240229",
        "claude-3-sonnet-20240229",
        "claude-3-haiku-20240307",
    ]
    models: list[dict[str, Any]] = [
        {
            "id": model_id,
            "object": "model",
            "created": 1696000000,
            "owned_by": "anthropic",
            "permission": [],
            "root": model_id,
            "parent": None,
        }
        for model_id in model_list
    ]
    return {"object": "list", "data": models}


@router.post("/v1/responses", response_model=None)
async def claude_v1_responses(
    request: Request,
    auth: ConditionalAuthDep,
    adapter: ClaudeAPIAdapterDep,
) -> StreamingResponse | Response | DeferredStreaming:
    """Response API compatible endpoint using Claude backend."""
    session_id = request.headers.get("session_id") or str(uuid.uuid4())
    return await _handle_adapter_request(
        request, adapter, "/v1/responses", session_id=session_id
    )


@router.post("/{session_id}/v1/responses", response_model=None)
async def claude_v1_responses_with_session(
    session_id: str,
    request: Request,
    auth: ConditionalAuthDep,
    adapter: ClaudeAPIAdapterDep,
) -> StreamingResponse | Response | DeferredStreaming:
    """Response API with session_id using Claude backend."""
    return await _handle_adapter_request(
        request, adapter, "/{session_id}/v1/responses", session_id=session_id
    )
