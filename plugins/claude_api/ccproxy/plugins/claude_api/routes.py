"""Claude API plugin routes."""

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
ClaudeAPIAdapterDep = Annotated[Any, Depends(get_plugin_adapter("claude_api"))]

router = APIRouter()


def claude_api_path_transformer(path: str) -> str:
    """Transform stripped paths for Claude API.

    The path comes in already stripped of the /claude-api prefix.
    Maps various endpoint patterns to their Claude API equivalents.
    """
    # Map OpenAI chat completions to Anthropic messages
    if path == "/v1/chat/completions":
        return "/v1/messages"

    # Map Response API format to Anthropic messages
    if path == "/v1/responses" or path == "/responses":
        return "/v1/messages"

    # Pass through native Anthropic paths
    return path


# Note: Routes now call adapters directly. Hook emissions are handled by HooksMiddleware.


@router.post("/v1/messages", response_model=None)
async def create_anthropic_message(
    request: Request,
    auth: ConditionalAuthDep,
    adapter: ClaudeAPIAdapterDep,
) -> Response | StreamingResponse | DeferredStreaming:
    """Create a message using Claude AI with native Anthropic format.

    This endpoint handles Anthropic API format requests and forwards them
    directly to the Claude API without format conversion.
    """
    # Call adapter directly - hooks are now handled by HooksMiddleware
    from typing import cast as _cast

    result = await adapter.handle_request(
        request=request,
        endpoint="/v1/messages",
        method=request.method,
    )
    return _cast(Response | StreamingResponse | DeferredStreaming, result)


@router.post("/v1/chat/completions", response_model=None)
async def create_openai_chat_completion(
    request: Request,
    auth: ConditionalAuthDep,
    adapter: ClaudeAPIAdapterDep,
) -> Response | StreamingResponse | DeferredStreaming:
    """Create a chat completion using Claude AI with OpenAI-compatible format.

    This endpoint handles OpenAI format requests and converts them
    to/from Anthropic format transparently.
    """
    # Call adapter directly - hooks are now handled by HooksMiddleware
    from typing import cast as _cast

    result = await adapter.handle_request(
        request=request,
        endpoint="/v1/chat/completions",
        method=request.method,
    )
    return _cast(Response | StreamingResponse | DeferredStreaming, result)


@router.get("/v1/models", response_model=None)
async def list_models(
    request: Request,
    auth: ConditionalAuthDep,
) -> dict[str, Any]:
    """List available Claude models.

    Returns a list of available models in OpenAI-compatible format.
    """

    # Build OpenAI-compatible model list
    models = []
    model_list = [
        "claude-3-5-sonnet-20241022",
        "claude-3-5-haiku-20241022",
        "claude-3-opus-20240229",
        "claude-3-sonnet-20240229",
        "claude-3-haiku-20240307",
    ]

    for model_id in model_list:
        models.append(
            {
                "id": model_id,
                "object": "model",
                "created": 1696000000,  # Placeholder timestamp
                "owned_by": "anthropic",
                "permission": [],
                "root": model_id,
                "parent": None,
            }
        )

    return {
        "object": "list",
        "data": models,
    }


@router.post("/v1/responses", response_model=None)
async def claude_v1_responses(
    request: Request,
    auth: ConditionalAuthDep,
    adapter: ClaudeAPIAdapterDep,
) -> StreamingResponse | Response | DeferredStreaming:
    """Response API compatible endpoint using Claude backend.

    This endpoint handles Response API format requests and converts them
    to/from Claude API format transparently.
    """
    # Get session_id from header if provided
    header_session_id = request.headers.get("session_id")
    session_id = header_session_id or str(uuid.uuid4())

    # Call adapter directly - hooks are now handled by HooksMiddleware
    result = await adapter.handle_request(
        request=request,
        endpoint="/v1/responses",
        method=request.method,
        session_id=session_id,
    )
    from typing import cast as _cast

    return _cast(StreamingResponse | Response | DeferredStreaming, result)


@router.post("/{session_id}/v1/responses", response_model=None)
async def claude_v1_responses_with_session(
    session_id: str,
    request: Request,
    auth: ConditionalAuthDep,
    adapter: ClaudeAPIAdapterDep,
) -> StreamingResponse | Response | DeferredStreaming:
    """Response API with session_id using Claude backend.

    This endpoint handles Response API format requests with a specific session_id.
    """
    # Call adapter directly - hooks are now handled by HooksMiddleware
    result = await adapter.handle_request(
        request=request,
        endpoint="/{session_id}/v1/responses",
        method=request.method,
        session_id=session_id,
    )
    from typing import cast as _cast

    return _cast(StreamingResponse | Response | DeferredStreaming, result)
