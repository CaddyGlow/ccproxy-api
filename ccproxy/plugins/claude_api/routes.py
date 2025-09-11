"""API routes for Claude API plugin."""

import uuid
from typing import TYPE_CHECKING, Annotated, Any

from fastapi import APIRouter, Depends, Request
from fastapi.responses import Response, StreamingResponse

from ccproxy.adapters.openai.models import (
    OpenAIChatCompletionResponse,
    OpenAIErrorResponse,
    OpenAIModelsResponse,
)
from ccproxy.adapters.openai.models.chat_completions import OpenAIChatCompletionRequest
from ccproxy.api.dependencies import get_plugin_adapter
from ccproxy.auth.conditional import ConditionalAuthDep
from ccproxy.core.logging import get_plugin_logger
from ccproxy.models.messages import MessageCreateParams, MessageResponse
from ccproxy.models.responses import APIError
from ccproxy.streaming import DeferredStreaming


if TYPE_CHECKING:
    pass

logger = get_plugin_logger()

ClaudeAPIAdapterDep = Annotated[Any, Depends(get_plugin_adapter("claude_api"))]

APIResponse = Response | StreamingResponse | DeferredStreaming

# Main API Router - Core Claude API endpoints
router = APIRouter()


def claude_api_path_transformer(path: str) -> str:
    """Transform stripped paths for Claude API."""
    if path in ("/v1/chat/completions", "/v1/responses", "/responses"):
        return "/v1/messages"
    return path


def _cast_result(result: object) -> APIResponse:
    from typing import cast as _cast

    return _cast(APIResponse, result)


async def _handle_adapter_request(
    request: Request,
    adapter: Any,
) -> APIResponse:
    result = await adapter.handle_request(request)
    return _cast_result(result)


@router.post("/v1/messages", response_model=MessageResponse | APIError)
async def create_anthropic_message(
    request: Request,
    _: MessageCreateParams,
    auth: ConditionalAuthDep,
    adapter: ClaudeAPIAdapterDep,
) -> APIResponse:
    """Create a message using Claude AI with native Anthropic format."""
    request.state.context.metadata["endpoint"] = "/v1/messages"
    return await _handle_adapter_request(request, adapter)


@router.post(
    "/v1/chat/completions",
    response_model=OpenAIChatCompletionResponse | OpenAIErrorResponse,
)
async def create_openai_chat_completion(
    request: Request,
    _: OpenAIChatCompletionRequest,
    auth: ConditionalAuthDep,
    adapter: ClaudeAPIAdapterDep,
) -> APIResponse:
    """Create a chat completion using Claude AI with OpenAI-compatible format."""
    request.state.context.format_chain = ["openai", "anthropic"]
    request.state.context.metadata["endpoint"] = "/v1/chat/completions"
    return await _handle_adapter_request(request, adapter)


@router.get("/v1/models", response_model=OpenAIModelsResponse)
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


@router.post("/v1/responses", response_model=MessageResponse | APIError)
async def claude_v1_responses(
    request: Request,
    auth: ConditionalAuthDep,
    adapter: ClaudeAPIAdapterDep,
) -> APIResponse:
    """Response API compatible endpoint using Claude backend."""
    request.state.context.format_chain = ["response_api", "anthropic"]
    request.state.context.metadata["endpoint"] = "/v1/responses"
    session_id = request.headers.get("session_id") or str(uuid.uuid4())
    return await _handle_adapter_request(request, adapter)


@router.post("/{session_id}/v1/responses", response_model=MessageResponse | APIError)
async def claude_v1_responses_with_session(
    session_id: str,
    request: Request,
    auth: ConditionalAuthDep,
    adapter: ClaudeAPIAdapterDep,
) -> APIResponse:
    """Response API with session_id using Claude backend."""
    request.state.context.format_chain = ["response_api", "anthropic"]
    request.state.context.metadata["endpoint"] = "/{session_id}/v1/responses"
    return await _handle_adapter_request(request, adapter)
