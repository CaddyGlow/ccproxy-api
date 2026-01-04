"""API routes for Claude API plugin."""

import uuid
from typing import TYPE_CHECKING, Annotated, Any, cast

from fastapi import APIRouter, Depends, Request
from fastapi.responses import Response, StreamingResponse

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
)
from ccproxy.core.logging import get_plugin_logger
from ccproxy.llms.models import anthropic as anthropic_models
from ccproxy.llms.models import openai as openai_models
from ccproxy.streaming import DeferredStreaming

from .config import ClaudeAPISettings


if TYPE_CHECKING:
    pass

logger = get_plugin_logger()

ClaudeAPIAdapterDep = Annotated[Any, Depends(get_plugin_adapter("claude_api"))]
ClaudeAPIConfigDep = Annotated[
    ClaudeAPISettings,
    Depends(get_provider_config_dependency("claude_api", ClaudeAPISettings)),
]

APIResponse = Response | StreamingResponse | DeferredStreaming

# Main API Router - Core Claude API endpoints
router = APIRouter()


def _cast_result(result: object) -> APIResponse:
    return cast(APIResponse, result)


async def _handle_adapter_request(
    request: Request,
    adapter: Any,
) -> APIResponse:
    result = await adapter.handle_request(request)
    return _cast_result(result)


@router.post(
    "/v1/messages",
    response_model=anthropic_models.MessageResponse | anthropic_models.APIError,
)
@with_format_chain(
    [FORMAT_ANTHROPIC_MESSAGES], endpoint=UPSTREAM_ENDPOINT_ANTHROPIC_MESSAGES
)
async def create_anthropic_message(
    request: Request,
    _: anthropic_models.CreateMessageRequest,
    auth: ConditionalAuthDep,
    adapter: ClaudeAPIAdapterDep,
) -> APIResponse:
    """Create a message using Claude AI with native Anthropic format."""
    return await _handle_adapter_request(request, adapter)


@router.post(
    "/v1/chat/completions",
    response_model=openai_models.ChatCompletionResponse | openai_models.ErrorResponse,
)
@with_format_chain(
    [FORMAT_OPENAI_CHAT, FORMAT_ANTHROPIC_MESSAGES],
    endpoint=UPSTREAM_ENDPOINT_ANTHROPIC_MESSAGES,
)
async def create_openai_chat_completion(
    request: Request,
    _: openai_models.ChatCompletionRequest,
    auth: ConditionalAuthDep,
    adapter: ClaudeAPIAdapterDep,
) -> APIResponse:
    """Create a chat completion using Claude AI with OpenAI-compatible format."""
    return await _handle_adapter_request(request, adapter)


@router.post("/v1/responses", response_model=None)
@with_format_chain(
    [FORMAT_OPENAI_RESPONSES, FORMAT_ANTHROPIC_MESSAGES],
    endpoint=UPSTREAM_ENDPOINT_ANTHROPIC_MESSAGES,
)
async def claude_v1_responses(
    request: Request,
    auth: ConditionalAuthDep,
    adapter: ClaudeAPIAdapterDep,
) -> APIResponse:
    """Response API compatible endpoint using Claude backend."""
    # Ensure format chain is present for request/response conversion
    # format chain and endpoint set by decorator
    session_id = request.headers.get("session_id") or str(uuid.uuid4())
    return await _handle_adapter_request(request, adapter)


@router.get("/v1/models", response_model=openai_models.ModelList)
async def list_models(
    request: Request,
    auth: ConditionalAuthDep,
    config: ClaudeAPIConfigDep,
) -> dict[str, Any]:
    """List available Claude models from configuration."""
    models = [card.model_dump(mode="json") for card in config.models_endpoint]
    return {"object": "list", "data": models}
