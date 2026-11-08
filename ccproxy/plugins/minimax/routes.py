"""API routes for the MiniMax provider plugin."""

from typing import Annotated, Any, cast

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
    UPSTREAM_ENDPOINT_OPENAI_CHAT_COMPLETIONS,
)
from ccproxy.core.logging import get_plugin_logger
from ccproxy.llms.models import anthropic as anthropic_models
from ccproxy.llms.models import openai as openai_models
from ccproxy.streaming import DeferredStreaming
from ccproxy.utils.model_mapper import ModelMapper

from .config import MINIMAX_REGION_ENDPOINTS, MiniMaxConfig
from .models import MiniMaxMessageRequest


logger = get_plugin_logger()

MiniMaxAdapterDep = Annotated[Any, Depends(get_plugin_adapter("minimax"))]
MiniMaxConfigDep = Annotated[
    MiniMaxConfig,
    Depends(get_provider_config_dependency("minimax", MiniMaxConfig)),
]

APIResponse = Response | StreamingResponse | DeferredStreaming

router = APIRouter()


def _cast_result(result: object) -> APIResponse:
    return cast(APIResponse, result)


async def _handle_adapter_request(request: Request, adapter: Any) -> APIResponse:
    result = await adapter.handle_request(request)
    return _cast_result(result)


@router.post(
    "/v1/chat/completions",
    response_model=openai_models.ChatCompletionResponse | openai_models.ErrorResponse,
)
async def create_openai_chat_completion(
    request: Request,
    _: openai_models.ChatCompletionRequest,
    auth: ConditionalAuthDep,
    adapter: MiniMaxAdapterDep,
) -> APIResponse:
    """Create a chat completion using MiniMax with the OpenAI-compatible format."""
    request.state.context.metadata["endpoint"] = (
        UPSTREAM_ENDPOINT_OPENAI_CHAT_COMPLETIONS
    )
    return await _handle_adapter_request(request, adapter)


@router.post(
    "/v1/messages",
    response_model=anthropic_models.MessageResponse | anthropic_models.APIError,
)
@with_format_chain(
    [FORMAT_ANTHROPIC_MESSAGES, FORMAT_OPENAI_CHAT],
    endpoint=UPSTREAM_ENDPOINT_OPENAI_CHAT_COMPLETIONS,
)
async def create_anthropic_message(
    request: Request,
    payload: MiniMaxMessageRequest,
    auth: ConditionalAuthDep,
    adapter: MiniMaxAdapterDep,
) -> APIResponse:
    """Preserve native M3 history and retain legacy model conversion."""
    mapper = adapter.model_mapper or ModelMapper(adapter.config.model_mappings)
    endpoints = MINIMAX_REGION_ENDPOINTS[adapter.config.region]
    native_endpoint = (
        adapter.config.base_url.rstrip("/") == endpoints["openai_base_url"]
        or adapter.config.anthropic_base_url.rstrip("/")
        != endpoints["anthropic_base_url"]
    )
    if native_endpoint and mapper.map(payload.model).mapped in {
        "MiniMax-M3.1-Flash",
        "MiniMax-M3",
    }:
        request.state.context.format_chain = [FORMAT_ANTHROPIC_MESSAGES]
        request.state.context.metadata["endpoint"] = "/v1/messages"
    return await _handle_adapter_request(request, adapter)


@router.get("/v1/models", response_model=openai_models.ModelList)
async def list_models(
    request: Request,
    auth: ConditionalAuthDep,
    config: MiniMaxConfigDep,
) -> dict[str, Any]:
    """List available MiniMax models from configuration."""
    models = [card.model_dump(mode="json") for card in config.models_endpoint]
    return {"object": "list", "data": models}
