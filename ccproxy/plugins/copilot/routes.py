"CopilotEmbeddingRequestAPI routes for GitHub Copilot plugin."

from typing import TYPE_CHECKING, Annotated, Any, Literal, cast

from fastapi import APIRouter, Body, Depends, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from ccproxy.api.decorators import with_format_chain
from ccproxy.api.dependencies import (
    get_plugin_adapter,
    get_provider_config_dependency,
)
from ccproxy.core.constants import (
    FORMAT_ANTHROPIC_MESSAGES,
    FORMAT_OPENAI_CHAT,
    FORMAT_OPENAI_RESPONSES,
    UPSTREAM_ENDPOINT_COPILOT_INTERNAL_TOKEN,
    UPSTREAM_ENDPOINT_COPILOT_INTERNAL_USER,
    UPSTREAM_ENDPOINT_OPENAI_CHAT_COMPLETIONS,
    UPSTREAM_ENDPOINT_OPENAI_EMBEDDINGS,
    UPSTREAM_ENDPOINT_OPENAI_MODELS,
)
from ccproxy.core.logging import get_plugin_logger
from ccproxy.llms.models import anthropic as anthropic_models
from ccproxy.llms.models import openai as openai_models
from ccproxy.streaming import DeferredStreaming

from .config import CopilotProviderConfig
from .models import (
    CopilotHealthResponse,
    CopilotTokenStatus,
    CopilotUserInternalResponse,
)


if TYPE_CHECKING:
    pass

logger = get_plugin_logger()

CopilotAdapterDep = Annotated[Any, Depends(get_plugin_adapter("copilot"))]
CopilotConfigDep = Annotated[
    CopilotProviderConfig,
    Depends(get_provider_config_dependency("copilot", CopilotProviderConfig)),
]

APIResponse = Response | StreamingResponse | DeferredStreaming
OpenAIResponse = APIResponse | openai_models.ErrorResponse

# V1 API Router - OpenAI/Anthropic compatible endpoints
router_v1 = APIRouter()

# GitHub Copilot specific router - usage, token, health endpoints
router_github = APIRouter()


def _cast_result(result: object) -> OpenAIResponse:
    return cast(APIResponse, result)


async def _handle_adapter_request(
    request: Request,
    adapter: Any,
) -> OpenAIResponse:
    result = await adapter.handle_request(request)
    return _cast_result(result)


def _get_request_body(request: Request) -> Any:
    """Hidden dependency to get raw body."""

    async def _inner() -> Any:
        return await request.json()

    return _inner


@router_v1.post(
    "/chat/completions",
    response_model=openai_models.ChatCompletionResponse,
)
async def create_openai_chat_completion(
    request: Request,
    adapter: CopilotAdapterDep,
    _: openai_models.ChatCompletionRequest = Body(..., include_in_schema=True),
    body: dict[str, Any] = Depends(_get_request_body, use_cache=False),
) -> openai_models.ChatCompletionResponse | OpenAIResponse:
    """Create a chat completion using Copilot with OpenAI-compatible format."""
    request.state.context.metadata["endpoint"] = (
        UPSTREAM_ENDPOINT_OPENAI_CHAT_COMPLETIONS
    )
    return await _handle_adapter_request(request, adapter)


@router_v1.post(
    "/messages",
    response_model=anthropic_models.MessageResponse,
)
@with_format_chain(
    [FORMAT_ANTHROPIC_MESSAGES, FORMAT_OPENAI_CHAT],
    endpoint=UPSTREAM_ENDPOINT_OPENAI_CHAT_COMPLETIONS,
)
async def create_anthropic_message(
    request: Request,
    _: anthropic_models.CreateMessageRequest,
    adapter: CopilotAdapterDep,
) -> anthropic_models.MessageResponse | OpenAIResponse:
    return await _handle_adapter_request(request, adapter)


@with_format_chain(
    [FORMAT_OPENAI_RESPONSES, FORMAT_OPENAI_CHAT],
    endpoint=UPSTREAM_ENDPOINT_OPENAI_CHAT_COMPLETIONS,
)
@router_v1.post(
    "/responses",
    response_model=anthropic_models.MessageResponse,
)
async def create_responses_message(
    request: Request,
    _: openai_models.ResponseRequest,
    adapter: CopilotAdapterDep,
) -> anthropic_models.MessageResponse | OpenAIResponse:
    """Create a message using Response API with OpenAI provider."""
    # Ensure format chain is present in context even if decorator injection is bypassed
    request.state.context.metadata["endpoint"] = (
        UPSTREAM_ENDPOINT_OPENAI_CHAT_COMPLETIONS
    )
    # Explicitly set format_chain so BaseHTTPAdapter applies request conversion
    try:
        prev_chain = getattr(request.state.context, "format_chain", None)
        new_chain = [FORMAT_OPENAI_RESPONSES, FORMAT_OPENAI_CHAT]
        request.state.context.format_chain = new_chain
        logger.debug(
            "copilot_responses_route_enter",
            prev_chain=prev_chain,
            applied_chain=new_chain,
            category="format",
        )
        # Peek at incoming body keys for debugging
        try:
            body_json = await request.json()
            stream_flag = (
                body_json.get("stream") if isinstance(body_json, dict) else None
            )
            logger.debug(
                "copilot_responses_request_body_inspect",
                keys=list(body_json.keys()) if isinstance(body_json, dict) else None,
                stream=stream_flag,
                category="format",
            )
        except Exception as exc:  # best-effort logging only
            logger.debug("copilot_responses_request_body_parse_failed", error=str(exc))
    except Exception as exc:  # defensive
        logger.debug("copilot_responses_set_chain_failed", error=str(exc))
    return await _handle_adapter_request(request, adapter)


@router_v1.post(
    "/embeddings",
    response_model=openai_models.EmbeddingResponse,
)
async def create_embeddings(
    request: Request, _: openai_models.EmbeddingRequest, adapter: CopilotAdapterDep
) -> openai_models.EmbeddingResponse | OpenAIResponse:
    request.state.context.metadata["endpoint"] = UPSTREAM_ENDPOINT_OPENAI_EMBEDDINGS
    return await _handle_adapter_request(request, adapter)


@router_v1.get("/models", response_model=openai_models.ModelList)
async def list_models_v1(
    request: Request,
    adapter: CopilotAdapterDep,
    config: CopilotConfigDep,
) -> OpenAIResponse:
    """List available Copilot models."""
    # if config.models_endpoint:
    #     models = [card.model_dump(mode="json") for card in config.models_endpoint]
    #     return JSONResponse(content={"object": "list", "data": models})

    # Forward request to upstream Copilot API when no override configured
    request.state.context.metadata["endpoint"] = UPSTREAM_ENDPOINT_OPENAI_MODELS
    return await _handle_adapter_request(request, adapter)


@router_github.get("/usage", response_model=CopilotUserInternalResponse)
async def get_usage_stats(adapter: CopilotAdapterDep, request: Request) -> Response:
    """Get Copilot usage statistics."""
    request.state.context.metadata["endpoint"] = UPSTREAM_ENDPOINT_COPILOT_INTERNAL_USER
    request.state.context.metadata["method"] = "get"
    result = await adapter.handle_request_gh_api(request)
    return cast(Response, result)


@router_github.get("/token", response_model=CopilotTokenStatus)
async def get_token_status(adapter: CopilotAdapterDep, request: Request) -> Response:
    """Get Copilot usage statistics."""
    request.state.context.metadata["endpoint"] = (
        UPSTREAM_ENDPOINT_COPILOT_INTERNAL_TOKEN
    )
    request.state.context.metadata["method"] = "get"
    result = await adapter.handle_request_gh_api(request)
    return cast(Response, result)


@router_github.get("/health", response_model=CopilotHealthResponse)
async def health_check(adapter: CopilotAdapterDep) -> JSONResponse:
    """Check Copilot plugin health."""
    try:
        logger.debug("performing_health_check")

        # Check components
        details: dict[str, Any] = {}

        # Check OAuth provider
        oauth_healthy = True
        if adapter.oauth_provider:
            try:
                oauth_healthy = await adapter.oauth_provider.is_authenticated()
                details["oauth"] = {
                    "authenticated": oauth_healthy,
                    "provider": "github_copilot",
                }
            except Exception as e:
                oauth_healthy = False
                details["oauth"] = {
                    "authenticated": False,
                    "error": str(e),
                }
        else:
            oauth_healthy = False
            details["oauth"] = {"error": "OAuth provider not initialized"}

        # Check detection service
        detection_healthy = True
        if adapter.detection_service:
            try:
                cli_info = adapter.detection_service.get_cli_health_info()
                details["github_cli"] = {
                    "available": cli_info.available,
                    "version": cli_info.version,
                    "authenticated": cli_info.authenticated,
                    "username": cli_info.username,
                    "error": cli_info.error,
                }
                detection_healthy = cli_info.available and cli_info.authenticated
            except Exception as e:
                detection_healthy = False
                details["github_cli"] = {"error": str(e)}
        else:
            details["github_cli"] = {"error": "Detection service not initialized"}

        # Overall health
        overall_status: Literal["healthy", "unhealthy"] = (
            "healthy" if oauth_healthy and detection_healthy else "unhealthy"
        )

        health_response = CopilotHealthResponse(
            status=overall_status,
            provider="copilot",
            details=details,
        )

        status_code = 200 if overall_status == "healthy" else 503

        logger.info(
            "health_check_completed",
            status=overall_status,
            oauth_healthy=oauth_healthy,
            detection_healthy=detection_healthy,
        )

        return JSONResponse(
            content=health_response.model_dump(),
            status_code=status_code,
        )

    except Exception as e:
        logger.error(
            "health_check_failed",
            error=str(e),
            exc_info=e,
        )

        health_response = CopilotHealthResponse(
            status="unhealthy",
            provider="copilot",
            details={"error": str(e)},
        )

        return JSONResponse(
            content=health_response.model_dump(),
            status_code=503,
        )
