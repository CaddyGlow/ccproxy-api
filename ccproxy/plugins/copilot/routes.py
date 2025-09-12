"CopilotEmbeddingRequestAPI routes for GitHub Copilot plugin."

from typing import TYPE_CHECKING, Annotated, Any, Literal

from fastapi import APIRouter, Body, Depends, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from ccproxy.adapters.anthropic.models.messages import (
    MessageCreateParams,
    MessageResponse,
)
from ccproxy.adapters.openai.models import (
    OpenAIChatCompletionResponse,
    OpenAIErrorResponse,
    OpenAIModelsResponse,
)
from ccproxy.adapters.openai.models.chat_completions import OpenAIChatCompletionRequest
from ccproxy.adapters.openai.models.embedding import (
    OpenAIEmbedding,
    OpenAIEmbeddingResponse,
)
from ccproxy.adapters.openai.models.responses import ResponseRequest
from ccproxy.api.decorators import base_format, format_chain
from ccproxy.api.dependencies import get_plugin_adapter
from ccproxy.core.logging import get_plugin_logger
from ccproxy.streaming import DeferredStreaming

from .models import (
    CopilotHealthResponse,
    CopilotTokenStatus,
    CopilotUserInternalResponse,
)


if TYPE_CHECKING:
    pass

logger = get_plugin_logger()

CopilotAdapterDep = Annotated[Any, Depends(get_plugin_adapter("copilot"))]

APIResponse = Response | StreamingResponse | DeferredStreaming
OpenAIResponse = APIResponse | OpenAIErrorResponse

# V1 API Router - OpenAI/Anthropic compatible endpoints
router_v1 = APIRouter()

# GitHub Copilot specific router - usage, token, health endpoints
router_github = APIRouter()


def _cast_result(result: object) -> OpenAIResponse:
    from typing import cast as _cast

    return _cast(APIResponse, result)


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
    response_model=OpenAIChatCompletionResponse,
)
@base_format("openai")
async def create_openai_chat_completion(
    request: Request,
    adapter: CopilotAdapterDep,
    _: OpenAIChatCompletionRequest = Body(..., include_in_schema=True),
    body: dict[str, Any] = Depends(_get_request_body, use_cache=False),
) -> OpenAIChatCompletionResponse | OpenAIResponse:
    """Create a chat completion using Copilot with OpenAI-compatible format."""
    request.state.context.metadata["endpoint"] = "/chat/completions"
    return await _handle_adapter_request(request, adapter)


@router_v1.post(
    "/messages",
    response_model=MessageResponse,
)
@format_chain(
    ["anthropic", "openai"]
)  # Client expects Anthropic, provider speaks OpenAI
async def create_anthropic_message(
    request: Request,
    _: MessageCreateParams,
    adapter: CopilotAdapterDep,
) -> MessageResponse | OpenAIResponse:
    """Create a message using Copilot with native Anthropic format."""
    request.state.context.metadata["endpoint"] = "/chat/completions"
    return await _handle_adapter_request(request, adapter)


@format_chain(
    ["response_api", "anthropic", "openai"]
)  # Request: Response API -> Anthropic -> OpenAI
@router_v1.post(
    "/responses",
    response_model=MessageResponse,
)
async def create_responses_message(
    request: Request,
    _: ResponseRequest,
    adapter: CopilotAdapterDep,
) -> MessageResponse | OpenAIResponse:
    """Create a message using Response API with OpenAI provider.

    Request conversion: Response API -> Anthropic -> OpenAI.
    Response conversion: OpenAI -> Anthropic.
    """
    # Ensure format chain is present in context even if decorator injection is bypassed
    if not getattr(request.state, "context", None) or not getattr(
        request.state.context, "format_chain", None
    ):
        # Lazily create minimal context if missing (mirrors middleware behavior)
        import time
        import uuid

        from ccproxy.core.request_context import RequestContext

        if not getattr(request.state, "context", None):
            request.state.context = RequestContext(
                request_id=str(uuid.uuid4()),
                start_time=time.perf_counter(),
                logger=get_plugin_logger(),
            )
        request.state.context.format_chain = [
            "response_api",
            "anthropic",
            "openai",
        ]
    request.state.context.metadata["endpoint"] = "/chat/completions"
    return await _handle_adapter_request(request, adapter)


@router_v1.post(
    "/embeddings",
    response_model=OpenAIEmbeddingResponse,
)
async def create_embeddings(
    request: Request, _: OpenAIEmbedding, adapter: CopilotAdapterDep
) -> OpenAIEmbeddingResponse | OpenAIResponse:
    request.state.context.metadata["endpoint"] = "/embeddings"
    return await _handle_adapter_request(request, adapter)


@router_v1.get("/models", response_model=OpenAIModelsResponse)
async def list_models_v1(
    request: Request, adapter: CopilotAdapterDep
) -> OpenAIResponse:
    """List available Copilot models."""
    # Forward request to upstream Copilot API
    request.state.context.metadata["endpoint"] = "/models"
    return await _handle_adapter_request(request, adapter)


@router_github.get("/usage", response_model=CopilotUserInternalResponse)
async def get_usage_stats(adapter: CopilotAdapterDep, request: Request) -> Response:
    """Get Copilot usage statistics."""
    request.state.context.metadata["endpoint"] = "/copilot_internal/user"
    request.state.context.metadata["method"] = "get"
    result = await adapter.handle_request_gh_api(request)
    from typing import cast

    return cast(Response, result)


@router_github.get("/token", response_model=CopilotTokenStatus)
async def get_token_status(adapter: CopilotAdapterDep, request: Request) -> Response:
    """Get Copilot usage statistics."""
    request.state.context.metadata["endpoint"] = "/copilot_internal/v2/token"
    request.state.context.metadata["method"] = "get"
    result = await adapter.handle_request_gh_api(request)
    from typing import cast

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
