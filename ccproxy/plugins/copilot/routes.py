"""API routes for GitHub Copilot plugin."""

from typing import TYPE_CHECKING, Annotated, Any, Literal

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from ccproxy.adapters.openai.models import (
    OpenAIChatCompletionResponse,
    OpenAIErrorResponse,
    OpenAIModelsResponse,
)
from ccproxy.adapters.openai.models.chat_completions import OpenAIChatCompletionRequest
from ccproxy.api.dependencies import get_plugin_adapter
from ccproxy.core.logging import get_plugin_logger
from ccproxy.models.messages import MessageCreateParams, MessageResponse
from ccproxy.models.responses import APIError
from ccproxy.streaming import DeferredStreaming

from .models import (
    CopilotEmbeddingRequest,
    CopilotHealthResponse,
    CopilotTokenStatus,
    CopilotUserInternalResponse,
)


if TYPE_CHECKING:
    pass

logger = get_plugin_logger()

CopilotAdapterDep = Annotated[Any, Depends(get_plugin_adapter("copilot"))]
router = APIRouter(tags=["copilot"])


def _cast_result(result: object) -> Response | StreamingResponse | DeferredStreaming:
    from typing import cast as _cast

    return _cast(Response | StreamingResponse | DeferredStreaming, result)


async def _handle_adapter_request(
    request: Request,
    adapter: Any,
) -> Response | StreamingResponse | DeferredStreaming:
    result = await adapter.handle_request(request)
    return _cast_result(result)


@router.post(
    "/v1/chat/completions",
    response_model=OpenAIChatCompletionResponse | OpenAIErrorResponse,
)
async def create_openai_chat_completion(
    request: Request,
    _: OpenAIChatCompletionRequest,
    adapter: CopilotAdapterDep,
) -> Response | StreamingResponse | DeferredStreaming:
    """Create a chat completion using Copilot with OpenAI-compatible format."""
    request.state.context.metadata["endpoint"] = "/chat/completions"
    return await _handle_adapter_request(request, adapter)


@router.post("/v1/messages", response_model=MessageResponse | APIError)
async def create_anthropic_message(
    request: Request,
    _: MessageCreateParams,
    adapter: CopilotAdapterDep,
) -> Response | StreamingResponse | DeferredStreaming:
    """Create a message using Copilot with native Anthropic format."""
    request.state.context.metadata["endpoint"] = "/chat/completions"
    request.state.context.format_chain = ["anthropic", "openai"]
    return await _handle_adapter_request(request, adapter)


@router.post("/v1/embeddings", response_model=CopilotEmbeddingRequest)
async def create_embeddings(request: Request, adapter: CopilotAdapterDep) -> Response:
    request.state.context.metadata["endpoint"] = "/embeddings"
    return await _handle_adapter_request(request, adapter)


@router.get("/v1/models", response_model=OpenAIModelsResponse)
async def list_models_v1(request: Request, adapter: CopilotAdapterDep) -> Response:
    """List available Copilot models."""
    # Forward request to upstream Copilot API
    request.state.context.metadata["endpoint"] = "/models"
    return await _handle_adapter_request(request, adapter)


@router.get("/usage", response_model=CopilotUserInternalResponse)
async def get_usage_stats(adapter: CopilotAdapterDep, request: Request) -> JSONResponse:
    """Get Copilot usage statistics."""
    request.state.context.metadata["endpoint"] = "/copilot_internal/user"
    request.state.context.metadata["method"] = "get"
    return await adapter.handle_request_gh_api(request)


@router.get("/token", response_model=CopilotTokenStatus)
async def get_token_status(
    adapter: CopilotAdapterDep, request: Request
) -> JSONResponse:
    """Get Copilot usage statistics."""
    request.state.context.metadata["endpoint"] = "/copilot_internal/v2/token"
    request.state.context.metadata["method"] = "get"
    return await adapter.handle_request_gh_api(request)


@router.get("/health", response_model=CopilotHealthResponse)
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
