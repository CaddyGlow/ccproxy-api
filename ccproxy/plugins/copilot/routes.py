"""API routes for GitHub Copilot plugin."""

import json
from typing import TYPE_CHECKING, Annotated, Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from ccproxy.adapters.openai.models import (
    OpenAIChatCompletionResponse,
    OpenAIErrorResponse,
    OpenAIModelInfo,
    OpenAIModelsResponse,
)
from ccproxy.adapters.openai.models.chat_completions import OpenAIChatCompletionRequest
from ccproxy.api.dependencies import get_plugin_adapter
from ccproxy.core.logging import get_plugin_logger
from ccproxy.models.messages import MessageCreateParams, MessageResponse
from ccproxy.models.responses import APIError
from ccproxy.plugins.copilot.adapter import CopilotAdapter
from ccproxy.plugins.copilot.oauth.provider import CopilotOAuthProvider
from ccproxy.streaming import DeferredStreaming

from .models import (
    CopilotEmbeddingRequest,
    CopilotErrorResponse,
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


# Core Copilot API endpoints


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
    request.state.context.metadata["endpoint"] = "/v1/chat/completions"
    return await _handle_adapter_request(request, adapter)


@router.post("/v1/messages", response_model=MessageResponse | APIError)
async def create_anthropic_message(
    request: Request,
    _: MessageCreateParams,
    adapter: CopilotAdapterDep,
) -> Response | StreamingResponse | DeferredStreaming:
    """Create a message using Copilot with native Anthropic format."""
    request.state.context.metadata["endpoint"] = "/v1/messages"
    request.state.context.format_chain = ["anthropic", "openai"]
    return await _handle_adapter_request(request, adapter)


@router.post(
    "/v1/engines/codegen/completions",
    response_model=OpenAIChatCompletionResponse | OpenAIErrorResponse,
)
async def create_openai_codgen_completion(
    request: Request,
    _: OpenAIChatCompletionRequest,
    adapter: CopilotAdapterDep,
) -> Response | StreamingResponse | DeferredStreaming:
    """Create a completion using Copilot with OpenAI-compatible format."""
    request.state.context.metadata["endpoint"] = "/v1/engines/codegen/completions"
    return await _handle_adapter_request(request, adapter)


@router.post("/v1/engines/codegen/messsage", response_model=MessageResponse | APIError)
async def create_anthropic_codegen_completion(
    request: Request,
    _: MessageCreateParams,
    adapter: CopilotAdapterDep,
) -> Response | StreamingResponse | DeferredStreaming:
    """Create a completion using Copilot with OpenAI-compatible format."""
    request.state.context.metadata["endpoint"] = "/v1/engines/codegen/completions"
    request.state.context.format_chain = ["anthropic", "openai"]
    return await _handle_adapter_request(request, adapter)


@router.post(
    "/v1/engines/copilot-codex/completions",
    response_model=OpenAIChatCompletionResponse | OpenAIErrorResponse,
)
async def create_openai_completion(
    request: Request,
    _: OpenAIChatCompletionRequest,
    adapter: CopilotAdapterDep,
) -> Response | StreamingResponse | DeferredStreaming:
    """Create a chat completion using Copilot with OpenAI-compatible format."""
    request.state.context.metadata["endpoint"] = "/v1/engines/copilot-codex/completions"
    return await _handle_adapter_request(request, adapter)


@router.post(
    "/v1/engines/copilot-codex/messages", response_model=MessageResponse | APIError
)
async def create_anthropic_completion(
    request: Request,
    _: MessageCreateParams,
    adapter: CopilotAdapterDep,
) -> Response | StreamingResponse | DeferredStreaming:
    """Create a chat completion using Copilot with OpenAI-compatible format."""
    request.state.context.metadata["endpoint"] = "/v1/engines/copilot-codex/completions"
    request.state.context.format_chain = ["anthropic", "openai"]
    return await _handle_adapter_request(request, adapter)


@router.post("/v1/embeddings", response_model=CopilotEmbeddingRequest)
async def create_embeddings(
    request: Request, adapter: CopilotAdapterDep
) -> JSONResponse:
    """Create embeddings using Copilot."""
    try:
        # Set endpoint metadata for adapter usage (no format chain needed - direct pass-through)
        request.state.context.metadata["endpoint"] = "/v1/embeddings"

        # Parse request body
        body = await request.body()
        try:
            request_data = json.loads(body) if body else {}
            embedding_request = CopilotEmbeddingRequest.model_validate(request_data)
        except json.JSONDecodeError as e:
            raise HTTPException(
                status_code=400, detail=f"Invalid JSON: {str(e)}"
            ) from e
        except ValueError as e:
            raise HTTPException(
                status_code=400, detail=f"Invalid request format: {str(e)}"
            ) from e

        logger.info(
            "embedding_request",
            model=embedding_request.model,
            input_type=type(embedding_request.input).__name__,
        )

        # Delegate to adapter
        response = await adapter.handle_request(request)

        # Ensure return type compatibility
        if isinstance(response, Response) and not isinstance(response, JSONResponse):
            return JSONResponse(
                content=json.loads(
                    response.body.decode()
                    if isinstance(response.body, bytes)
                    else str(response.body)
                )
                if response.body
                else {},
                status_code=response.status_code,
                headers=dict(response.headers),
            )
        # Type cast to satisfy mypy - we know response is JSONResponse
        return response  # type: ignore[no-any-return]

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "embedding_failed",
            error=str(e),
            exc_info=e,
        )
        raise HTTPException(
            status_code=500, detail=f"Embedding failed: {str(e)}"
        ) from e


# Copilot-specific information endpoints


@router.get("/usage", response_model=CopilotUserInternalResponse)
async def get_usage_stats(adapter: CopilotAdapterDep) -> JSONResponse:
    """Get Copilot usage statistics."""
    try:
        logger.debug("getting_usage_stats")

        # Forward request to upstream GitHub API
        response = await adapter.forward_to_github_api("/copilot_internal/user")

        content = response.body
        if isinstance(content, bytes):
            content_data = json.loads(content.decode()) if content else {}
        else:
            content_data = json.loads(content) if content else {}

        return JSONResponse(
            content=content_data,
            status_code=response.status_code,
            headers=dict(response.headers),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "usage_stats_failed",
            error=str(e),
            exc_info=e,
        )
        raise HTTPException(
            status_code=500, detail=f"Failed to get usage stats: {str(e)}"
        ) from e


@router.get("/token", response_model=CopilotTokenStatus)
async def get_token_status(adapter: CopilotAdapterDep) -> JSONResponse:
    """Get current token status."""
    try:
        if not adapter.oauth_provider:
            raise HTTPException(status_code=503, detail="OAuth provider not available")

        try:
            # Forward request to upstream GitHub API for token status
            response = await adapter.forward_to_github_api(
                "/copilot_internal/v2/token", use_oauth_token=True
            )
            return response

        except Exception as upstream_error:
            logger.warning(
                "upstream_token_status_failed_using_local",
                error=str(upstream_error),
                exc_info=upstream_error,
            )
    except Exception as e:
        logger.warning(
            "upstream_error",
            error=str(e),
            exc_info=e,
        )
        raise HTTPException(status_code=500, detail=f"upstream error: {str(e)}")


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
            headers={"X-Copilot-Provider": "ccproxy"},
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


@router.get("/v2/models", response_model=OpenAIModelsResponse)
async def list_models(adapter: CopilotAdapterDep) -> JSONResponse:
    """List available Copilot models."""
    try:
        # Forward request to upstream GitHub API for models
        response = await adapter.forward_to_github_api(
            "/copilot_internal/v2/models", use_oauth_token=True
        )
        return response

    except Exception as e:
        logger.warning(
            "upstream_models_failed_using_fallback",
            error=str(e),
            exc_info=e,
        )
        raise HTTPException(status_code=500, detail=f"Failed to list models: {str(e)}")


@router.get("/v1/models", response_model=OpenAIModelsResponse)
async def list_models_v1(adapter: CopilotAdapterDep) -> JSONResponse:
    """List available Copilot models."""

    try:
        # Forward request to upstream GitHub API for models
        response = await adapter.forward_to_github_api(
            "/copilot_internal/v1/models", use_oauth_token=True
        )
        return response

    except Exception as e:
        logger.warning(
            "upstream_error",
            error=str(e),
            exc_info=e,
        )
        raise HTTPException(status_code=500, detail=f"Failed to list models: {str(e)}")
