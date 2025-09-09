"""API routes for GitHub Copilot plugin."""

import json
from typing import TYPE_CHECKING, Annotated, Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from ccproxy.api.dependencies import get_plugin_adapter
from ccproxy.core.logging import get_plugin_logger
from ccproxy.streaming import DeferredStreaming

from .models import (
    CopilotChatRequest,
    CopilotEmbeddingRequest,
    CopilotHealthResponse,
    CopilotModel,
    CopilotModelsResponse,
    CopilotTokenStatus,
    CopilotUserInternalResponse,
)


if TYPE_CHECKING:
    pass

logger = get_plugin_logger()

CopilotAdapterDep = Annotated[Any, Depends(get_plugin_adapter("copilot"))]
router = APIRouter(tags=["copilot"])


# Core Copilot API endpoints


@router.get("/v1/models", response_model=CopilotModelsResponse)
async def list_models(adapter: CopilotAdapterDep) -> JSONResponse:
    """List available Copilot models."""
    try:
        logger.debug("listing_copilot_models")

        # Load models from fallback data
        import pathlib

        fallback_path = pathlib.Path(__file__).parent / "data" / "copilot_fallback.json"

        try:
            with fallback_path.open("r") as f:
                fallback_data = json.load(f)
                models_data = fallback_data.get("models", [])
        except Exception as e:
            logger.warning(
                "fallback_data_load_failed",
                error=str(e),
                exc_info=e,
            )
            # Provide basic model list as fallback
            models_data = [
                {
                    "id": "gpt-4",
                    "object": "model",
                    "created": 1687882411,
                    "owned_by": "github",
                },
                {
                    "id": "gpt-3.5-turbo",
                    "object": "model",
                    "created": 1687882411,
                    "owned_by": "github",
                },
            ]

        models = [CopilotModel.model_validate(model) for model in models_data]
        response = CopilotModelsResponse(data=models)

        logger.info(
            "models_listed",
            model_count=len(models),
        )

        return JSONResponse(
            content=response.model_dump(),
            headers={"X-Copilot-Provider": "ccproxy"},
        )

    except Exception as e:
        logger.error(
            "models_listing_failed",
            error=str(e),
            exc_info=e,
        )
        raise HTTPException(
            status_code=500, detail=f"Failed to list models: {str(e)}"
        ) from e


@router.post("/v1/chat/completions", response_model=None)
async def chat_completions(
    request: Request,
    adapter: CopilotAdapterDep,
) -> JSONResponse | StreamingResponse | DeferredStreaming:
    """Create chat completions using Copilot."""
    try:
        # Parse request body
        body = await request.body()
        try:
            request_data = json.loads(body) if body else {}
            chat_request = CopilotChatRequest.model_validate(request_data)
        except json.JSONDecodeError as e:
            raise HTTPException(
                status_code=400, detail=f"Invalid JSON: {str(e)}"
            ) from e
        except ValueError as e:
            raise HTTPException(
                status_code=400, detail=f"Invalid request format: {str(e)}"
            ) from e

        logger.info(
            "chat_completion_request",
            model=chat_request.model,
            messages_count=len(chat_request.messages),
            stream=chat_request.stream,
        )

        # Delegate to adapter
        response = await adapter.handle_request(request, "/chat/completions", "POST")

        # Ensure return type compatibility
        if isinstance(response, Response) and not isinstance(
            response, JSONResponse | StreamingResponse
        ):
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
        # Type cast to satisfy mypy - we know response is one of the expected types
        return response  # type: ignore[no-any-return]

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "chat_completion_failed",
            error=str(e),
            exc_info=e,
        )
        raise HTTPException(
            status_code=500, detail=f"Chat completion failed: {str(e)}"
        ) from e


@router.post("/v1/embeddings", response_model=None)
async def create_embeddings(
    request: Request, adapter: CopilotAdapterDep
) -> JSONResponse:
    """Create embeddings using Copilot."""
    try:
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
        response = await adapter.handle_request(request, "/v1/embeddings", "POST")

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
async def get_usage_stats() -> JSONResponse:
    """Get Copilot usage statistics."""
    try:
        logger.debug("getting_usage_stats")

        # This endpoint requires upstream /copilot_internal/user data
        # Return error if not available since we don't have real data
        raise HTTPException(
            status_code=503,
            detail="Usage data not available - requires upstream /copilot_internal/user integration",
        )

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
        logger.debug("getting_token_status")

        if not adapter.oauth_provider:
            raise HTTPException(status_code=503, detail="OAuth provider not available")

        # Check authentication status
        is_authenticated = await adapter.oauth_provider.is_authenticated()
        token_info = await adapter.oauth_provider.get_token_info()

        if is_authenticated and token_info:
            profile = await adapter.oauth_provider.get_user_profile("")
            token_status = CopilotTokenStatus(
                valid=True,
                expires_at=token_info.copilot_expires_at,
                account_type=token_info.account_type,
                copilot_access=token_info.copilot_access,
                username=profile.display_name if profile else None,
            )
        else:
            token_status = CopilotTokenStatus(
                valid=False,
                expires_at=None,
                account_type="unknown",
                copilot_access=False,
                username=None,
            )

        logger.info(
            "token_status_retrieved",
            valid=token_status.valid,
            account_type=token_status.account_type,
            copilot_access=token_status.copilot_access,
        )

        return JSONResponse(
            content=token_status.model_dump(),
            headers={"X-Copilot-Provider": "ccproxy"},
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "token_status_failed",
            error=str(e),
            exc_info=e,
        )
        raise HTTPException(
            status_code=500, detail=f"Failed to get token status: {str(e)}"
        ) from e


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
            headers={"X-Copilot-Provider": "ccproxy"},
        )
