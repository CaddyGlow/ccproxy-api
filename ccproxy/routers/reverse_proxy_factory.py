"""Factory for creating reverse proxy routers with different transformation modes."""

from typing import Any

from fastapi import APIRouter, Depends, Request
from fastapi.responses import Response, StreamingResponse

from ccproxy.config.settings import get_settings
from ccproxy.middleware.auth import get_auth_dependency
from ccproxy.services.credentials import CredentialsManager
from ccproxy.services.reverse_proxy import ReverseProxyService
from ccproxy.utils.logging import get_logger


logger = get_logger(__name__)


def create_reverse_proxy_router(proxy_mode: str) -> APIRouter:
    """Create a reverse proxy router with specific transformation mode.

    Args:
        proxy_mode: Transformation mode - "minimal" or "full"

    Returns:
        APIRouter configured for the specified proxy mode
    """
    router = APIRouter(tags=["reverse-proxy"])

    # Initialize the reverse proxy service with settings and mode
    settings = get_settings()
    credentials_manager = CredentialsManager(config=settings.credentials)
    proxy_service = ReverseProxyService(
        target_base_url=settings.reverse_proxy_target_url,
        timeout=settings.reverse_proxy_timeout,
        proxy_mode=proxy_mode,
        credentials_manager=credentials_manager,
    )

    @router.api_route(
        "/{path:path}",
        methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"],
        response_model=None,
    )
    async def proxy_to_anthropic(
        request: Request, path: str, _: None = Depends(get_auth_dependency())
    ) -> Response | StreamingResponse:
        """Proxy requests to api.anthropic.com with mode-specific transformations.

        This endpoint handles all HTTP methods and applies transformations based on
        the configured proxy mode:
        - minimal: OAuth + basic headers only
        - full: Complete transformations (system prompt, format conversion, etc.)

        Args:
            request: FastAPI request object
            path: Path to forward to Anthropic API

        Returns:
            Response from the Anthropic API or StreamingResponse for streaming endpoints
        """
        # Extract request details
        method = request.method
        headers = dict(request.headers)
        query_params = dict(request.query_params) if request.query_params else None

        # Read request body
        body = None
        if method in ("POST", "PUT", "PATCH"):
            body = await request.body()

        # Ensure path starts with /
        if not path.startswith("/"):
            path = f"/{path}"

        logger.debug(f"Proxying {method} {path} in {proxy_mode} mode")

        # Proxy the request
        result = await proxy_service.proxy_request(
            method=method,
            path=path,
            headers=headers,
            body=body,
            query_params=query_params,
        )

        # Handle streaming response
        if isinstance(result, StreamingResponse):
            return result

        # Handle regular response
        status_code, response_headers, response_body = result

        return Response(
            content=response_body,
            status_code=status_code,
            headers=response_headers,
        )

    return router
