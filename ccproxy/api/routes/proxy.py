"""Proxy endpoints for Claude Code Proxy API Server."""

from typing import Any

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from ccproxy.api.dependencies import get_proxy_service
from ccproxy.middleware.auth import verify_token
from ccproxy.services.proxy_service import ProxyService


# Create the router for proxy endpoints
router = APIRouter(prefix="/proxy", tags=["proxy"])


@router.get("/status")
async def proxy_status():
    """Get proxy status."""
    return {"status": "proxy endpoint available"}


# Reverse proxy endpoints that handle /unclaude prefix
@router.api_route(
    "/unclaude/{path:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"],
    response_model=None,
)
async def reverse_proxy_handler(
    request: Request,
    path: str,
    proxy_service: ProxyService = Depends(get_proxy_service),
    credentials: HTTPAuthorizationCredentials | None = Depends(
        HTTPBearer(auto_error=False)
    ),
) -> Any:
    """Handle reverse proxy requests for /unclaude endpoints.

    This endpoint strips the /unclaude prefix and forwards requests
    to the target API with proper authentication and transformations.
    """
    # Verify authentication
    verify_token(credentials, request)
    # Get request body
    body = await request.body()

    # Get headers and query params
    headers = dict(request.headers)
    query_params = dict(request.query_params) if request.query_params else None

    # Forward to proxy service (path already has the /unclaude prefix stripped)
    response = await proxy_service.handle_request(
        method=request.method,
        path=f"/{path}",  # Add leading slash since path doesn't include it
        headers=headers,
        body=body,
        query_params=query_params,
    )

    # Handle streaming response
    if isinstance(response, StreamingResponse):
        return response

    # Handle regular response
    status_code, response_headers, response_body = response

    # Create appropriate response
    from fastapi import Response

    return Response(
        content=response_body,
        status_code=status_code,
        headers=response_headers,
    )
