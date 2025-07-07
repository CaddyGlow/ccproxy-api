"""Reverse proxy router for /unclaude endpoints."""

from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import Response, StreamingResponse

from claude_code_proxy.config.settings import get_settings
from claude_code_proxy.services.reverse_proxy import ReverseProxyService
from claude_code_proxy.utils.logging import get_logger


logger = get_logger(__name__)

router = APIRouter(tags=["reverse-proxy"])

# Initialize the reverse proxy service with settings
settings = get_settings()
proxy_service = ReverseProxyService(
    target_base_url=settings.reverse_proxy_target_url,
    timeout=settings.reverse_proxy_timeout,
)


@router.api_route(
    "/unclaude/{path:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"],
    response_model=None,
)
async def proxy_to_anthropic(
    request: Request, path: str
) -> Response | StreamingResponse:
    """Proxy requests from /unclaude/* to api.anthropic.com/*.

    This endpoint handles all HTTP methods and transforms:
    - Path: /unclaude/v1/messages -> https://api.anthropic.com/v1/messages
    - Headers: Injects OAuth authentication and required headers
    - Body: Adds Claude Code system prompt and maps model aliases

    Args:
        request: FastAPI request object
        path: Path after /unclaude/ prefix

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

    # Show what the transformed path will be for logging
    from claude_code_proxy.services.request_transformer import RequestTransformer

    transformer = RequestTransformer()
    transformed_path = transformer.transform_path(path)

    logger.debug(f"Proxying {method} /unclaude{path} -> {method} {transformed_path}")

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
