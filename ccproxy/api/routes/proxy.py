"""Proxy endpoints for Claude Code Proxy API Server."""

import json
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from ccproxy.adapters.openai.adapter import OpenAIAdapter
from ccproxy.api.dependencies import ProxyServiceDep
from ccproxy.core.errors import (
    AuthenticationError,
    ModelNotFoundError,
    TimeoutError,
    ValidationError,
)
from ccproxy.core.logging import get_structlog_logger


# Create the router for proxy endpoints
router = APIRouter(tags=["proxy"])

# Create structured logger
logger = get_structlog_logger(__name__)


def _parse_error_response(status_code: int, response_body: bytes) -> HTTPException:
    """Parse error response and map to appropriate custom exceptions."""
    try:
        error_data = json.loads(response_body.decode())
        error_message = error_data.get("error", {}).get("message", "Unknown error")
        error_type = error_data.get("error", {}).get("type", "unknown_error")

        # Log error details for debugging
        logger.error(
            "API error response",
            status_code=status_code,
            error_type=error_type,
            error_message=error_message,
        )

        # Map specific error types to clean user-facing messages
        if (
            status_code == 404
            or "model" in error_message.lower()
            and "not found" in error_message.lower()
        ):
            clean_message = (
                "Invalid model name"
                if "model" in error_message.lower()
                else error_message
            )
            return HTTPException(
                status_code=404,
                detail={
                    "error": {"type": "invalid_request_error", "message": clean_message}
                },
            )
        elif status_code == 400:
            return HTTPException(
                status_code=400,
                detail={
                    "error": {"type": "invalid_request_error", "message": error_message}
                },
            )
        elif status_code == 401:
            return HTTPException(
                status_code=401,
                detail={
                    "error": {
                        "type": "authentication_error",
                        "message": "Invalid authentication",
                    }
                },
            )
        elif status_code == 429:
            return HTTPException(
                status_code=429,
                detail={
                    "error": {
                        "type": "rate_limit_error",
                        "message": "Rate limit exceeded",
                    }
                },
            )
        else:
            return HTTPException(
                status_code=status_code,
                detail={"error": {"type": "api_error", "message": error_message}},
            )
    except json.JSONDecodeError:
        # If response is not JSON, use raw response as message
        logger.error(
            "Non-JSON error response",
            status_code=status_code,
            response_body=response_body.decode()[:200],  # Log first 200 chars
        )
        return HTTPException(
            status_code=status_code,
            detail={"error": {"type": "api_error", "message": response_body.decode()}},
        )


@router.post("/v1/chat/completions", response_model=None)
async def create_openai_chat_completion(
    request: Request,
    proxy_service: ProxyServiceDep,
) -> StreamingResponse | dict[str, Any]:
    """Create a chat completion using Claude AI with OpenAI-compatible format.

    This endpoint handles OpenAI API format requests and forwards them
    directly to Claude via the proxy service.
    """
    try:
        # Get request body
        body = await request.body()

        # Get headers and query params
        headers = dict(request.headers)
        query_params: dict[str, str | list[str]] | None = (
            dict(request.query_params) if request.query_params else None
        )

        # Handle the request using proxy service directly
        response = await proxy_service.handle_request(
            method=request.method,
            path=request.url.path,
            headers=headers,
            body=body,
            query_params=query_params,
        )

        # Return appropriate response type
        if isinstance(response, StreamingResponse):
            # Already a streaming response
            return response
        else:
            # Tuple response - handle regular response
            status_code, response_headers, response_body = response
            if status_code >= 400:
                raise _parse_error_response(status_code, response_body)

            # Check if this is a streaming response based on content-type
            content_type = response_headers.get("content-type", "")
            if "text/event-stream" in content_type:
                # Return as streaming response
                async def stream_generator() -> AsyncIterator[bytes]:
                    # Split the SSE data into chunks
                    for line in response_body.decode().split("\n"):
                        if line.strip():
                            yield f"{line}\n".encode()

                return StreamingResponse(
                    stream_generator(),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                    },
                )
            else:
                # Parse JSON response
                response_data = json.loads(response_body.decode())

                # Convert Anthropic response back to OpenAI format for /chat/completions
                openai_adapter = OpenAIAdapter()
                openai_response = openai_adapter.adapt_response(response_data)
                return openai_response

    except json.JSONDecodeError:
        logger.error("Failed to parse JSON response", path=request.url.path)
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "type": "internal_error",
                    "message": "Invalid response format",
                }
            },
        ) from None
    except Exception as e:
        logger.error(
            "Unexpected error in OpenAI endpoint",
            error=str(e),
            path=request.url.path,
            method=request.method,
        )
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "type": "internal_error",
                    "message": "An internal error occurred",
                }
            },
        ) from None


@router.post("/v1/messages", response_model=None)
async def create_anthropic_message(
    request: Request,
    proxy_service: ProxyServiceDep,
) -> StreamingResponse | dict[str, Any]:
    """Create a message using Claude AI with Anthropic format.

    This endpoint handles Anthropic API format requests and forwards them
    directly to Claude via the proxy service.
    """
    try:
        # Get request body
        body = await request.body()

        # Get headers and query params
        headers = dict(request.headers)
        query_params: dict[str, str | list[str]] | None = (
            dict(request.query_params) if request.query_params else None
        )

        # Handle the request using proxy service directly
        response = await proxy_service.handle_request(
            method=request.method,
            path=request.url.path,
            headers=headers,
            body=body,
            query_params=query_params,
        )

        # Return appropriate response type
        if isinstance(response, StreamingResponse):
            # Already a streaming response
            return response
        else:
            # Tuple response - handle regular response
            status_code, response_headers, response_body = response
            if status_code >= 400:
                raise _parse_error_response(status_code, response_body)

            # Check if this is a streaming response based on content-type
            content_type = response_headers.get("content-type", "")
            if "text/event-stream" in content_type:
                # Return as streaming response
                async def stream_generator() -> AsyncIterator[bytes]:
                    # Split the SSE data into chunks
                    for line in response_body.decode().split("\n"):
                        if line.strip():
                            yield f"{line}\n".encode()

                return StreamingResponse(
                    stream_generator(),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                    },
                )
            else:
                # Parse JSON response
                response_data = json.loads(response_body.decode())
                return response_data  # type: ignore[no-any-return]

    except json.JSONDecodeError:
        logger.error("Failed to parse JSON response", path=request.url.path)
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "type": "internal_error",
                    "message": "Invalid response format",
                }
            },
        ) from None
    except Exception as e:
        logger.error(
            "Unexpected error in Anthropic endpoint",
            error=str(e),
            path=request.url.path,
            method=request.method,
        )
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "type": "internal_error",
                    "message": "An internal error occurred",
                }
            },
        ) from None


@router.get("/models")
async def list_models() -> dict[str, Any]:
    """List available models for the proxy API."""
    return {
        "object": "list",
        "data": [
            {
                "id": "claude-3-5-sonnet-20241022",
                "object": "model",
                "created": 1677652288,
                "owned_by": "anthropic",
            },
            {
                "id": "claude-3-5-haiku-20241022",
                "object": "model",
                "created": 1677652288,
                "owned_by": "anthropic",
            },
            {
                "id": "claude-3-opus-20240229",
                "object": "model",
                "created": 1677652288,
                "owned_by": "anthropic",
            },
        ],
    }


@router.get("/status")
async def proxy_status() -> dict[str, str]:
    """Get proxy status."""
    return {"status": "proxy API available", "version": "1.0.0"}


@router.get("/health")
async def proxy_health() -> dict[str, str]:
    """Health check endpoint for the proxy API."""
    return {"status": "healthy", "service": "claude-proxy-api"}
