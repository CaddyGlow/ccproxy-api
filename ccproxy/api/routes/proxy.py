"""Proxy endpoints for Claude Code Proxy API Server."""

import json
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from ccproxy.adapters.openai.adapter import OpenAIAdapter
from ccproxy.api.dependencies import ProxyServiceDep


# Create the router for proxy endpoints
router = APIRouter(tags=["proxy"])


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
        query_params = dict(request.query_params) if request.query_params else None

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
                raise HTTPException(
                    status_code=status_code, detail=response_body.decode()
                )

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

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Internal server error: {str(e)}"
        ) from e


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
        query_params = dict(request.query_params) if request.query_params else None

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
                raise HTTPException(
                    status_code=status_code, detail=response_body.decode()
                )

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

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Internal server error: {str(e)}"
        ) from e


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
