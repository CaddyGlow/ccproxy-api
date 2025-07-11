"""Anthropic API endpoints for Claude Code Proxy API Server."""

from collections.abc import AsyncIterator
from typing import Any, Union

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from ccproxy.api.dependencies import get_hybrid_service
from ccproxy.services.hybrid_service import HybridService


# Create the router for Anthropic API endpoints
router = APIRouter(tags=["anthropic"])


@router.post("/messages", response_model=None)
async def create_message(
    request: Request,
    hybrid_service: HybridService = Depends(get_hybrid_service),
) -> StreamingResponse | dict[str, Any]:
    """Create a message using Claude AI.

    This endpoint handles Anthropic API format requests and forwards them
    to the appropriate service (SDK or proxy) based on request characteristics.
    """
    try:
        # Get request body
        body = await request.body()

        # Get headers and query params
        headers = dict(request.headers)
        query_params = dict(request.query_params) if request.query_params else None

        # Handle the request using hybrid service
        response = await hybrid_service.handle_request(
            method=request.method,
            path=request.url.path,
            headers=headers,
            body=body,
            query_params=query_params,
        )

        # Return appropriate response type
        if hasattr(response, "__aiter__"):
            # Streaming response
            async def stream_generator() -> AsyncIterator[bytes]:
                async for chunk in response:  # type: ignore[union-attr]
                    if isinstance(chunk, dict):
                        yield f"data: {chunk.get('data', '')}\n\n".encode()
                    else:
                        yield str(chunk).encode()

            return StreamingResponse(
                stream_generator(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                },
            )
        else:
            # Regular response
            status_code, response_headers, response_body = response
            if status_code >= 400:
                raise HTTPException(
                    status_code=status_code, detail=response_body.decode()
                )

            # Parse JSON response
            import json

            response_data = json.loads(response_body.decode())
            return response_data  # type: ignore[no-any-return]

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Internal server error: {str(e)}"
        ) from e


@router.get("/models")
async def list_models() -> dict[str, Any]:
    """List available Claude models."""
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
        ],
    }


@router.get("/status")
async def anthropic_status() -> dict[str, str]:
    """Get Anthropic API status."""
    return {"status": "anthropic endpoint available"}
