"""Anthropic API endpoints for Claude Code Proxy API Server."""

from typing import Any

from fastapi import APIRouter, Request


# Hybrid service imports removed - functionality moved to /sdk/ and /api/ routes


# Create the router for Anthropic API endpoints
router = APIRouter(tags=["anthropic"])


@router.post("/messages", response_model=None)
async def create_message(
    request: Request,
) -> dict[str, Any]:
    """Create a message using Claude AI.

    NOTE: This legacy endpoint is deprecated. Please use:
    - /sdk/v1/messages for Claude SDK endpoints
    - /api/v1/messages for proxy endpoints
    """
    return {
        "error": {
            "message": "Legacy endpoint deprecated. Use /sdk/v1/messages or /api/v1/messages instead",
            "type": "deprecated_endpoint",
            "code": "endpoint_moved",
        }
    }


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
