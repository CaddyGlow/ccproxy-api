"""OpenAI-compatible API endpoints for Claude Code Proxy API Server."""

from typing import Any

from fastapi import APIRouter, Request


# Hybrid service imports removed - functionality moved to /sdk/ and /api/ routes


# Create the router for OpenAI-compatible API endpoints
router = APIRouter(tags=["openai"])


@router.post("/chat/completions", response_model=None)
async def create_chat_completion(
    request: Request,
) -> dict[str, Any]:
    """Create a chat completion using Claude AI with OpenAI-compatible format.

    NOTE: This legacy endpoint is deprecated. Please use:
    - /sdk/v1/chat/completions for Claude SDK endpoints
    - /api/v1/chat/completions for proxy endpoints
    """
    return {
        "error": {
            "message": "Legacy endpoint deprecated. Use /sdk/v1/chat/completions or /api/v1/chat/completions instead",
            "type": "deprecated_endpoint",
            "code": "endpoint_moved",
        }
    }


@router.get("/models")
async def list_models() -> dict[str, Any]:
    """List available models in OpenAI format."""
    return {
        "object": "list",
        "data": [
            {
                "id": "gpt-4o",
                "object": "model",
                "created": 1677652288,
                "owned_by": "anthropic",
            },
            {
                "id": "gpt-4o-mini",
                "object": "model",
                "created": 1677652288,
                "owned_by": "anthropic",
            },
        ],
    }


@router.get("/status")
async def openai_status() -> dict[str, str]:
    """Get OpenAI API status."""
    return {"status": "openai endpoint available"}
