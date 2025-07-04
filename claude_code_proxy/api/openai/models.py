"""OpenAI-compatible models endpoint."""

from fastapi import APIRouter, Depends

from claude_code_proxy.middleware.auth import get_auth_dependency
from claude_code_proxy.models.openai_models import OpenAIModelsResponse


router = APIRouter()


@router.get("/models", response_model=OpenAIModelsResponse)
async def list_models(_: None = Depends(get_auth_dependency())) -> OpenAIModelsResponse:
    """List available OpenAI-compatible models."""
    return OpenAIModelsResponse.create_default()
