"""OpenAI-compatible models endpoint."""

from fastapi import APIRouter

from claude_proxy.models.openai_models import OpenAIModelsResponse


router = APIRouter()


@router.get("/models", response_model=OpenAIModelsResponse)
async def list_models() -> OpenAIModelsResponse:
    """List available OpenAI-compatible models."""
    return OpenAIModelsResponse.create_default()
