"""OpenAI-compatible API routes."""

from .chat import router as chat_router
from .models import router as models_router


__all__ = ["chat_router", "models_router"]
