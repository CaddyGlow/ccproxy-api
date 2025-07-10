"""Router modules for the Claude Proxy API."""

from ccproxy.auth.oauth import router as oauth_router

from .claudecode.anthropic import router as anthropic_router
from .claudecode.openai import router as openai_router
from .reverse_proxy_factory import create_reverse_proxy_router


__all__ = [
    "anthropic_router",
    "openai_router",
    "oauth_router",
    "create_reverse_proxy_router",
]
