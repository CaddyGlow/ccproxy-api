"""API routes for Claude Code Proxy API Server."""

# from ccproxy.api.routes.auth import router as auth_router  # Module doesn't exist
from ccproxy.api.routes.claude import router as claude_router
from ccproxy.api.routes.health import router as health_router
from ccproxy.api.routes.metrics import router as metrics_router
from ccproxy.api.routes.proxy import router as proxy_router


__all__ = [
    # "auth_router",  # Module doesn't exist
    "claude_router",
    "health_router",
    "metrics_router",
    "proxy_router",
]
