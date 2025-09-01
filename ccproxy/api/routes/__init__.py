"""API routes for CCProxy API Server."""

# from ccproxy.api.routes.auth import router as auth_router  # Module doesn't exist
from ccproxy.api.routes.health import router as health_router


# proxy routes are now handled by plugin system


__all__ = [
    # "auth_router",  # Module doesn't exist
    "health_router",
    # Metrics, logs, and dashboard routes are provided by plugins now
    # "proxy_router", # Removed - handled by plugin system
]
