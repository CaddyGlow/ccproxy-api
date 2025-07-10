"""API middleware for Claude Code Proxy API Server."""

from ccproxy.api.middleware.cors import get_cors_config, setup_cors_middleware
from ccproxy.api.middleware.errors import setup_error_handlers


__all__ = [
    "setup_cors_middleware",
    "get_cors_config",
    "setup_error_handlers",
]
