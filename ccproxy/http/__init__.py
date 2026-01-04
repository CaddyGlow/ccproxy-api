"""HTTP package for CCProxy - consolidated HTTP functionality."""

from .base import BaseHTTPHandler
from .client import (
    HTTPClientFactory,
    HTTPConnectionError,
    HTTPError,
    HTTPTimeoutError,
    get_proxy_url,
    get_ssl_context,
)
from .hooks import HookableHTTPClient
from .pool import HTTPPoolManager


__all__ = [
    # Client
    "HTTPClientFactory",
    "HookableHTTPClient",
    # Errors
    "HTTPError",
    "HTTPTimeoutError",
    "HTTPConnectionError",
    # Services
    "HTTPPoolManager",
    "BaseHTTPHandler",
    # Utils
    "get_proxy_url",
    "get_ssl_context",
]
