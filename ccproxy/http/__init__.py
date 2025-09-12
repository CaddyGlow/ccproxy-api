"""HTTP package for CCProxy - consolidated HTTP functionality."""

from .base import BaseHTTPHandler
from .client import (
    BaseProxyClient,
    HTTPClient,
    HTTPClientFactory,
    HTTPConnectionError,
    HTTPError,
    HTTPTimeoutError,
    HTTPXClient,
    get_proxy_url,
    get_ssl_context,
)
from .hooks import HookableHTTPClient
from .pool import HTTPPoolManager


__all__ = [
    # Client
    "HTTPClient",
    "HTTPClientFactory",
    "HTTPXClient",
    "BaseProxyClient",
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
