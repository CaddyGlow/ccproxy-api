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
from .plugin_handler import PluginHTTPHandler
from .pool import HTTPPoolManager
from .processor import RequestProcessor


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
    "RequestProcessor",
    "PluginHTTPHandler",
    "HTTPPoolManager",
    "BaseHTTPHandler",
    # Utils
    "get_proxy_url",
    "get_ssl_context",
]
