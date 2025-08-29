"""
Backward compatibility module for request context.

All request context functionality has been moved to ccproxy.core.request_context.
This file provides re-exports for backward compatibility.
"""

from ccproxy.core.request_context import (
    ProxyRequestContext,
    RequestContext,
    create_proxy_context,
    get_context_tracker,
    request_context,
    timed_operation,
    tracked_request_context,
)


__all__ = [
    "ProxyRequestContext",
    "RequestContext",
    "create_proxy_context",
    "get_context_tracker",
    "request_context",
    "timed_operation",
    "tracked_request_context",
]
