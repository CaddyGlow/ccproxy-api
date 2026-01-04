"""Public router shim for OAuth flows."""

from ccproxy.auth.oauth.registry import OAuthProviderProtocol
from ccproxy.auth.oauth.routes import get_oauth_flow_result, register_oauth_flow, router


__all__ = [
    "router",
    "register_oauth_flow",
    "get_oauth_flow_result",
    "OAuthProviderProtocol",
]
