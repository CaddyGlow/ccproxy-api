"""OAuth authentication module for Anthropic OAuth login."""

from ccproxy.auth.oauth.models import (
    OAuthCallbackRequest,
    OAuthState,
    OAuthTokenRequest,
    OAuthTokenResponse,
)
from ccproxy.auth.oauth.routes import (
    get_oauth_flow_result,
    register_oauth_flow,
    router,
)


__all__ = [
    # Router
    "router",
    "register_oauth_flow",
    "get_oauth_flow_result",
    # Models
    "OAuthState",
    "OAuthCallbackRequest",
    "OAuthTokenRequest",
    "OAuthTokenResponse",
]
