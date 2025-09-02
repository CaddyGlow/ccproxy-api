"""OAuth configuration for Claude OAuth plugin."""

from pydantic import BaseModel, Field


class ClaudeOAuthConfig(BaseModel):
    """OAuth-specific configuration for Claude."""

    enabled: bool = Field(
        default=True,
        description="Enablded the plugin",
    )

    base_url: str = Field(
        default="https://console.anthropic.com",
        description="Base URL for OAuth API endpoints",
    )
    token_url: str = Field(
        default="https://console.anthropic.com/v1/oauth/token",
        description="OAuth token endpoint URL",
    )
    authorize_url: str = Field(
        default="https://claude.ai/oauth/authorize",
        description="OAuth authorization endpoint URL",
    )
    profile_url: str = Field(
        default="https://api.anthropic.com/api/oauth/profile",
        description="OAuth profile endpoint URL",
    )
    client_id: str = Field(
        default="9d1c250a-e61b-44d9-88ed-5944d1962f5e",
        description="OAuth client ID",
    )
    redirect_uri: str | None = Field(
        # default="https://console.anthropic.com/oauth/code/callback",
        default=None,
        # default="http://localhost:54545/callback",
        description="OAuth redirect URI",
    )
    scopes: list[str] = Field(
        default_factory=lambda: [
            "org:create_api_key",
            "user:profile",
            "user:inference",
        ],
        description="OAuth scopes to request",
    )
    headers: dict[str, str] = Field(
        default_factory=lambda: {
            # "anthropic-beta": "oauth-2025-04-20",
            # "User-Agent": "Claude-Code/1.0.43",  # Match default user agent  in config
        },
        description="Additional headers for OAuth requests",
    )
    request_timeout: int = Field(
        default=30,
        description="Timeout in seconds for OAuth requests",
    )
    callback_timeout: int = Field(
        default=300,
        description="Timeout in seconds for OAuth callback",
        ge=60,
        le=600,
    )
    callback_port: int = Field(
        default=35593,
        # default=54545,
        description="Port for OAuth callback server",
        ge=1024,
        le=65535,
    )
    use_pkce: bool = Field(
        default=True,
        description="Whether to use PKCE flow (required for Claude OAuth)",
    )

    def get_redirect_uri(self) -> str:
        """Return redirect URI, auto-generated from callback_port when unset.

        Uses the standard plugin callback path: `/callback`.
        """
        if self.redirect_uri:
            return self.redirect_uri
        return f"http://localhost:{self.callback_port}/callback"
