"""OpenAI Codex-specific configuration settings."""

from pydantic import BaseModel, Field


class CodexOAuthConfig(BaseModel):
    """OAuth-specific configuration for OpenAI Codex."""

    enabled: bool = Field(
        default=True,
        description="Enable the plugin",
    )

    # Core OAuth endpoints and identifiers (aligns with Claude config structure)
    base_url: str = Field(
        default="https://auth.openai.com",
        description="Base URL for OAuth API endpoints",
    )
    token_url: str = Field(
        default="https://auth.openai.com/oauth/token",
        description="OAuth token endpoint URL",
    )
    authorize_url: str = Field(
        default="https://auth.openai.com/oauth/authorize",
        description="OAuth authorization endpoint URL",
    )
    profile_url: str = Field(
        default="https://api.openai.com/oauth/profile",
        description="OAuth profile endpoint URL",
    )
    client_id: str = Field(
        default="app_EMoamEEZ73f0CkXaXp7hrann",
        description="OpenAI OAuth client ID",
    )
    redirect_uri: str | None = Field(
        default=None,
        description="OAuth redirect URI (auto-generated from callback_port if not set)",
    )
    scopes: list[str] = Field(
        default_factory=lambda: [
            "openid",
            "profile",
            "email",
            "offline_access",
        ],
        description="OAuth scopes to request",
    )

    # Additional request configuration (mirrors Claude config shape)
    headers: dict[str, str] = Field(
        default_factory=lambda: {
            "User-Agent": "Codex-Code/1.0.43",  # Match default user agent in config
        },
        description="Additional headers for OAuth requests",
    )
    # Optional audience parameter for auth requests (OpenAI specific)
    audience: str = Field(
        default="https://api.openai.com/v1",
        description="OAuth audience parameter for OpenAI",
    )
    # Convenience user agent string (mirrors headers[\"User-Agent\"]) for typed access
    user_agent: str = Field(
        default="Codex-Code/1.0.43",
        description="User-Agent header value for OAuth requests",
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
        default=1455,
        description="Port for OAuth callback server",
        ge=1024,
        le=65535,
    )

    def get_redirect_uri(self) -> str:
        """Return redirect URI, auto-generated from callback_port when unset.

        Uses the standard plugin callback path: `/auth/callback`.
        """
        if self.redirect_uri:
            return self.redirect_uri
        return f"http://localhost:{self.callback_port}/auth/callback"

    use_pkce: bool = Field(
        default=True,
        description="Whether to use PKCE flow (OpenAI requires it)",
    )
