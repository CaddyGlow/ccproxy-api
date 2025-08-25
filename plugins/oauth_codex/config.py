"""OAuth configuration for Codex OAuth plugin."""

from pydantic import BaseModel, Field


class CodexOAuthConfig(BaseModel):
    """Configuration for Codex/OpenAI OAuth provider."""

    client_id: str = Field(
        default="openai_client_production",
        description="OAuth client ID for OpenAI",
    )
    redirect_uri: str = Field(
        default="http://localhost:9999/oauth/codex/callback",
        description="OAuth redirect URI",
    )
    base_url: str = Field(
        default="https://auth0.openai.com",
        description="Base URL for OpenAI OAuth",
    )
    authorize_url: str = Field(
        default="https://auth0.openai.com/authorize",
        description="Authorization endpoint URL",
    )
    token_url: str = Field(
        default="https://auth0.openai.com/oauth/token",
        description="Token exchange endpoint URL",
    )
    scopes: list[str] = Field(
        default_factory=lambda: ["openid", "profile", "email", "offline_access"],
        description="OAuth scopes to request",
    )
    audience: str = Field(
        default="https://api.openai.com",
        description="OAuth audience parameter",
    )
    user_agent: str = Field(
        default="ccproxy-codex-oauth/1.0",
        description="User agent for OAuth requests",
    )
    use_pkce: bool = Field(
        default=True,
        description="Whether to use PKCE flow (OpenAI requires it)",
    )
