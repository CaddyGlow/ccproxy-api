"""OAuth configuration for Claude OAuth plugin."""

from pydantic import BaseModel, Field


class ClaudeOAuthConfig(BaseModel):
    """Configuration for Claude OAuth provider."""

    client_id: str = Field(
        default="anthropic_client_production",
        description="OAuth client ID for Claude",
    )
    redirect_uri: str = Field(
        default="http://localhost:9999/oauth/claude-api/callback",
        description="OAuth redirect URI",
    )
    base_url: str = Field(
        default="https://claude.ai",
        description="Base URL for Claude OAuth",
    )
    authorize_url: str = Field(
        default="https://claude.ai/api/auth/oauth_login",
        description="Authorization endpoint URL",
    )
    token_url: str = Field(
        default="https://claude.ai/api/auth/oauth_exchange_code",
        description="Token exchange endpoint URL",
    )
    scopes: list[str] = Field(
        default_factory=lambda: ["read", "write", "claude_pro"],
        description="OAuth scopes to request",
    )
    beta_version: str = Field(
        default="2024-01-01",
        description="Anthropic API beta version",
    )
    user_agent: str = Field(
        default="ccproxy-claude-oauth/1.0",
        description="User agent for OAuth requests",
    )
    use_pkce: bool = Field(
        default=False,
        description="Whether to use PKCE flow (Claude doesn't require it)",
    )
