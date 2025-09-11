"""Configuration models for GitHub Copilot plugin."""

from pydantic import BaseModel, Field

from ccproxy.models.provider import ProviderConfig


class CopilotOAuthConfig(BaseModel):
    """OAuth-specific configuration for GitHub Copilot."""

    "https://api.individual.githubcopilot.com/chat/completions"
    client_id: str = Field(
        default="Iv1.b507a08c87ecfe98",
        description="GitHub Copilot OAuth client ID",
    )
    authorize_url: str = Field(
        default="https://github.com/login/device/code",
        description="GitHub OAuth device code authorization endpoint",
    )
    token_url: str = Field(
        default="https://github.com/login/oauth/access_token",
        description="GitHub OAuth token endpoint",
    )
    copilot_token_url: str = Field(
        default="https://api.github.com/copilot_internal/v2/token",
        description="GitHub Copilot token exchange endpoint",
    )
    scopes: list[str] = Field(
        default_factory=lambda: ["read:user"],
        description="OAuth scopes to request from GitHub",
    )
    use_pkce: bool = Field(
        default=True,
        description="Whether to use PKCE flow for security",
    )
    request_timeout: int = Field(
        default=30,
        description="Timeout in seconds for OAuth requests",
        ge=1,
        le=300,
    )
    callback_timeout: int = Field(
        default=300,
        description="Timeout in seconds for OAuth callback",
        ge=60,
        le=600,
    )
    callback_port: int = Field(
        default=8080,
        description="Port for OAuth callback server",
        ge=1024,
        le=65535,
    )
    redirect_uri: str | None = Field(
        default=None,
        description="OAuth redirect URI (auto-generated from callback_port if not set)",
    )

    def get_redirect_uri(self) -> str:
        """Return redirect URI, auto-generated from callback_port when unset."""
        if self.redirect_uri:
            return self.redirect_uri
        return f"http://localhost:{self.callback_port}/callback"


class CopilotProviderConfig(ProviderConfig):
    """Provider-specific configuration for GitHub Copilot API."""

    name: str = "copilot"
    base_url: str = "https://api.githubcopilot.com"
    supports_streaming: bool = True
    requires_auth: bool = True
    auth_type: str | None = "oauth"

    # Claude API specific settings
    enabled: bool = True
    priority: int = 5  # Higher priority than SDK-based approach
    default_max_tokens: int = 4096

    # Supported models
    models: list[str] = []

    account_type: str = Field(
        default="individual",
        description="Account type: individual, business, or enterprise",
    )
    request_timeout: int = Field(
        default=30,
        description="Timeout for API requests in seconds",
        ge=1,
        le=300,
    )
    max_retries: int = Field(
        default=3,
        description="Maximum number of retries for failed requests",
        ge=0,
        le=10,
    )
    retry_delay: float = Field(
        default=1.0,
        description="Base delay between retries in seconds",
        ge=0.1,
        le=60.0,
    )

    api_headers: dict[str, str] = Field(
        default_factory=lambda: {
            "Content-Type": "application/json",
            "Copilot-Integration-Id": "vscode-chat",
            "Editor-Version": "vscode/1.85.0",
            "Editor-Plugin-Version": "copilot-chat/0.26.7",
            "User-Agent": "GitHubCopilotChat/0.26.7",
            "X-GitHub-Api-Version": "2025-04-01",
        },
        description="Default headers for Copilot API requests",
    )


class CopilotConfig(CopilotProviderConfig):
    """Complete configuration for GitHub Copilot plugin."""

    oauth: CopilotOAuthConfig = Field(
        default_factory=CopilotOAuthConfig,
        description="OAuth authentication configuration",
    )
