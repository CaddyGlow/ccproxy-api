"""Codex plugin-specific configuration settings."""

from pydantic import BaseModel, Field, field_validator

from ccproxy.models.provider import ProviderConfig


class OAuthSettings(BaseModel):
    """OAuth configuration for OpenAI authentication."""

    base_url: str = Field(
        default="https://auth.openai.com",
        description="OpenAI OAuth base URL",
    )

    client_id: str = Field(
        default="app_EMoamEEZ73f0CkXaXp7hrann",
        description="OpenAI OAuth client ID",
    )

    scopes: list[str] = Field(
        default_factory=lambda: ["openid", "profile", "email", "offline_access"],
        description="OAuth scopes to request",
    )

    @field_validator("base_url")
    @classmethod
    def validate_base_url(cls, v: str) -> str:
        """Validate OAuth base URL format."""
        if not v.startswith(("http://", "https://")):
            raise ValueError("OAuth base URL must start with http:// or https://")
        return v.rstrip("/")


class CodexSettings(ProviderConfig):
    """Codex plugin configuration extending base ProviderConfig."""

    # Base ProviderConfig fields will be inherited

    # Codex-specific OAuth settings
    oauth: OAuthSettings = Field(
        default_factory=OAuthSettings,
        description="OAuth configuration settings",
    )

    callback_port: int = Field(
        default=1455,
        ge=1024,
        le=65535,
        description="Port for OAuth callback server (1024-65535)",
    )

    redirect_uri: str = Field(
        default="http://localhost:1455/auth/callback",
        description="OAuth redirect URI (auto-generated from callback_port if not set)",
    )

    verbose_logging: bool = Field(
        default=False,
        description="Enable verbose logging for Codex operations",
    )

    # Override base_url default for Codex
    base_url: str = Field(
        default="https://chatgpt.com/backend-api/codex",
        description="OpenAI Codex API base URL",
    )

    # Set defaults for inherited fields
    name: str = Field(default="codex", description="Provider name")
    supports_streaming: bool = Field(
        default=True, description="Whether the provider supports streaming"
    )
    requires_auth: bool = Field(
        default=True, description="Whether the provider requires authentication"
    )
    auth_type: str | None = Field(
        default="oauth", description="Authentication type (bearer, api_key, etc.)"
    )
    models: list[str] = Field(
        default_factory=lambda: ["gpt-5"],
        description="List of supported models",
    )

    @field_validator("base_url")
    @classmethod
    def validate_base_url(cls, v: str) -> str:
        """Validate Codex base URL format."""
        if not v.startswith(("http://", "https://")):
            raise ValueError("Codex base URL must start with http:// or https://")
        return v.rstrip("/")

    @field_validator("redirect_uri")
    @classmethod
    def validate_redirect_uri(cls, v: str) -> str:
        """Validate redirect URI format."""
        if not v.startswith(("http://", "https://")):
            raise ValueError("Redirect URI must start with http:// or https://")
        return v

    @field_validator("callback_port")
    @classmethod
    def validate_callback_port(cls, v: int) -> int:
        """Validate callback port range."""
        if not (1024 <= v <= 65535):
            raise ValueError("Callback port must be between 1024 and 65535")
        return v

    def get_redirect_uri(self) -> str:
        """Get the redirect URI, auto-generating if needed."""
        if (
            self.redirect_uri
            and self.redirect_uri
            != f"http://localhost:{self.callback_port}/auth/callback"
        ):
            return self.redirect_uri
        return f"http://localhost:{self.callback_port}/auth/callback"
