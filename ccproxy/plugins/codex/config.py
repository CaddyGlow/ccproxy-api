"""Codex plugin-specific configuration settings."""

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from ccproxy.core.constants import (
    FORMAT_ANTHROPIC_MESSAGES,
    FORMAT_OPENAI_CHAT,
    FORMAT_OPENAI_RESPONSES,
)
from ccproxy.models.provider import ModelCard, ModelMappingRule, ProviderConfig
from ccproxy.plugins.codex.model_defaults import (
    DEFAULT_CODEX_MODEL_CARDS,
    DEFAULT_CODEX_MODEL_MAPPINGS,
)


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

    # NEW: Auth manager override support
    auth_manager: str | None = Field(
        default=None,
        description="Override auth manager name (e.g., 'oauth_codex_lb' for load balancing)",
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
    model_mappings: list[ModelMappingRule] = Field(
        default_factory=lambda: [
            rule.model_copy(deep=True) for rule in DEFAULT_CODEX_MODEL_MAPPINGS
        ],
        description="List of client-to-upstream model mapping rules",
    )
    models_endpoint: list[ModelCard] = Field(
        default_factory=lambda: [
            card.model_copy(deep=True) for card in DEFAULT_CODEX_MODEL_CARDS
        ],
        description="Model metadata served via the /models endpoint",
    )

    supported_input_formats: list[str] = Field(
        default_factory=lambda: [
            FORMAT_OPENAI_RESPONSES,
            FORMAT_OPENAI_CHAT,
            FORMAT_ANTHROPIC_MESSAGES,
        ],
        description="List of supported input formats",
    )
    preferred_upstream_mode: Literal["streaming", "non_streaming"] = Field(
        default="streaming", description="Preferred upstream mode for requests"
    )
    buffer_non_streaming: bool = Field(
        default=True, description="Whether to buffer non-streaming requests"
    )
    enable_format_registry: bool = Field(
        default=True, description="Whether to enable format adapter registry"
    )

    # Detection configuration
    detection_home_mode: Literal["temp", "home"] = Field(
        default="temp",
        description="Home directory mode for CLI detection: 'temp' uses temporary directory, 'home' uses actual user HOME",
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
