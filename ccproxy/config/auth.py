"""Authentication and credentials configuration."""

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator


def _get_default_storage_paths() -> list[Path]:
    """Get default storage paths"""
    return [
        Path("~/.config/ccproxy/credentials.json"),
        Path("~/.claude/.credentials.json"),
        Path("~/.config/claude/.credentials.json"),
    ]


class OAuthSettings(BaseModel):
    """OAuth-specific settings."""

    base_url: str = Field(
        default="https://console.anthropic.com",
        description="Base URL for OAuth API endpoints",
    )
    beta_version: str = Field(
        default="oauth-2025-04-20",
        description="OAuth beta version header",
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
    redirect_uri: str = Field(
        default="http://localhost:54545/callback",
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
    request_timeout: int = Field(
        default=30,
        description="Timeout in seconds for OAuth requests",
    )
    user_agent: str = Field(
        default="Claude-Code/1.0.43",
        description="User agent string for OAuth requests",
    )
    callback_timeout: int = Field(
        default=300,
        description="Timeout in seconds for OAuth callback",
        ge=60,
        le=600,
    )
    callback_port: int = Field(
        default=54545,
        description="Port for OAuth callback server",
        ge=1024,
        le=65535,
    )


class CredentialStorageSettings(BaseModel):
    """Settings for credential storage locations."""

    storage_paths: list[Path] = Field(
        default_factory=lambda: _get_default_storage_paths(),
        description="Paths to search for credentials files",
    )
    auto_refresh: bool = Field(
        default=True,
        description="Automatically refresh expired tokens",
    )
    refresh_buffer_seconds: int = Field(
        default=300,
        description="Refresh token this many seconds before expiry",
        ge=0,
    )


class AuthSettings(BaseModel):
    """Combined authentication and credentials configuration."""

    oauth: OAuthSettings = Field(
        default_factory=OAuthSettings,
        description="OAuth configuration",
    )
    storage: CredentialStorageSettings = Field(
        default_factory=CredentialStorageSettings,
        description="Credential storage configuration",
    )

    @field_validator("oauth", mode="before")
    @classmethod
    def validate_oauth(cls, v: Any) -> Any:
        """Validate and convert OAuth configuration."""
        if v is None:
            return OAuthSettings()

        # If it's already an OAuthSettings instance, return as-is
        if isinstance(v, OAuthSettings):
            return v

        # If it's a dict, create OAuthSettings from it
        if isinstance(v, dict):
            return OAuthSettings(**v)

        # Try to convert to dict if possible
        if hasattr(v, "model_dump"):
            return OAuthSettings(**v.model_dump())
        elif hasattr(v, "__dict__"):
            return OAuthSettings(**v.__dict__)

        return v

    @field_validator("storage", mode="before")
    @classmethod
    def validate_storage(cls, v: Any) -> Any:
        """Validate and convert storage configuration."""
        if v is None:
            return CredentialStorageSettings()

        # If it's already a CredentialStorageSettings instance, return as-is
        if isinstance(v, CredentialStorageSettings):
            return v

        # If it's a dict, create CredentialStorageSettings from it
        if isinstance(v, dict):
            return CredentialStorageSettings(**v)

        # Try to convert to dict if possible
        if hasattr(v, "model_dump"):
            return CredentialStorageSettings(**v.model_dump())
        elif hasattr(v, "__dict__"):
            return CredentialStorageSettings(**v.__dict__)

        return v
