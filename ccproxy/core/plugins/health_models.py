"""Common provider plugin health detail models.

These models standardize the `details` payload returned by provider plugins
in their health checks, enabling consistent inspection across plugins.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class CLIHealth(BaseModel):
    """Standardized CLI health information for a provider plugin."""

    available: bool = Field(description="Whether the CLI is available")
    status: str = Field(description="CLI status string from plugin detector")
    version: str | None = Field(default=None, description="Detected CLI version")
    path: str | None = Field(default=None, description="Resolved CLI binary path")


class AuthHealth(BaseModel):
    """Standardized authentication health information."""

    configured: bool = Field(description="Whether auth is configured for this plugin")
    token_available: bool | None = Field(
        default=None, description="Valid, non-expired token is available"
    )
    token_expired: bool | None = Field(default=None, description="Token is expired")
    account_id: str | None = Field(default=None, description="Associated account id")
    expires_at: str | None = Field(default=None, description="Token expiry ISO time")
    error: str | None = Field(default=None, description="Auth error or reason text")


class ConfigHealth(BaseModel):
    """Standardized configuration summary for a provider plugin."""

    model_count: int | None = Field(default=None, description="Configured model count")
    supports_openai_format: bool | None = Field(
        default=None, description="Whether OpenAI-compatible format is supported"
    )
    verbose_logging: bool | None = Field(
        default=None, description="Whether plugin verbose logging is enabled"
    )
    extra: dict[str, Any] | None = Field(
        default=None, description="Additional provider-specific configuration"
    )


class ProviderHealthDetails(BaseModel):
    """Top-level standardized provider health details payload."""

    provider: str = Field(description="Provider plugin name")
    enabled: bool = Field(description="Whether this plugin is enabled")
    base_url: str | None = Field(default=None, description="Provider base URL")
    cli: CLIHealth | None = Field(default=None, description="CLI health")
    auth: AuthHealth | None = Field(default=None, description="Auth health")
    config: ConfigHealth | None = Field(default=None, description="Config summary")
