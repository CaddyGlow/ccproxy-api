"""Configuration for credentials and OAuth."""

import os

from pydantic import BaseModel, Field


def _get_default_storage_paths() -> list[str]:
    """Get default storage paths, with test override support."""
    # Allow tests to override credential paths
    if os.getenv("CCPROXY_TEST_MODE") == "true":
        # Use a test-specific location that won't pollute real credentials
        return [
            "/tmp/ccproxy-test/.claude/.credentials.json",
            "/tmp/ccproxy-test/.config/claude/.credentials.json",
        ]

    return [
        "~/.config/ccproxy/credentials.json",  # Primary location in app config
        "~/.claude/.credentials.json",  # Legacy location
        "~/.config/claude/.credentials.json",  # Alternative legacy location
    ]


class OAuthConfig(BaseModel):
    """OAuth configuration settings."""

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


class CredentialsConfig(BaseModel):
    """Configuration for credentials management."""

    storage_paths: list[str] = Field(
        default_factory=lambda: _get_default_storage_paths(),
        description="Paths to search for credentials files",
    )
    oauth: OAuthConfig = Field(
        default_factory=OAuthConfig,
        description="OAuth configuration",
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
