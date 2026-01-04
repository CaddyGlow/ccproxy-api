"""No-op auth manager for Claude SDK plugin."""

from typing import Any

from pydantic import SecretStr

from ccproxy.auth.oauth.protocol import StandardProfileFields
from ccproxy.plugins.oauth_claude.models import ClaudeCredentials, ClaudeOAuthToken


class NoOpAuthManager:
    """No-operation auth manager for Claude SDK.

    The SDK handles authentication internally through the CLI,
    so we don't need to manage auth headers.
    """

    async def get_access_token(self) -> str:
        """Return empty token since SDK handles auth internally."""
        return ""

    async def get_credentials(self) -> ClaudeCredentials:
        """Return dummy credentials since SDK handles auth internally."""
        # Create minimal credentials object with OAuthToken

        oauth_token = ClaudeOAuthToken(
            accessToken=SecretStr("sdk-managed"),
            refreshToken=SecretStr("sdk-managed"),
            expiresAt=None,
            scopes=[],
            subscriptionType="sdk",
        )
        return ClaudeCredentials(claudeAiOauth=oauth_token)

    async def is_authenticated(self) -> bool:
        """Always return True since SDK handles auth internally."""
        return True

    async def get_user_profile(self) -> StandardProfileFields | None:
        """Return ``None`` because the SDK does not surface profile metadata."""
        return None

    async def __aenter__(self) -> "NoOpAuthManager":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """No cleanup needed."""
        pass

    async def validate_credentials(self) -> bool:
        """Always return True since SDK handles auth internally."""
        return True

    def get_provider_name(self) -> str:
        """Get the provider name for logging."""
        return "claude-sdk"
