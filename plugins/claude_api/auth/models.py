"""Claude-specific authentication models."""

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import computed_field

from ccproxy.auth.models import ClaudeCredentials
from ccproxy.auth.models.base import BaseProfileInfo, BaseTokenInfo


class ClaudeTokenWrapper(BaseTokenInfo):
    """Wrapper for Claude credentials that adds computed properties.

    This wrapper maintains the original ClaudeCredentials structure
    while providing a unified interface through BaseTokenInfo.
    """

    # Embed the original credentials to preserve JSON schema
    credentials: ClaudeCredentials

    @computed_field  # type: ignore[prop-decorator]
    @property
    def access_token_value(self) -> str:
        """Extract access token from Claude OAuth structure."""
        return self.credentials.claude_ai_oauth.access_token.get_secret_value()

    @property
    def refresh_token_value(self) -> str | None:
        """Extract refresh token from Claude OAuth structure."""
        token = self.credentials.claude_ai_oauth.refresh_token
        return token.get_secret_value() if token else None

    @property
    def expires_at_datetime(self) -> datetime:
        """Convert Claude's millisecond timestamp to datetime."""
        expires_at = self.credentials.claude_ai_oauth.expires_at
        if not expires_at:
            # No expiration means token doesn't expire
            return datetime.max.replace(tzinfo=UTC)
        # Claude stores expires_at in milliseconds
        return datetime.fromtimestamp(expires_at / 1000, tz=UTC)

    @property
    def subscription_type(self) -> str | None:
        """Get subscription type from OAuth token."""
        return self.credentials.claude_ai_oauth.subscription_type

    @property
    def scopes(self) -> list[str]:
        """Get OAuth scopes."""
        return self.credentials.claude_ai_oauth.scopes


class ClaudeProfileInfo(BaseProfileInfo):
    """Claude-specific profile information from API.

    Created from the /api/organizations/me endpoint response.
    """

    provider_type: Literal["claude-api"] = "claude-api"

    @classmethod
    def from_api_response(cls, data: dict[str, Any]) -> "ClaudeProfileInfo":
        """Create profile from Claude API response.

        Args:
            data: Response from /api/organizations/me endpoint

        Returns:
            ClaudeProfileInfo instance with all data preserved
        """
        # Extract account information if present
        account = data.get("account", {})
        organization = data.get("organization", {})

        # Extract common fields for easy access
        account_id = account.get("uuid", "")
        email = account.get("email", "")
        display_name = account.get("full_name")

        # Store entire response in extras for complete information
        # This includes: has_claude_pro, has_claude_max, organization details, etc.
        return cls(
            account_id=account_id,
            email=email,
            display_name=display_name,
            extras=data,  # Preserve complete API response
        )

    @property
    def has_claude_pro(self) -> bool | None:
        """Check if user has Claude Pro subscription."""
        account = self.extras.get("account", {})
        value = account.get("has_claude_pro")
        return bool(value) if value is not None else None

    @property
    def has_claude_max(self) -> bool | None:
        """Check if user has Claude Max subscription."""
        account = self.extras.get("account", {})
        value = account.get("has_claude_max")
        return bool(value) if value is not None else None

    @property
    def organization_name(self) -> str | None:
        """Get organization name if available."""
        org = self.extras.get("organization", {})
        name = org.get("name")
        return str(name) if name is not None else None
