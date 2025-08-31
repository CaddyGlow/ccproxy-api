"""Base models for authentication across all providers."""

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field, computed_field


class OrganizationInfo(BaseModel):
    """Organization information from OAuth API."""

    uuid: str
    name: str
    organization_type: str | None = None
    billing_type: str | None = None
    rate_limit_tier: str | None = None


class AccountInfo(BaseModel):
    """Account information from OAuth API.

    Core fields are required, provider-specific fields go in extras.
    """

    uuid: str
    email: str = ""  # Make optional with default empty string for providers that don't provide it
    full_name: str | None = None
    display_name: str | None = None
    extras: dict[str, Any] = Field(
        default_factory=dict, description="Provider-specific extra fields"
    )

    @property
    def email_address(self) -> str:
        """Compatibility property for email_address."""
        return self.email

    def has_subscription(self) -> bool:
        """Check if user has any subscription. Override in provider-specific implementations."""
        return False

    def get_subscription_level(self) -> str | None:
        """Get subscription level. Override in provider-specific implementations."""
        return None

    @property
    def has_claude_max(self) -> bool | None:
        """Compatibility property for Claude-specific field."""
        return self.extras.get("has_claude_max")

    @property
    def has_claude_pro(self) -> bool | None:
        """Compatibility property for Claude-specific field."""
        return self.extras.get("has_claude_pro")


class UserProfile(BaseModel):
    """User profile information from OAuth API."""

    organization: OrganizationInfo | None = None
    account: AccountInfo | None = None


class BaseTokenInfo(BaseModel):
    """Base model for token information across all providers.

    This abstract base provides a common interface for token operations
    while allowing each provider to maintain its specific implementation.
    """

    @computed_field  # type: ignore[prop-decorator]
    @property
    def access_token_value(self) -> str:
        """Get the actual access token string.
        Must be implemented by provider-specific subclasses.
        """
        raise NotImplementedError

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_expired(self) -> bool:
        """Check if token is expired.
        Uses the expires_at_datetime property for comparison.
        """
        now = datetime.now(UTC)
        return now >= self.expires_at_datetime

    @property
    def expires_at_datetime(self) -> datetime:
        """Get expiration as datetime object.
        Must be implemented by provider-specific subclasses.
        """
        raise NotImplementedError

    @property
    def refresh_token_value(self) -> str | None:
        """Get refresh token if available.
        Default returns None, override if provider supports refresh.
        """
        return None


class BaseProfileInfo(BaseModel):
    """Base model for user profile information across all providers.

    Provides common fields with a flexible extras dict for
    provider-specific data.
    """

    account_id: str
    provider_type: str

    # Common fields with sensible defaults
    email: str = ""
    display_name: str | None = None

    # All provider-specific data stored here
    # This preserves all information for future use
    extras: dict[str, Any] = Field(
        default_factory=dict,
        description="Provider-specific data (JWT claims, API responses, etc.)",
    )
