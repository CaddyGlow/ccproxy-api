"""Base models for authentication across all providers."""

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field, computed_field


class BaseTokenInfo(BaseModel):
    """Base model for token information across all providers.

    This abstract base provides a common interface for token operations
    while allowing each provider to maintain its specific implementation.
    """

    @computed_field
    def access_token_value(self) -> str:
        """Get the actual access token string.
        Must be implemented by provider-specific subclasses.
        """
        raise NotImplementedError

    @computed_field
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
