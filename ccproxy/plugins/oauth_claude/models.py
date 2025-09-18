"""Claude-specific authentication models."""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    computed_field,
    field_serializer,
    field_validator,
)

from ccproxy.auth.models.base import BaseProfileInfo, BaseTokenInfo


class ClaudeOAuthToken(BaseModel):
    """OAuth token information from Claude credentials."""

    model_config = ConfigDict(
        populate_by_name=True, use_enum_values=True, arbitrary_types_allowed=True
    )

    access_token: SecretStr = Field(..., alias="accessToken")
    refresh_token: SecretStr = Field(..., alias="refreshToken")
    expires_at: int | None = Field(None, alias="expiresAt")
    scopes: list[str] = Field(default_factory=list)
    subscription_type: str | None = Field(None, alias="subscriptionType")

    @field_serializer("access_token", "refresh_token")
    def serialize_secret(self, value: SecretStr) -> str:
        """Serialize SecretStr to plain string for JSON output."""
        return value.get_secret_value() if value else ""

    @field_validator("access_token", "refresh_token", mode="before")
    @classmethod
    def validate_tokens(cls, v: str | SecretStr | None) -> SecretStr | None:
        """Convert string values to SecretStr."""
        if v is None:
            return None
        if isinstance(v, str):
            return SecretStr(v)
        return v

    def __repr__(self) -> str:
        """Safe string representation that masks sensitive tokens."""
        access_token_str = self.access_token.get_secret_value()
        refresh_token_str = self.refresh_token.get_secret_value()

        access_preview = (
            f"{access_token_str[:8]}...{access_token_str[-8:]}"
            if len(access_token_str) > 16
            else "***"
        )
        refresh_preview = (
            f"{refresh_token_str[:8]}...{refresh_token_str[-8:]}"
            if len(refresh_token_str) > 16
            else "***"
        )

        expires_at = (
            datetime.fromtimestamp(self.expires_at / 1000, tz=UTC).isoformat()
            if self.expires_at is not None
            else "None"
        )
        return (
            f"OAuthToken(access_token='{access_preview}', "
            f"refresh_token='{refresh_preview}', "
            f"expires_at={expires_at}, "
            f"scopes={self.scopes}, "
            f"subscription_type='{self.subscription_type}')"
        )

    @property
    def is_expired(self) -> bool:
        """Check if the token is expired."""
        if self.expires_at is None:
            # If no expiration info, assume not expired for backward compatibility
            return False
        now = datetime.now(UTC).timestamp() * 1000  # Convert to milliseconds
        return now >= self.expires_at

    @property
    def expires_at_datetime(self) -> datetime:
        """Get expiration as datetime object."""
        if self.expires_at is None:
            # Return a far future date if no expiration info
            return datetime.fromtimestamp(2147483647, tz=UTC)  # Year 2038
        return datetime.fromtimestamp(self.expires_at / 1000, tz=UTC)


class ClaudeCredentials(BaseModel):
    """Claude credentials from the credentials file."""

    model_config = ConfigDict(
        populate_by_name=True, use_enum_values=True, arbitrary_types_allowed=True
    )

    claude_ai_oauth: ClaudeOAuthToken = Field(..., alias="claudeAiOauth")

    def __repr__(self) -> str:
        """Safe string representation that masks sensitive tokens."""
        return f"ClaudeCredentials(claude_ai_oauth={repr(self.claude_ai_oauth)})"

    def is_expired(self) -> bool:
        """Check if the credentials are expired.

        Returns:
            True if expired, False otherwise
        """
        return self.claude_ai_oauth.is_expired

    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        """Override model_dump to use by_alias=True by default."""
        kwargs.setdefault("by_alias", True)
        return super().model_dump(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage.

        Returns:
            Dictionary representation
        """
        return self.model_dump(mode="json", exclude_none=True)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ClaudeCredentials":
        """Create from dictionary.

        Args:
            data: Dictionary containing credential data

        Returns:
            ClaudeCredentials instance
        """
        return cls.model_validate(data)


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
        """Compute subscription type from stored profile info.

        Attempts to read the Claude profile file ("~/.claude/.account.json")
        and derive the subscription from account flags:
          - "max" if has_claude_max is true
          - "pro" if has_claude_pro is true
          - "free" otherwise

        Falls back to the token's own subscription_type if profile is unavailable.
        """
        # Lazy, best-effort read of local profile data; keep this non-fatal.
        try:
            profile_path = Path.home() / ".claude" / ".account.json"
            if profile_path.exists():
                with profile_path.open("r") as f:
                    data = json.load(f)
                account = data.get("account", {})
                if account.get("has_claude_max") is True:
                    return "max"
                if account.get("has_claude_pro") is True:
                    return "pro"
                # If account is present but neither flag set, assume free tier
                if account:
                    return "free"
        except Exception:
            # Ignore any profile read/parse errors and fall back
            pass

        # Fallback to stored token field for backward compatibility
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
