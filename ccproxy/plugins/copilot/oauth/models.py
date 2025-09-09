"""GitHub Copilot-specific authentication models."""

from datetime import UTC, datetime
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


class CopilotOAuthToken(BaseModel):
    """OAuth token information for GitHub Copilot."""

    model_config = ConfigDict(
        populate_by_name=True, use_enum_values=True, arbitrary_types_allowed=True
    )

    access_token: SecretStr = Field(..., alias="access_token")
    token_type: str = Field(default="bearer", alias="token_type")
    expires_in: int | None = Field(None, alias="expires_in")
    refresh_token: SecretStr | None = Field(default=None, alias="refresh_token")
    scope: str = Field(default="read:user", alias="scope")
    created_at: int | None = Field(None, alias="created_at")

    @field_serializer("access_token", "refresh_token")
    def serialize_secret(self, value: SecretStr | None) -> str | None:
        """Serialize SecretStr to plain string for JSON output."""
        return value.get_secret_value() if value else None

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
        access_preview = (
            f"{access_token_str[:8]}...{access_token_str[-8:]}"
            if len(access_token_str) > 16
            else "***"
        )

        refresh_preview = "***"
        if self.refresh_token:
            refresh_token_str = self.refresh_token.get_secret_value()
            refresh_preview = (
                f"{refresh_token_str[:8]}...{refresh_token_str[-8:]}"
                if len(refresh_token_str) > 16
                else "***"
            )

        expires_at = (
            datetime.fromtimestamp(
                self.created_at + self.expires_in, tz=UTC
            ).isoformat()
            if self.expires_in and self.created_at
            else "None"
        )

        return (
            f"CopilotOAuthToken(access_token='{access_preview}', "
            f"refresh_token='{refresh_preview}', "
            f"expires_at={expires_at}, "
            f"scope='{self.scope}')"
        )

    @property
    def is_expired(self) -> bool:
        """Check if the token is expired."""
        if not self.expires_in or not self.created_at:
            # If no expiration info, assume not expired
            return False

        now = datetime.now(UTC).timestamp()
        expires_at = self.created_at + self.expires_in
        return now >= expires_at

    @property
    def expires_at_datetime(self) -> datetime:
        """Get expiration as datetime object."""
        if not self.expires_in or not self.created_at:
            # Return a far future date if no expiration info
            return datetime.fromtimestamp(2147483647, tz=UTC)  # Year 2038

        return datetime.fromtimestamp(self.created_at + self.expires_in, tz=UTC)


class CopilotTokenResponse(BaseModel):
    """Copilot token exchange response."""

    token: SecretStr = Field(..., description="Copilot service token")
    expires_at: str | None = Field(None, description="Token expiration time")
    refresh_in: int | None = Field(None, description="Refresh interval in seconds")

    @field_serializer("token")
    def serialize_secret(self, value: SecretStr) -> str:
        """Serialize SecretStr to plain string for JSON output."""
        return value.get_secret_value()

    @field_validator("token", mode="before")
    @classmethod
    def validate_token(cls, v: str | SecretStr) -> SecretStr:
        """Convert string values to SecretStr."""
        if isinstance(v, str):
            return SecretStr(v)
        return v


class CopilotCredentials(BaseModel):
    """Copilot credentials containing OAuth and Copilot tokens."""

    model_config = ConfigDict(
        populate_by_name=True, use_enum_values=True, arbitrary_types_allowed=True
    )

    oauth_token: CopilotOAuthToken = Field(..., description="GitHub OAuth token")
    copilot_token: CopilotTokenResponse | None = Field(
        default=None, description="Copilot service token"
    )
    account_type: str = Field(
        default="individual",
        description="Account type (individual/business/enterprise)",
    )
    created_at: int = Field(
        default_factory=lambda: int(datetime.now(UTC).timestamp()),
        description="Timestamp when credentials were created",
    )
    updated_at: int = Field(
        default_factory=lambda: int(datetime.now(UTC).timestamp()),
        description="Timestamp when credentials were last updated",
    )

    def __repr__(self) -> str:
        """Safe representation without exposing secrets."""
        copilot_status = "present" if self.copilot_token else "missing"
        return (
            f"CopilotCredentials(oauth_token={repr(self.oauth_token)}, "
            f"copilot_token={copilot_status}, "
            f"account_type='{self.account_type}')"
        )

    def is_expired(self) -> bool:
        """Check if credentials are expired (BaseCredentials protocol)."""
        return self.oauth_token.is_expired

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage (BaseCredentials protocol)."""
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CopilotCredentials":
        """Create from dictionary (BaseCredentials protocol)."""
        return cls.model_validate(data)

    def refresh_updated_at(self) -> None:
        """Update the updated_at timestamp."""
        self.updated_at = int(datetime.now(UTC).timestamp())


class CopilotProfileInfo(BaseProfileInfo):
    """GitHub profile information for Copilot users."""

    # Required fields from BaseProfileInfo
    account_id: str = Field(..., description="GitHub user ID")
    provider_type: str = Field(default="copilot", description="Provider type")

    # GitHub-specific fields
    login: str = Field(..., description="GitHub username")
    name: str | None = Field(None, description="Full name")
    avatar_url: str | None = Field(None, description="Avatar URL")
    html_url: str | None = Field(None, description="Profile URL")
    copilot_plan: str | None = Field(None, description="Copilot subscription plan")
    copilot_access: bool = Field(default=False, description="Has Copilot access")

    @computed_field
    def computed_display_name(self) -> str:
        """Display name for UI."""
        if self.display_name:
            return self.display_name
        return self.name or self.login


class CopilotTokenInfo(BaseTokenInfo):
    """Token information for Copilot credentials."""

    provider: Literal["copilot"] = "copilot"
    oauth_expires_at: datetime | None = None
    copilot_expires_at: datetime | None = None
    account_type: str = "individual"
    copilot_access: bool = False

    @computed_field
    def computed_is_expired(self) -> bool:
        """Check if any token is expired."""
        now = datetime.now(UTC)

        # Check OAuth token expiration
        if self.oauth_expires_at and now >= self.oauth_expires_at:
            return True

        # Check Copilot token expiration if available
        return bool(self.copilot_expires_at and now >= self.copilot_expires_at)

    @computed_field
    def computed_display_name(self) -> str:
        """Display name for UI."""
        return f"GitHub Copilot ({self.account_type})"


class DeviceCodeResponse(BaseModel):
    """GitHub device code authorization response."""

    device_code: str = Field(..., description="Device verification code")
    user_code: str = Field(..., description="User verification code")
    verification_uri: str = Field(..., description="Verification URL")
    expires_in: int = Field(..., description="Code expiration time in seconds")
    interval: int = Field(..., description="Polling interval in seconds")


class DeviceTokenPollResponse(BaseModel):
    """Response from device code token polling."""

    access_token: str | None = Field(None, description="Access token if authorized")
    token_type: str | None = Field(None, description="Token type")
    scope: str | None = Field(None, description="Granted scopes")
    error: str | None = Field(None, description="Error code if any")
    error_description: str | None = Field(None, description="Error description")
    error_uri: str | None = Field(None, description="Error URI")

    @property
    def is_pending(self) -> bool:
        """Check if authorization is still pending."""
        return self.error == "authorization_pending"

    @property
    def is_slow_down(self) -> bool:
        """Check if we should slow down polling."""
        return self.error == "slow_down"

    @property
    def is_expired(self) -> bool:
        """Check if device code has expired."""
        return self.error == "expired_token"

    @property
    def is_denied(self) -> bool:
        """Check if user denied authorization."""
        return self.error == "access_denied"

    @property
    def is_success(self) -> bool:
        """Check if authorization was successful."""
        return self.access_token is not None and self.error is None
