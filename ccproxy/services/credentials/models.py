"""Data models for credentials."""

from datetime import UTC, datetime

from pydantic import BaseModel, Field


class OAuthToken(BaseModel):
    """OAuth token information from Claude credentials."""

    access_token: str = Field(..., alias="accessToken")
    refresh_token: str = Field(..., alias="refreshToken")
    expires_at: int = Field(..., alias="expiresAt")
    scopes: list[str] = Field(default_factory=list)
    subscription_type: str | None = Field(None, alias="subscriptionType")

    @property
    def is_expired(self) -> bool:
        """Check if the token is expired."""
        now = datetime.now(UTC).timestamp() * 1000  # Convert to milliseconds
        return now >= self.expires_at

    @property
    def expires_at_datetime(self) -> datetime:
        """Get expiration as datetime object."""
        return datetime.fromtimestamp(self.expires_at / 1000, tz=UTC)


class OrganizationInfo(BaseModel):
    """Organization information from OAuth API."""

    uuid: str
    name: str
    organization_type: str | None = None
    billing_type: str | None = None
    rate_limit_tier: str | None = None


class AccountInfo(BaseModel):
    """Account information from OAuth API."""

    uuid: str
    email: str
    full_name: str | None = None
    display_name: str | None = None
    has_claude_max: bool | None = None
    has_claude_pro: bool | None = None

    @property
    def email_address(self) -> str:
        """Compatibility property for email_address."""
        return self.email


class UserProfile(BaseModel):
    """User profile information from Anthropic OAuth API."""

    organization: OrganizationInfo | None = None
    account: AccountInfo | None = None


class ClaudeCredentials(BaseModel):
    """Claude credentials from the credentials file."""

    claude_ai_oauth: OAuthToken = Field(..., alias="claudeAiOauth")
