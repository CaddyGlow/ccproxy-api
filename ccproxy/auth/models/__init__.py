"""Data models for authentication."""

from datetime import UTC, datetime
from typing import Any

import jwt
from pydantic import BaseModel, Field, SecretStr, field_validator


class OAuthToken(BaseModel):
    """OAuth token information from Claude credentials."""

    access_token: SecretStr = Field(..., alias="accessToken")
    refresh_token: SecretStr = Field(..., alias="refreshToken")
    expires_at: int | None = Field(None, alias="expiresAt")
    scopes: list[str] = Field(default_factory=list)
    subscription_type: str | None = Field(None, alias="subscriptionType")
    token_type: str = Field(default="Bearer", alias="tokenType")

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
            f"subscription_type='{self.subscription_type}', "
            f"token_type='{self.token_type}')"
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
    """User profile information from Anthropic OAuth API."""

    organization: OrganizationInfo | None = None
    account: AccountInfo | None = None


class ClaudeCredentials(BaseModel):
    """Claude credentials from the credentials file."""

    claude_ai_oauth: OAuthToken = Field(..., alias="claudeAiOauth")

    def __repr__(self) -> str:
        """Safe string representation that masks sensitive tokens."""
        return f"ClaudeCredentials(claude_ai_oauth={repr(self.claude_ai_oauth)})"


class ValidationResult(BaseModel):
    """Result of credentials validation."""

    valid: bool
    expired: bool | None = None
    credentials: ClaudeCredentials | None = None
    path: str | None = None


class OpenAICredentials(BaseModel):
    """OpenAI authentication credentials model."""

    access_token: str = Field(..., description="OpenAI access token (JWT)")
    refresh_token: str = Field(..., description="OpenAI refresh token")
    id_token: str | None = Field(None, description="OpenAI ID token (JWT)")
    expires_at: datetime = Field(..., description="Token expiration timestamp")
    account_id: str = Field(..., description="OpenAI account ID extracted from token")
    active: bool = Field(default=True, description="Whether credentials are active")

    @field_validator("expires_at", mode="before")
    @classmethod
    def parse_expires_at(cls, v: Any) -> datetime:
        """Parse expiration timestamp."""
        if isinstance(v, datetime):
            # Ensure timezone-aware datetime
            if v.tzinfo is None:
                return v.replace(tzinfo=UTC)
            return v

        if isinstance(v, str):
            # Handle ISO format strings
            try:
                dt = datetime.fromisoformat(v.replace("Z", "+00:00"))
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=UTC)
                return dt
            except ValueError as e:
                raise ValueError(f"Invalid datetime format: {v}") from e

        if isinstance(v, int | float):
            # Handle Unix timestamps
            return datetime.fromtimestamp(v, tz=UTC)

        raise ValueError(f"Cannot parse datetime from {type(v)}: {v}")

    @field_validator("account_id", mode="before")
    @classmethod
    def extract_account_id(cls, v: Any, info: Any) -> str:
        """Extract account ID from tokens if not provided.

        Prioritizes chatgpt_account_id (UUID format) from id_token,
        falls back to auth0 sub claim if not found.
        """
        if isinstance(v, str) and v:
            return v

        # Try to extract from id_token first (contains chatgpt_account_id UUID)
        id_token = None
        if hasattr(info, "data") and info.data and isinstance(info.data, dict):
            id_token = info.data.get("id_token")

        if id_token and isinstance(id_token, str):
            account_id = cls._extract_account_id_from_token(id_token, "id_token")
            if account_id:
                return account_id

        # Try to extract from access_token
        access_token = None
        if hasattr(info, "data") and info.data and isinstance(info.data, dict):
            access_token = info.data.get("access_token")

        if access_token and isinstance(access_token, str):
            account_id = cls._extract_account_id_from_token(
                access_token, "access_token"
            )
            if account_id:
                return account_id

        raise ValueError(
            "account_id is required and could not be extracted from tokens"
        )

    @classmethod
    def _extract_account_id_from_token(cls, token: str, token_type: str) -> str | None:
        """Helper to extract account ID from a JWT token."""
        import structlog

        logger = structlog.get_logger(__name__)

        try:
            # Decode JWT without verification to extract claims
            decoded = jwt.decode(token, options={"verify_signature": False})

            # Look for OpenAI auth claims with chatgpt_account_id (proper UUID)
            if "https://api.openai.com/auth" in decoded:
                auth_claims = decoded["https://api.openai.com/auth"]
                if isinstance(auth_claims, dict):
                    # Use chatgpt_account_id if available (this is the proper UUID)
                    if "chatgpt_account_id" in auth_claims and isinstance(
                        auth_claims["chatgpt_account_id"], str
                    ):
                        account_id = auth_claims["chatgpt_account_id"]
                        logger.info(
                            f"Using chatgpt_account_id from {token_type}",
                            account_id=account_id,
                        )
                        return account_id

                    # Also check organization_id as a fallback
                    if "organization_id" in auth_claims and isinstance(
                        auth_claims["organization_id"], str
                    ):
                        org_id = auth_claims["organization_id"]
                        if not org_id.startswith("auth0|"):
                            logger.info(
                                f"Using organization_id from {token_type}",
                                org_id=org_id,
                            )
                            return org_id

            # Check top-level claims
            if "account_id" in decoded and isinstance(decoded["account_id"], str):
                return decoded["account_id"]
            elif "org_id" in decoded and isinstance(decoded["org_id"], str):
                # Check if org_id looks like a UUID (not auth0|xxx format)
                org_id = decoded["org_id"]
                if not org_id.startswith("auth0|"):
                    return org_id
            elif (
                token_type == "access_token"
                and "sub" in decoded
                and isinstance(decoded["sub"], str)
            ):
                # Fallback to auth0 sub (not ideal but maintains compatibility)
                sub = decoded["sub"]
                logger.warning(
                    "Falling back to auth0 sub as account_id - consider updating to use chatgpt_account_id",
                    sub=sub,
                )
                return sub

        except (jwt.DecodeError, jwt.InvalidTokenError, KeyError, ValueError) as e:
            logger.debug(f"{token_type}_decode_failed", error=str(e))

        return None

    def is_expired(self) -> bool:
        """Check if the access token is expired."""
        now = datetime.now(UTC)
        return now >= self.expires_at

    def expires_in_seconds(self) -> int:
        """Get seconds until token expires."""
        now = datetime.now(UTC)
        delta = self.expires_at - now
        return max(0, int(delta.total_seconds()))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "access_token": self.access_token,
            "refresh_token": self.refresh_token,
            "id_token": self.id_token,
            "expires_at": self.expires_at.isoformat(),
            "account_id": self.account_id,
            "active": self.active,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OpenAICredentials":
        """Create from dictionary."""
        return cls(**data)
