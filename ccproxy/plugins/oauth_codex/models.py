"""OpenAI-specific authentication models."""

from datetime import UTC, datetime
from typing import Any, Literal

import jwt
from pydantic import (
    BaseModel,
    Field,
    SecretStr,
    computed_field,
    field_serializer,
    field_validator,
)

from ccproxy.auth.models.base import BaseProfileInfo, BaseTokenInfo
from ccproxy.core.logging import get_plugin_logger


logger = get_plugin_logger()


class OpenAITokens(BaseModel):
    """Nested token structure from OpenAI OAuth."""

    id_token: SecretStr = Field(..., description="OpenAI ID token (JWT)")
    access_token: SecretStr = Field(..., description="OpenAI access token (JWT)")
    refresh_token: SecretStr = Field(..., description="OpenAI refresh token")
    account_id: str = Field(..., description="OpenAI account ID")

    @field_serializer("id_token", "access_token", "refresh_token")
    def serialize_secret(self, value: SecretStr) -> str:
        """Serialize SecretStr to plain string for JSON output."""
        return value.get_secret_value() if value else ""

    @field_validator("id_token", "access_token", "refresh_token", mode="before")
    @classmethod
    def validate_tokens(cls, v: str | SecretStr | None) -> SecretStr | None:
        """Convert string values to SecretStr."""
        if v is None:
            return None
        if isinstance(v, str):
            return SecretStr(v)
        return v


class OpenAICredentials(BaseModel):
    """OpenAI authentication credentials model matching actual auth file schema."""

    OPENAI_API_KEY: str | None = Field(
        None, description="Legacy API key (usually null)"
    )
    tokens: OpenAITokens = Field(..., description="OAuth token information")
    last_refresh: str = Field(..., description="Last refresh timestamp as ISO string")
    active: bool = Field(default=True, description="Whether credentials are active")
    # No legacy compatibility shims; callers must provide nested `tokens` structure

    @property
    def access_token(self) -> str:
        """Get access token from nested structure."""
        return self.tokens.access_token.get_secret_value()

    @property
    def refresh_token(self) -> str:
        """Get refresh token from nested structure."""
        return self.tokens.refresh_token.get_secret_value()

    @property
    def id_token(self) -> str:
        """Get ID token from nested structure."""
        return self.tokens.id_token.get_secret_value()

    @property
    def account_id(self) -> str:
        """Get account ID from nested structure."""
        return self.tokens.account_id

    @property
    def expires_at(self) -> datetime:
        """Extract expiration from access token JWT."""
        try:
            # Decode JWT without verification to extract 'exp' claim
            decoded = jwt.decode(
                self.tokens.access_token.get_secret_value(),
                options={"verify_signature": False},
            )
            exp_timestamp = decoded.get("exp")
            if exp_timestamp:
                return datetime.fromtimestamp(exp_timestamp, tz=UTC)
        except (jwt.DecodeError, jwt.InvalidTokenError, KeyError, ValueError) as e:
            logger.debug("Failed to extract expiration from access token", error=str(e))

        # Fallback to a reasonable default if we can't decode
        return datetime.now(UTC).replace(hour=23, minute=59, second=59)

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
        """Convert to dictionary for storage.

        Implements BaseCredentials protocol.
        """
        return {
            "OPENAI_API_KEY": self.OPENAI_API_KEY,
            "tokens": {
                "id_token": self.tokens.id_token.get_secret_value(),
                "access_token": self.tokens.access_token.get_secret_value(),
                "refresh_token": self.tokens.refresh_token.get_secret_value(),
                "account_id": self.tokens.account_id,
            },
            "last_refresh": self.last_refresh,
            "active": self.active,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OpenAICredentials":
        """Create from dictionary.

        Implements BaseCredentials protocol.
        """
        return cls(**data)


class OpenAITokenWrapper(BaseTokenInfo):
    """Wrapper for OpenAI credentials that adds computed properties.

    This wrapper maintains the original OpenAICredentials structure
    while providing a unified interface through BaseTokenInfo.
    """

    # Embed the original credentials to preserve JSON schema
    credentials: OpenAICredentials

    @computed_field
    def access_token_value(self) -> str:
        """Get access token (now SecretStr in OpenAI)."""
        return self.credentials.access_token

    @property
    def refresh_token_value(self) -> str | None:
        """Get refresh token."""
        return self.credentials.refresh_token

    @property
    def expires_at_datetime(self) -> datetime:
        """Get expiration (already a datetime in OpenAI)."""
        return self.credentials.expires_at

    @property
    def account_id(self) -> str:
        """Get account ID (extracted from JWT by validator)."""
        return self.credentials.account_id

    @property
    def id_token(self) -> str | None:
        """Get ID token if available."""
        return self.credentials.id_token


class OpenAIProfileInfo(BaseProfileInfo):
    """OpenAI-specific profile extracted from JWT tokens.

    OpenAI embeds profile information in JWT claims rather
    than providing a separate API endpoint.
    """

    provider_type: Literal["openai"] = "openai"

    @classmethod
    def from_token(cls, credentials: OpenAICredentials) -> "OpenAIProfileInfo":
        """Extract profile from JWT token claims.

        Args:
            credentials: OpenAI credentials containing JWT tokens

        Returns:
            OpenAIProfileInfo with all JWT claims preserved
        """
        # Prefer id_token as it has more claims, fallback to access_token
        token_to_decode = credentials.id_token or credentials.access_token

        try:
            # Decode without verification to extract claims
            claims = jwt.decode(token_to_decode, options={"verify_signature": False})
            logger.debug(
                "Extracted JWT claims", num_claims=len(claims), category="auth"
            )
        except Exception as e:
            logger.warning("failed_to_decode_jwt_token", error=str(e), category="auth")
            claims = {}

        # Use the account_id already extracted by OpenAICredentials validator
        account_id = credentials.account_id

        # Extract common fields if present in claims
        email = claims.get("email", "")
        display_name = claims.get("name") or claims.get("given_name")

        # Store ALL JWT claims in extras for complete information
        # This includes: sub, aud, iss, exp, iat, org_id, chatgpt_account_id, etc.
        return cls(
            account_id=account_id,
            email=email,
            display_name=display_name,
            extras=claims,  # Preserve all JWT claims
        )

    @property
    def chatgpt_account_id(self) -> str | None:
        """Get ChatGPT account ID from JWT claims."""
        auth_claims = self.extras.get("https://api.openai.com/auth", {})
        if isinstance(auth_claims, dict):
            return auth_claims.get("chatgpt_account_id")
        return None

    @property
    def organization_id(self) -> str | None:
        """Get organization ID from JWT claims."""
        # Check in auth claims first
        auth_claims = self.extras.get("https://api.openai.com/auth", {})
        if isinstance(auth_claims, dict) and "organization_id" in auth_claims:
            return str(auth_claims["organization_id"])
        # Fallback to top-level org_id
        org_id = self.extras.get("org_id")
        return str(org_id) if org_id is not None else None

    @property
    def auth0_subject(self) -> str | None:
        """Get Auth0 subject (sub claim)."""
        return self.extras.get("sub")
