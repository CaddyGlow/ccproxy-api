"""OpenAI-specific authentication models."""

from datetime import UTC, datetime
from typing import Any, Literal

import jwt
from pydantic import BaseModel, Field, computed_field, field_validator

from ccproxy.auth.models.base import BaseProfileInfo, BaseTokenInfo
from ccproxy.core.logging import get_plugin_logger


logger = get_plugin_logger()


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
        """Convert to dictionary for storage.

        Implements BaseCredentials protocol.
        """
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

    @computed_field  # type: ignore[prop-decorator]
    @property
    def access_token_value(self) -> str:
        """Get access token (already a plain string in OpenAI)."""
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
