"""OpenAI-specific authentication models."""

from datetime import datetime
from typing import Literal

import jwt
from pydantic import computed_field

from ccproxy.auth.models import OpenAICredentials
from ccproxy.auth.models.base import BaseProfileInfo, BaseTokenInfo
from ccproxy.core.logging import get_plugin_logger


logger = get_plugin_logger()


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
