"""Bearer token authentication implementation."""

from typing import Any

from ccproxy.auth.exceptions import AuthenticationError
from ccproxy.auth.models.base import UserProfile
from ccproxy.auth.models.credentials import BaseCredentials


class BearerCredentials:
    """Simple bearer token credentials that implement BaseCredentials protocol."""

    def __init__(self, token: str):
        """Initialize with a bearer token.

        Args:
            token: Bearer token string
        """
        self.token = token

    def is_expired(self) -> bool:
        """Check if credentials are expired.

        Bearer tokens don't have expiration in this implementation.

        Returns:
            Always False for bearer tokens
        """
        return False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage.

        Returns:
            Dictionary with token
        """
        return {"token": self.token, "type": "bearer"}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BearerCredentials":
        """Create from dictionary.

        Args:
            data: Dictionary containing token

        Returns:
            BearerCredentials instance
        """
        return cls(token=data["token"])


class BearerTokenAuthManager:
    """Authentication manager for static bearer tokens."""

    def __init__(self, token: str) -> None:
        """Initialize with a static bearer token.

        Args:
            token: Bearer token string
        """
        self.token = token.strip()
        if not self.token:
            raise ValueError("Token cannot be empty")

    async def get_access_token(self) -> str:
        """Get the bearer token.

        Returns:
            Bearer token string

        Raises:
            AuthenticationError: If token is invalid
        """
        if not self.token:
            raise AuthenticationError("No bearer token available")
        return self.token

    async def get_credentials(self) -> BaseCredentials:
        """Get credentials as BearerCredentials.

        Returns:
            BearerCredentials instance wrapping the token
        """
        return BearerCredentials(token=self.token)

    async def is_authenticated(self) -> bool:
        """Check if bearer token is available.

        Returns:
            True if token is available, False otherwise
        """
        return bool(self.token)

    async def get_user_profile(self) -> UserProfile | None:
        """Get user profile (not supported for bearer tokens).

        Returns:
            None - bearer tokens don't support user profiles
        """
        return None

    async def __aenter__(self) -> "BearerTokenAuthManager":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        pass

    # ==================== Provider-Generic Methods ====================

    async def validate_credentials(self) -> bool:
        """Validate that credentials are available and valid.

        Returns:
            True if credentials are valid, False otherwise
        """
        return bool(self.token)

    def get_provider_name(self) -> str:
        """Get the provider name for logging.

        Returns:
            Provider name string
        """
        return "bearer-token"
