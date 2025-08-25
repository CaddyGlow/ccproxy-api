"""Bearer token authentication implementation."""

from typing import Any

from ccproxy.auth.exceptions import AuthenticationError
from ccproxy.auth.models import ClaudeCredentials, UserProfile


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

    async def get_credentials(self) -> ClaudeCredentials:
        """Get credentials (not supported for bearer tokens).

        Raises:
            AuthenticationError: Bearer tokens don't support full credentials
        """
        raise AuthenticationError(
            "Bearer token authentication doesn't support full credentials"
        )

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

    async def get_auth_headers(self) -> dict[str, str]:
        """Get authentication headers for the request.

        Returns:
            Dictionary of headers to add to requests
        """
        if not self.token:
            raise AuthenticationError("No bearer token available")
        return {"Authorization": f"Bearer {self.token}"}

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
