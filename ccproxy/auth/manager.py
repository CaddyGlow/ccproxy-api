"""Unified authentication manager interface for all providers."""

from typing import Any, Protocol, runtime_checkable

from ccproxy.auth.models.base import UserProfile
from ccproxy.auth.models.credentials import BaseCredentials


@runtime_checkable
class AuthManager(Protocol):
    """Unified authentication manager protocol for all providers.

    This protocol defines the complete interface that all authentication managers
    must implement, supporting both provider-specific methods (like Claude credentials)
    and generic methods (like auth headers) for maximum flexibility.
    """

    # ==================== Core Authentication Methods ====================

    async def get_access_token(self) -> str:
        """Get valid access token.

        Returns:
            Access token string

        Raises:
            AuthenticationError: If authentication fails
        """
        ...

    async def get_credentials(self) -> BaseCredentials:
        """Get valid credentials.

        Note: For non-Claude providers, this may return minimal/dummy credentials
        or raise AuthenticationError if not supported.

        Returns:
            Valid credentials

        Raises:
            AuthenticationError: If authentication fails or not supported
        """
        ...

    async def is_authenticated(self) -> bool:
        """Check if current authentication is valid.

        Returns:
            True if authenticated, False otherwise
        """
        ...

    async def get_user_profile(self) -> UserProfile | None:
        """Get user profile information.

        Returns:
            UserProfile if available, None otherwise
        """
        ...

    # ==================== Provider-Generic Methods ====================

    async def validate_credentials(self) -> bool:
        """Validate that credentials are available and valid.

        Returns:
            True if credentials are valid, False otherwise
        """
        ...

    def get_provider_name(self) -> str:
        """Get the provider name for logging.

        Returns:
            Provider name string (e.g., "anthropic-claude", "openai-codex")
        """
        ...

    # ==================== Context Manager Support ====================

    async def __aenter__(self) -> "AuthManager":
        """Async context manager entry."""
        ...

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        ...
