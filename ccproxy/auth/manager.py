"""Authentication manager interfaces for centralized auth handling."""

from abc import ABC, abstractmethod
from typing import Any, Protocol

from ccproxy.auth.models import ClaudeCredentials, UserProfile


class AuthManager(Protocol):
    """Protocol for authentication managers."""

    async def get_access_token(self) -> str:
        """Get valid access token.

        Returns:
            Access token string

        Raises:
            AuthenticationError: If authentication fails
        """
        ...

    async def get_credentials(self) -> ClaudeCredentials:
        """Get valid credentials.

        Returns:
            Valid credentials

        Raises:
            AuthenticationError: If authentication fails
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


class BaseAuthManager(ABC):
    """Base class for authentication managers."""

    @abstractmethod
    async def get_access_token(self) -> str:
        """Get valid access token.

        Returns:
            Access token string

        Raises:
            AuthenticationError: If authentication fails
        """
        pass

    @abstractmethod
    async def get_credentials(self) -> ClaudeCredentials:
        """Get valid credentials.

        Returns:
            Valid credentials

        Raises:
            AuthenticationError: If authentication fails
        """
        pass

    @abstractmethod
    async def is_authenticated(self) -> bool:
        """Check if current authentication is valid.

        Returns:
            True if authenticated, False otherwise
        """
        pass

    async def get_user_profile(self) -> UserProfile | None:
        """Get user profile information.

        Returns:
            UserProfile if available, None otherwise
        """
        return None

    async def __aenter__(self) -> "BaseAuthManager":
        """Async context manager entry."""
        return self

    @abstractmethod
    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        pass
