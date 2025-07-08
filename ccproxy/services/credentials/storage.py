"""Abstract base classes and interfaces for credential storage."""

from abc import ABC, abstractmethod
from pathlib import Path

from ccproxy.services.credentials.models import ClaudeCredentials


class CredentialsStorageBackend(ABC):
    """Abstract base class for credential storage backends."""

    @abstractmethod
    async def load(self) -> ClaudeCredentials | None:
        """Load credentials from storage.

        Returns:
            Parsed credentials if found and valid, None otherwise
        """
        pass

    @abstractmethod
    async def save(self, credentials: ClaudeCredentials) -> bool:
        """Save credentials to storage.

        Args:
            credentials: Credentials to save

        Returns:
            True if saved successfully, False otherwise
        """
        pass

    @abstractmethod
    async def exists(self) -> bool:
        """Check if credentials exist in storage.

        Returns:
            True if credentials exist, False otherwise
        """
        pass

    @abstractmethod
    async def delete(self) -> bool:
        """Delete credentials from storage.

        Returns:
            True if deleted successfully, False otherwise
        """
        pass

    @abstractmethod
    def get_location(self) -> str:
        """Get the storage location description.

        Returns:
            Human-readable description of where credentials are stored
        """
        pass
