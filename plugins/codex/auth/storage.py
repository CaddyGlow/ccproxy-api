"""Token storage for Codex/OpenAI OAuth."""

import json
from pathlib import Path

from ccproxy.auth.models import OpenAICredentials
from ccproxy.auth.storage.base import TokenStorage
from ccproxy.core.logging import get_plugin_logger


logger = get_plugin_logger()


class CodexTokenStorage(TokenStorage[OpenAICredentials]):
    """Codex/OpenAI-specific token storage implementation."""

    def __init__(self, storage_path: Path | None = None):
        """Initialize Codex token storage.

        Args:
            storage_path: Path to storage file
        """
        if storage_path is None:
            # Default to standard OpenAI credentials location
            storage_path = Path.home() / ".ccproxy" / "openai_credentials.json"

        self.file_path = storage_path
        self.provider_name = "codex"

        # Ensure directory exists
        self.file_path.parent.mkdir(parents=True, exist_ok=True)

    async def save(self, credentials: OpenAICredentials) -> bool:
        """Save OpenAI credentials.

        Args:
            credentials: OpenAI credentials to save

        Returns:
            True if saved successfully, False otherwise
        """
        # Convert to dict for storage
        data = credentials.model_dump(mode="json", exclude_none=True)

        # Save to file
        try:
            self.file_path.write_text(json.dumps(data, indent=2))
            logger.info(
                "openai_credentials_saved",
                has_refresh_token=bool(credentials.refresh_token),
                storage_path=str(self.file_path),
                category="auth",
            )
            return True
        except Exception as e:
            logger.error(
                "Failed to save credentials", error=str(e), exc_info=e, category="auth"
            )
            return False

    async def load(self) -> OpenAICredentials | None:
        """Load OpenAI credentials.

        Returns:
            Stored credentials or None
        """
        # Load from file
        if not self.file_path.exists():
            return None

        try:
            data = json.loads(self.file_path.read_text())
        except Exception as e:
            logger.error(
                "Failed to load credentials", error=str(e), exc_info=e, category="auth"
            )
            return None

        try:
            credentials = OpenAICredentials.model_validate(data)
            logger.info(
                "openai_credentials_loaded",
                has_refresh_token=bool(credentials.refresh_token),
                category="auth",
            )
            return credentials
        except Exception as e:
            logger.error(
                "openai_credentials_load_error",
                error=str(e),
                exc_info=e,
                category="auth",
            )
            return None

    async def exists(self) -> bool:
        """Check if credentials exist in storage.

        Returns:
            True if credentials exist, False otherwise
        """
        return self.file_path.exists() and self.file_path.is_file()

    async def delete(self) -> bool:
        """Delete credentials from storage.

        Returns:
            True if deleted successfully, False otherwise
        """
        if self.file_path.exists():
            try:
                self.file_path.unlink()
                logger.info("openai_credentials_deleted", category="auth")
                return True
            except Exception as e:
                logger.error(
                    "Failed to delete credentials",
                    error=str(e),
                    exc_info=e,
                    category="auth",
                )
                return False
        return False

    def get_location(self) -> str:
        """Get the storage location description.

        Returns:
            Human-readable description of where credentials are stored
        """
        return str(self.file_path)

    # Keep compatibility methods for provider
    async def save_credentials(self, credentials: OpenAICredentials) -> None:
        """Save OpenAI credentials (compatibility method).

        Args:
            credentials: OpenAI credentials to save
        """
        await self.save(credentials)

    async def load_credentials(self) -> OpenAICredentials | None:
        """Load OpenAI credentials (compatibility method).

        Returns:
            Stored credentials or None
        """
        return await self.load()

    async def delete_credentials(self) -> None:
        """Delete stored OpenAI credentials (compatibility method)."""
        await self.delete()
