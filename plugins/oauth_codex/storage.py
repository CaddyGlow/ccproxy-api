"""Token storage for Codex OAuth plugin."""

from pathlib import Path

from ccproxy.auth.storage.base import BaseJsonStorage
from ccproxy.core.logging import get_plugin_logger

from .models import OpenAICredentials


logger = get_plugin_logger()


class CodexTokenStorage(BaseJsonStorage[OpenAICredentials]):
    """Codex/OpenAI OAuth-specific token storage implementation."""

    def __init__(self, storage_path: Path | None = None):
        """Initialize Codex token storage.

        Args:
            storage_path: Path to storage file
        """
        if storage_path is None:
            # Default to standard OpenAI credentials location
            storage_path = Path.home() / ".codex" / "auth.json"

        super().__init__(storage_path)
        self.provider_name = "codex"

    async def save(self, credentials: OpenAICredentials) -> bool:
        """Save OpenAI credentials.

        Args:
            credentials: OpenAI credentials to save

        Returns:
            True if saved successfully, False otherwise
        """
        try:
            # Convert to dict for storage
            data = credentials.model_dump(mode="json", exclude_none=True)

            # Use parent class's atomic write with backup
            await self._write_json(data)

            logger.info(
                "codex_oauth_credentials_saved",
                has_refresh_token=bool(credentials.refresh_token),
                storage_path=str(self.file_path),
                category="auth",
            )
            return True
        except Exception as e:
            logger.error(
                "codex_oauth_save_failed", error=str(e), exc_info=e, category="auth"
            )
            return False

    async def load(self) -> OpenAICredentials | None:
        """Load OpenAI credentials.

        Returns:
            Stored credentials or None
        """
        try:
            # Use parent class's read method (avoid redundant exists() checks)
            data = await self._read_json()
            if not data:
                logger.debug(
                    "codex_auth_file_empty",
                    storage_path=str(self.file_path),
                    category="auth",
                )
                return None

            credentials = OpenAICredentials.model_validate(data)
            logger.info(
                "codex_oauth_credentials_loaded",
                has_refresh_token=bool(credentials.refresh_token),
                category="auth",
            )
            return credentials
        except Exception as e:
            logger.error(
                "codex_oauth_credentials_load_error",
                error=str(e),
                exc_info=e,
                category="auth",
            )
            return None

    # The exists(), delete(), and get_location() methods are inherited from BaseJsonStorage
