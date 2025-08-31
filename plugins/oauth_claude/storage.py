"""Token storage for Claude OAuth plugin."""

from pathlib import Path
from typing import Any

from ccproxy.auth.storage.base import BaseJsonStorage
from ccproxy.core.logging import get_plugin_logger
from plugins.claude_api.auth.models import ClaudeCredentials


logger = get_plugin_logger()


class ClaudeOAuthStorage(BaseJsonStorage[ClaudeCredentials]):
    """Claude OAuth-specific token storage implementation."""

    def __init__(self, storage_path: Path | None = None):
        """Initialize Claude OAuth token storage.

        Args:
            storage_path: Path to storage file
        """
        if storage_path is None:
            # Default to standard Claude credentials location
            storage_path = Path.home() / ".claude" / ".credentials.json"

        super().__init__(storage_path)
        self.provider_name = "claude-api"

    async def save(self, credentials: ClaudeCredentials) -> bool:
        """Save Claude credentials.

        Args:
            credentials: Claude credentials to save

        Returns:
            True if saved successfully, False otherwise
        """
        try:
            # Convert to dict for storage
            data = credentials.model_dump(mode="json", exclude_none=True)

            # Use parent class's atomic write with backup
            await self._write_json(data)

            logger.info(
                "claude_oauth_credentials_saved",
                has_oauth=bool(credentials.claude_ai_oauth),
                storage_path=str(self.file_path),
                category="auth",
            )
            return True
        except Exception as e:
            logger.error(
                "claude_oauth_save_failed", error=str(e), exc_info=e, category="auth"
            )
            return False

    async def load(self) -> ClaudeCredentials | None:
        """Load Claude credentials.

        Returns:
            Stored credentials or None
        """
        try:
            # Use parent class's read method
            data = await self._read_json()
            if not data:
                return None

            credentials = ClaudeCredentials.model_validate(data)
            logger.info(
                "claude_oauth_credentials_loaded",
                has_oauth=bool(credentials.claude_ai_oauth),
                category="auth",
            )
            return credentials
        except Exception as e:
            logger.error(
                "claude_oauth_credentials_load_error",
                error=str(e),
                exc_info=e,
                category="auth",
            )
            return None

    # The exists(), delete(), and get_location() methods are inherited from BaseJsonStorage
