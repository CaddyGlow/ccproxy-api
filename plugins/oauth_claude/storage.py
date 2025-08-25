"""Token storage for Claude OAuth plugin."""

import json
from pathlib import Path

from ccproxy.auth.models import ClaudeCredentials
from ccproxy.auth.storage.base import TokenStorage
from ccproxy.core.logging import get_plugin_logger


logger = get_plugin_logger()


class ClaudeOAuthStorage(TokenStorage[ClaudeCredentials]):
    """Claude OAuth-specific token storage implementation."""

    def __init__(self, storage_path: Path | None = None):
        """Initialize Claude OAuth token storage.

        Args:
            storage_path: Path to storage file
        """
        if storage_path is None:
            # Default to standard Claude credentials location
            storage_path = Path.home() / ".claude" / ".credentials.json"

        self.file_path = storage_path
        self.provider_name = "claude-api"

        # Ensure directory exists
        self.file_path.parent.mkdir(parents=True, exist_ok=True)

    async def save(self, credentials: ClaudeCredentials) -> bool:
        """Save Claude credentials.

        Args:
            credentials: Claude credentials to save

        Returns:
            True if saved successfully, False otherwise
        """
        # Convert to dict for storage
        data = credentials.model_dump(mode="json", exclude_none=True)

        # Save to file
        try:
            self.file_path.write_text(json.dumps(data, indent=2))
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
        # Load from file
        if not self.file_path.exists():
            return None

        try:
            data = json.loads(self.file_path.read_text())
        except Exception as e:
            logger.error(
                "claude_oauth_load_failed", error=str(e), exc_info=e, category="auth"
            )
            return None

        try:
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
                logger.info("claude_oauth_credentials_deleted", category="auth")
                return True
            except Exception as e:
                logger.error(
                    "claude_oauth_delete_failed",
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
