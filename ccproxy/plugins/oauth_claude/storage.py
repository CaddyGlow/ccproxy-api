"""Token storage for Claude OAuth plugin."""

import asyncio
import json
import tempfile
from pathlib import Path
from typing import Any, cast

from ccproxy.auth.storage.base import BaseJsonStorage
from ccproxy.core.logging import get_plugin_logger

from .models import ClaudeCredentials, ClaudeProfileInfo


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
            # Convert to dict for storage (uses by_alias=True by default)
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


class ClaudeProfileStorage:
    """Claude profile storage implementation for .account.json."""

    def __init__(self, storage_path: Path | None = None):
        """Initialize Claude profile storage.

        Args:
            storage_path: Path to storage file
        """
        if storage_path is None:
            # Default to standard Claude account location
            storage_path = Path.home() / ".claude" / ".account.json"

        self.file_path = storage_path

    async def _write_json(self, data: dict[str, Any]) -> None:
        """Write JSON data to file atomically.

        Args:
            data: JSON data to write
        """
        # Ensure parent directory exists
        self.file_path.parent.mkdir(parents=True, exist_ok=True)

        # Write to temp file first for atomic operation
        def write_file() -> None:
            with tempfile.NamedTemporaryFile(
                mode="w",
                dir=self.file_path.parent,
                delete=False,
                prefix=".tmp_",
                suffix=".json",
            ) as tmp_file:
                json.dump(data, tmp_file, indent=2)
                tmp_path = Path(tmp_file.name)

            # Set proper permissions before moving
            tmp_path.chmod(0o600)
            # Atomic rename
            tmp_path.replace(self.file_path)

        await asyncio.to_thread(write_file)

    async def _read_json(self) -> dict[str, Any] | None:
        """Read JSON data from file.

        Returns:
            Parsed JSON data or None if file doesn't exist
        """
        if not self.file_path.exists():
            return None

        def read_file() -> dict[str, Any]:
            with self.file_path.open("r") as f:
                return cast(dict[str, Any], json.load(f))

        return cast(dict[str, Any], await asyncio.to_thread(read_file))

    async def save_profile(self, profile_data: dict[str, Any]) -> bool:
        """Save Claude profile data.

        Args:
            profile_data: Raw profile data from API

        Returns:
            True if saved successfully, False otherwise
        """
        try:
            # Write the raw profile data
            await self._write_json(profile_data)

            # Extract key info for logging
            account = profile_data.get("account", {})
            logger.info(
                "claude_profile_saved",
                account_id=account.get("uuid"),
                email=account.get("email"),
                has_claude_pro=account.get("has_claude_pro"),
                has_claude_max=account.get("has_claude_max"),
                storage_path=str(self.file_path),
                category="auth",
            )
            return True
        except Exception as e:
            logger.error(
                "claude_profile_save_failed",
                error=str(e),
                exc_info=e,
                category="auth",
            )
            return False

    async def load_profile(self) -> ClaudeProfileInfo | None:
        """Load Claude profile.

        Returns:
            ClaudeProfileInfo or None if not found
        """
        try:
            data = await self._read_json()
            if not data:
                return None

            profile = ClaudeProfileInfo.from_api_response(data)
            logger.info(
                "claude_profile_loaded",
                account_id=profile.account_id,
                email=profile.email,
                category="auth",
            )
            return profile
        except Exception as e:
            logger.error(
                "claude_profile_load_error",
                error=str(e),
                exc_info=e,
                category="auth",
            )
            return None
