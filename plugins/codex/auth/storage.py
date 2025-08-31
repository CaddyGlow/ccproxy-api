"""Token storage for Codex/OpenAI OAuth."""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ccproxy.auth.storage.base import BaseJsonStorage
from ccproxy.core.logging import get_plugin_logger
from plugins.codex.auth.models import OpenAICredentials


logger = get_plugin_logger()


class CodexTokenStorage(BaseJsonStorage[OpenAICredentials]):
    """Codex/OpenAI-specific token storage implementation."""

    def __init__(self, storage_path: Path | None = None):
        """Initialize Codex token storage.

        Args:
            storage_path: Path to storage file
        """
        if storage_path is None:
            # Use ~/.codex/auth.json as the standard location
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
            # Format data in Codex structure
            data = {
                "OPENAI_API_KEY": None,
                "tokens": {
                    "id_token": credentials.id_token,
                    "access_token": credentials.access_token,
                    "refresh_token": credentials.refresh_token,
                    "account_id": credentials.account_id,
                },
                "last_refresh": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            }

            # Use parent class's atomic write with backup
            await self._write_json(data)

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
        """Load OpenAI credentials from Codex format.

        Returns:
            Stored credentials or None
        """
        try:
            # Use parent class's read method
            data = await self._read_json()
            if not data:
                return None
        except Exception as e:
            logger.error(
                "Failed to load credentials", error=str(e), exc_info=e, category="auth"
            )
            return None

        try:
            # Handle new Codex format with nested tokens
            if "tokens" in data:
                tokens = data["tokens"]

                # Parse expiration from access token (JWT)
                expires_at = None
                if access_token := tokens.get("access_token"):
                    try:
                        import base64

                        parts = access_token.split(".")
                        if len(parts) == 3:
                            payload_b64 = parts[1] + "=" * (4 - len(parts[1]) % 4)
                            payload = json.loads(base64.urlsafe_b64decode(payload_b64))
                            if exp := payload.get("exp"):
                                expires_at = datetime.fromtimestamp(exp, tz=UTC)
                    except Exception:
                        pass

                # If no expiration from JWT, fall back to 10 hours from last refresh
                if not expires_at and (last_refresh := data.get("last_refresh")):
                    try:
                        last_refresh_dt = datetime.fromisoformat(
                            last_refresh.replace("Z", "+00:00")
                        )
                        # OpenAI tokens typically last 10 hours
                        from datetime import timedelta

                        expires_at = last_refresh_dt.replace(microsecond=0) + timedelta(
                            hours=10
                        )
                    except Exception:
                        pass

                # Default to expired if we can't determine
                if not expires_at:
                    expires_at = datetime.now(UTC)

                credentials = OpenAICredentials(
                    access_token=tokens.get("access_token", ""),
                    refresh_token=tokens.get("refresh_token", ""),
                    id_token=tokens.get("id_token"),
                    expires_at=expires_at,
                    account_id=tokens.get("account_id", "unknown"),
                    active=True,
                )
            else:
                # Fall back to old format
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

    # The exists(), delete(), and get_location() methods are inherited from BaseJsonStorage

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
