"""Token storage for Codex/OpenAI OAuth."""

import json
from datetime import UTC, datetime
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
            # Use ~/.codex/auth.json as the standard location
            storage_path = Path.home() / ".codex" / "auth.json"

        self.file_path = storage_path
        self.provider_name = "codex"

        # Ensure directory exists
        self.file_path.parent.mkdir(parents=True, exist_ok=True)

    async def save(self, credentials: OpenAICredentials) -> bool:
        """Save OpenAI credentials with backup.

        Args:
            credentials: OpenAI credentials to save

        Returns:
            True if saved successfully, False otherwise
        """
        try:
            # Create backup if file exists
            if self.file_path.exists():
                backup_name = (
                    f"auth.json.{datetime.now().strftime('%Y%m%d-%H%M%S')}.bak"
                )
                backup_path = self.file_path.parent / backup_name
                try:
                    import shutil

                    shutil.copy2(self.file_path, backup_path)
                    logger.info(
                        "Created backup",
                        backup_path=str(backup_path),
                        category="auth",
                    )
                except Exception as e:
                    logger.warning(
                        "Failed to create backup",
                        error=str(e),
                        category="auth",
                    )

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
        """Load OpenAI credentials from Codex format.

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
