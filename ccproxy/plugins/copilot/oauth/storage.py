"""Storage implementation for GitHub Copilot OAuth credentials."""

from pathlib import Path

from ccproxy.auth.storage.base import BaseJsonStorage
from ccproxy.core.logging import get_plugin_logger

from .models import CopilotCredentials, CopilotOAuthToken, CopilotTokenResponse


logger = get_plugin_logger()


class CopilotOAuthStorage(BaseJsonStorage[CopilotCredentials]):
    """Storage implementation for Copilot OAuth credentials."""

    def __init__(self, credentials_path: Path | None = None) -> None:
        """Initialize storage with credentials path.

        Args:
            credentials_path: Path to credentials file (uses default if None)
        """
        if credentials_path is None:
            # Use standard GitHub Copilot storage location
            credentials_path = Path.home() / ".config" / "copilot" / "credentials.json"

        super().__init__(credentials_path)

    async def save(self, credentials: CopilotCredentials) -> bool:
        """Store Copilot credentials to file.

        Args:
            credentials: Credentials to store
        """
        try:
            # Update timestamp
            credentials.refresh_updated_at()

            # Convert to dict for storage
            data = credentials.model_dump(mode="json", exclude_none=True)

            # Use parent class's atomic write with backup
            await self._write_json(data)

            logger.debug(
                "credentials_stored",
                path=str(self.file_path),
                account_type=credentials.account_type,
            )
            return True
        except Exception as e:
            logger.error("credentials_save_failed", error=str(e), exc_info=e)
            return False

    async def load(self) -> CopilotCredentials | None:
        """Load Copilot credentials from file.

        Returns:
            Credentials if found and valid, None otherwise
        """
        try:
            # Use parent class's read method
            data = await self._read_json()
            if not data:
                logger.debug(
                    "credentials_not_found",
                    path=str(self.file_path),
                )
                return None

            credentials = CopilotCredentials.model_validate(data)
            logger.debug(
                "credentials_loaded",
                path=str(self.file_path),
                account_type=credentials.account_type,
                is_expired=credentials.is_expired(),
            )
            return credentials
        except Exception as e:
            logger.error(
                "credentials_load_failed",
                error=str(e),
                exc_info=e,
            )
            return None

    async def delete(self) -> bool:
        """Clear stored credentials."""
        result = await super().delete()

        logger.debug(
            "credentials_cleared",
            path=str(self.file_path),
        )
        return result

    async def update_oauth_token(self, oauth_token: CopilotOAuthToken) -> None:
        """Update OAuth token in stored credentials.

        Args:
            oauth_token: New OAuth token to store
        """
        credentials = await self.load()
        if not credentials:
            # Create new credentials with just the OAuth token
            credentials = CopilotCredentials(
                oauth_token=oauth_token, copilot_token=None
            )
        else:
            # Update existing credentials
            credentials.oauth_token = oauth_token

        await self.save(credentials)

    async def update_copilot_token(self, copilot_token: CopilotTokenResponse) -> None:
        """Update Copilot service token in stored credentials.

        Args:
            copilot_token: New Copilot token to store
        """
        credentials = await self.load()
        if not credentials:
            logger.warning(
                "no_oauth_credentials_for_copilot_token",
                message="Cannot store Copilot token without OAuth credentials",
            )
            raise ValueError(
                "OAuth credentials must exist before storing Copilot token"
            )

        credentials.copilot_token = copilot_token
        await self.save(credentials)

    async def get_oauth_token(self) -> CopilotOAuthToken | None:
        """Get OAuth token from stored credentials.

        Returns:
            OAuth token if available, None otherwise
        """
        credentials = await self.load()
        return credentials.oauth_token if credentials else None

    async def get_copilot_token(self) -> CopilotTokenResponse | None:
        """Get Copilot service token from stored credentials.

        Returns:
            Copilot token if available, None otherwise
        """
        credentials = await self.load()
        return credentials.copilot_token if credentials else None

    # BaseOAuthStorage protocol methods

    # Additional convenience methods for Copilot-specific functionality

    async def load_credentials(self) -> CopilotCredentials | None:
        """Legacy method name for backward compatibility."""
        return await self.load()

    async def store_credentials(self, credentials: CopilotCredentials) -> None:
        """Legacy method name for backward compatibility."""
        await self.save(credentials)

    async def save_credentials(self, credentials: CopilotCredentials) -> None:
        """Save credentials method for OAuth provider compatibility."""
        await self.save(credentials)

    async def clear_credentials(self) -> None:
        """Legacy method name for backward compatibility."""
        await self.delete()
