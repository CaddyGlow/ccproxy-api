"""Enhanced base token manager with automatic token refresh."""

from typing import Any

from ccproxy.auth.managers.base import BaseTokenManager, CredentialsT
from ccproxy.core.logging import get_logger


logger = get_logger(__name__)


class EnhancedTokenManager(BaseTokenManager[CredentialsT]):
    """Enhanced token manager with automatic refresh capability."""

    async def get_access_token_with_refresh(
        self, oauth_client: Any = None
    ) -> str | None:
        """Get valid access token, automatically refreshing if expired.

        Args:
            oauth_client: Optional OAuth client for token refresh.
                         If not provided, will try to get from context.

        Returns:
            Access token if available and valid, None otherwise
        """
        credentials = await self.load_credentials()
        if not credentials:
            logger.debug("no_credentials_found")
            return None

        # Check if token is expired
        if self.is_expired(credentials):
            logger.info("token_expired_attempting_refresh")

            # Try to refresh if we have a refresh token and oauth client
            if oauth_client and hasattr(credentials, "refresh_token"):
                refreshed = await self.refresh_token(oauth_client)
                if refreshed:
                    logger.info("token_refreshed_successfully")
                    credentials = refreshed
                else:
                    logger.error("token_refresh_failed")
                    return None
            else:
                logger.warning(
                    "Cannot refresh token",
                    has_oauth_client=bool(oauth_client),
                    has_refresh_token=hasattr(credentials, "refresh_token"),
                )
                return None

        # Get access_token attribute from credentials
        if hasattr(credentials, "access_token"):
            return str(credentials.access_token)
        elif hasattr(credentials, "claude_ai_oauth"):
            # Handle Claude credentials format
            return str(credentials.claude_ai_oauth.access_token.get_secret_value())

        return None

    async def ensure_valid_token(self, oauth_client: Any = None) -> bool:
        """Ensure we have a valid (non-expired) token, refreshing if needed.

        Args:
            oauth_client: Optional OAuth client for token refresh

        Returns:
            True if we have a valid token (after refresh if needed), False otherwise
        """
        token = await self.get_access_token_with_refresh(oauth_client)
        return token is not None
