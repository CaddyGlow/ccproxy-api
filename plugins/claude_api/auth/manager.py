"""Claude API token manager implementation for the Claude API plugin."""

from datetime import datetime
from pathlib import Path
from typing import Any

from ccproxy.auth.managers.base import BaseTokenManager
from ccproxy.auth.models import ClaudeCredentials
from ccproxy.auth.oauth.registry import get_oauth_registry
from ccproxy.auth.storage.generic import GenericJsonStorage
from ccproxy.core.logging import get_plugin_logger

from .models import ClaudeProfileInfo, ClaudeTokenWrapper


logger = get_plugin_logger()


class ClaudeApiTokenManager(BaseTokenManager[ClaudeCredentials]):
    """Manager for Claude API token storage and refresh operations.

    Uses the generic storage and wrapper pattern for consistency.
    """

    def __init__(self, storage: GenericJsonStorage[ClaudeCredentials] | None = None):
        """Initialize Claude API token manager.

        Args:
            storage: Optional custom storage, defaults to standard location
        """
        if storage is None:
            storage = GenericJsonStorage(
                Path.home() / ".claude" / ".credentials.json", ClaudeCredentials
            )
        super().__init__(storage)
        self._profile_cache: ClaudeProfileInfo | None = None

    # ==================== Abstract Method Implementations ====================

    async def refresh_token(self, oauth_client: Any = None) -> ClaudeCredentials | None:
        """Refresh the access token using the refresh token.

        Args:
            oauth_client: Deprecated - OAuth provider is now looked up from registry

        Returns:
            Updated credentials or None if refresh failed
        """
        # Get OAuth provider from registry
        registry = get_oauth_registry()
        oauth_provider = registry.get_provider("claude-api")
        if not oauth_provider:
            logger.error("claude_oauth_provider_not_found", category="auth")
            return None

        credentials = await self.load_credentials()
        if not credentials:
            logger.error("no_credentials_to_refresh", category="auth")
            return None

        wrapper = ClaudeTokenWrapper(credentials=credentials)
        refresh_token = wrapper.refresh_token_value
        if not refresh_token:
            logger.error("no_refresh_token_available", category="auth")
            return None

        try:
            # Use OAuth provider to refresh
            new_credentials: ClaudeCredentials = (
                await oauth_provider.refresh_access_token(refresh_token)
            )

            # Save updated credentials
            if await self.save_credentials(new_credentials):
                logger.info("token_refreshed_successfully", category="auth")
                # Clear profile cache as token changed
                self._profile_cache = None
                return new_credentials

            logger.error("failed_to_save_refreshed_credentials", category="auth")
            return None

        except Exception as e:
            logger.error(
                "Token refresh failed",
                error=str(e),
                exc_info=e,
                category="auth",
            )
            return None

    def is_expired(self, credentials: ClaudeCredentials) -> bool:
        """Check if credentials are expired using wrapper."""
        wrapper = ClaudeTokenWrapper(credentials=credentials)
        return wrapper.is_expired

    def get_account_id(self, credentials: ClaudeCredentials) -> str | None:
        """Get account ID from credentials.

        Claude doesn't store account_id in tokens, would need
        to fetch from profile API.
        """
        if self._profile_cache:
            return self._profile_cache.account_id
        return None

    # ==================== Claude-Specific Methods ====================

    def get_expiration_time(self, credentials: ClaudeCredentials) -> datetime | None:
        """Get expiration time as datetime."""
        wrapper = ClaudeTokenWrapper(credentials=credentials)
        return wrapper.expires_at_datetime

    async def get_access_token(self) -> str | None:
        """Get valid access token, automatically refreshing if expired.

        Returns:
            Access token if available and valid, None otherwise
        """
        credentials = await self.load_credentials()
        if not credentials:
            logger.debug("no_credentials_found", category="auth")
            return None

        # Check if token is expired
        if self.is_expired(credentials):
            logger.info("claude_token_expired_attempting_refresh", category="auth")

            # Try to refresh if we have a refresh token
            wrapper = ClaudeTokenWrapper(credentials=credentials)
            refresh_token = wrapper.refresh_token_value
            if refresh_token:
                try:
                    refreshed = await self.refresh_token()
                    if refreshed:
                        logger.info(
                            "claude_token_refreshed_successfully", category="auth"
                        )
                        wrapper = ClaudeTokenWrapper(credentials=refreshed)
                        return wrapper.access_token_value
                    else:
                        logger.error("claude_token_refresh_failed", category="auth")
                        return None
                except Exception as e:
                    logger.error(
                        "Error refreshing Claude token", error=str(e), category="auth"
                    )
                    return None
            else:
                logger.warning(
                    "Cannot refresh Claude token - no refresh token available",
                    category="auth",
                )
                return None

        # Token is still valid
        wrapper = ClaudeTokenWrapper(credentials=credentials)
        return wrapper.access_token_value

    async def get_access_token_value(self) -> str | None:
        """Get the actual access token value.

        Returns:
            Access token string if available, None otherwise
        """
        credentials = await self.load_credentials()
        if not credentials:
            return None

        if self.is_expired(credentials):
            return None

        wrapper = ClaudeTokenWrapper(credentials=credentials)
        return wrapper.access_token_value

    async def get_profile(self) -> ClaudeProfileInfo | None:
        """Get user profile from cache or API.

        Returns:
            ClaudeProfileInfo or None if not authenticated
        """
        if self._profile_cache:
            return self._profile_cache

        credentials = await self.load_credentials()
        if not credentials or self.is_expired(credentials):
            return None

        # Would need HTTP client injected to fetch from API
        # For now, return None (actual implementation would call API)
        # response = await self.http_client.get("/api/organizations/me")
        # self._profile_cache = ClaudeProfileInfo.from_api_response(response)
        return self._profile_cache
