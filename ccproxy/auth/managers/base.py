"""Base token manager for all authentication providers."""

import json
from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ValidationError

from ccproxy.auth.exceptions import (
    CredentialsInvalidError,
    CredentialsStorageError,
)
from ccproxy.auth.storage.base import TokenStorage
from ccproxy.core.logging import get_logger
from ccproxy.utils.caching import AuthStatusCache, async_ttl_cache


logger = get_logger(__name__)

# Type variable for credentials
CredentialsT = TypeVar("CredentialsT", bound=BaseModel)


class BaseTokenManager(ABC, Generic[CredentialsT]):
    """Base manager for token storage and refresh operations.

    This generic base class provides common functionality for managing
    authentication tokens across different providers (OpenAI, Claude, etc.).

    Type Parameters:
        CredentialsT: The specific credential type (e.g., OpenAICredentials, ClaudeCredentials)
    """

    def __init__(self, storage: TokenStorage[CredentialsT]):
        """Initialize token manager.

        Args:
            storage: Token storage backend that matches the credential type
        """
        self.storage = storage
        self._auth_cache = AuthStatusCache(ttl=60.0)  # 1 minute TTL for auth status
        self._profile_cache: Any = None  # For subclasses that cache profiles

    # ==================== Core Operations ====================

    async def load_credentials(self) -> CredentialsT | None:
        """Load credentials from storage.

        Returns:
            Credentials if found and valid, None otherwise
        """
        try:
            return await self.storage.load()
        except (OSError, PermissionError) as e:
            logger.error("storage_access_failed", error=str(e), exc_info=e)
            return None
        except (CredentialsStorageError, CredentialsInvalidError) as e:
            logger.error("credentials_load_failed", error=str(e), exc_info=e)
            return None
        except json.JSONDecodeError as e:
            logger.error("credentials_json_decode_error", error=str(e), exc_info=e)
            return None
        except ValidationError as e:
            logger.error("credentials_validation_error", error=str(e), exc_info=e)
            return None
        except Exception as e:
            logger.error("unexpected_load_error", error=str(e), exc_info=e)
            return None

    async def save_credentials(self, credentials: CredentialsT) -> bool:
        """Save credentials to storage.

        Args:
            credentials: Credentials to save

        Returns:
            True if saved successfully, False otherwise
        """
        try:
            return await self.storage.save(credentials)
        except (OSError, PermissionError) as e:
            logger.error("storage_access_failed", error=str(e), exc_info=e)
            return False
        except CredentialsStorageError as e:
            logger.error("credentials_save_failed", error=str(e), exc_info=e)
            return False
        except json.JSONDecodeError as e:
            logger.error("credentials_json_encode_error", error=str(e), exc_info=e)
            return False
        except ValidationError as e:
            logger.error("credentials_validation_error", error=str(e), exc_info=e)
            return False
        except Exception as e:
            logger.error("unexpected_save_error", error=str(e), exc_info=e)
            return False

    async def clear_credentials(self) -> bool:
        """Clear stored credentials.

        Returns:
            True if cleared successfully, False otherwise
        """
        try:
            # Clear the cache
            self._auth_cache.clear()

            # Delete from storage
            return await self.storage.delete()
        except Exception as e:
            logger.error("failed_to_clear_credentials", error=str(e), exc_info=e)
            return False

    def get_storage_location(self) -> str:
        """Get the storage location for credentials.

        Returns:
            Storage location description
        """
        return self.storage.get_location()

    # ==================== Common Implementations ====================

    async def validate_token(self) -> bool:
        """Check if stored token is valid and not expired.

        Returns:
            True if valid, False otherwise
        """
        credentials = await self.load_credentials()
        if not credentials:
            return False

        if self.is_expired(credentials):
            logger.info("token_expired")
            return False

        return True

    @abstractmethod
    async def refresh_token(self, oauth_client: Any) -> CredentialsT | None:
        """Refresh the access token using the refresh token.

        Args:
            oauth_client: The OAuth client to use for refreshing

        Returns:
            Updated credentials or None if refresh failed
        """
        pass

    async def get_auth_status(self) -> dict[str, Any]:
        """Get current authentication status.

        Returns:
            Dictionary with authentication status information
        """
        credentials = await self.load_credentials()

        if not credentials:
            return {
                "authenticated": False,
                "reason": "No credentials found",
            }

        if self.is_expired(credentials):
            status = {
                "authenticated": False,
                "reason": "Token expired",
            }

            # Add expiration info if available
            expires_at = self.get_expiration_time(credentials)
            if expires_at:
                status["expires_at"] = expires_at.isoformat()

            # Add account ID if available
            account_id = self.get_account_id(credentials)
            if account_id:
                status["account_id"] = account_id

            return status

        # Token is valid
        status = {"authenticated": True}

        # Add expiration info if available
        expires_at = self.get_expiration_time(credentials)
        if expires_at:
            from datetime import UTC, datetime

            now = datetime.now(UTC)
            delta = expires_at - now
            status["expires_at"] = expires_at.isoformat()
            status["expires_in"] = max(0, int(delta.total_seconds()))

        # Add account ID if available
        account_id = self.get_account_id(credentials)
        if account_id:
            status["account_id"] = account_id

        return status

    @abstractmethod
    def is_expired(self, credentials: CredentialsT) -> bool:
        """Check if credentials are expired.

        Args:
            credentials: Credentials to check

        Returns:
            True if expired, False otherwise
        """
        pass

    @abstractmethod
    def get_account_id(self, credentials: CredentialsT) -> str | None:
        """Get account ID from credentials.

        Args:
            credentials: Credentials to extract account ID from

        Returns:
            Account ID if available, None otherwise
        """
        pass

    def get_expiration_time(self, credentials: CredentialsT) -> Any:
        """Get expiration time from credentials.

        Args:
            credentials: Credentials to extract expiration time from

        Returns:
            Expiration datetime if available, None otherwise
        """
        # Default implementation - plugins can override
        from datetime import UTC, datetime

        if hasattr(credentials, "expires_at"):
            if isinstance(credentials.expires_at, datetime):
                return credentials.expires_at
            elif isinstance(credentials.expires_at, int | float):
                # Assume Unix timestamp in seconds
                return datetime.fromtimestamp(credentials.expires_at, tz=UTC)
        elif hasattr(credentials, "claude_ai_oauth"):
            # Handle Claude credentials format
            expires_at = credentials.claude_ai_oauth.expires_at
            if expires_at:
                return datetime.fromtimestamp(
                    expires_at / 1000, tz=UTC
                )  # Convert from milliseconds
        return None

    # ==================== Unified Profile Support ====================

    async def get_profile(self) -> Any:
        """Get profile information.

        To be implemented by provider-specific managers.
        Returns provider-specific profile model.
        """
        return None

    async def get_unified_profile(self) -> dict[str, Any]:
        """Get profile in a unified format across all providers.

        Returns:
            Dictionary with standardized fields plus provider-specific extras
        """
        profile = await self.get_profile()
        if not profile:
            return {}

        # Handle both old UserProfile and new BaseProfileInfo
        if hasattr(profile, "provider_type"):
            # New BaseProfileInfo-based profile
            return {
                "account_id": profile.account_id,
                "email": profile.email,
                "display_name": profile.display_name,
                "provider": profile.provider_type,
                "extras": profile.extras,  # All provider-specific data
            }
        else:
            # Legacy UserProfile format
            account = getattr(profile, "account", None)
            if account:
                return {
                    "account_id": account.uuid,
                    "email": account.email,
                    "display_name": account.full_name,
                    "provider": "unknown",
                    "extras": account.extras if hasattr(account, "extras") else {},
                }
            return {}

    async def clear_cache(self) -> None:
        """Clear any cached data (profiles, etc.).

        Should be called after token refresh or logout.
        """
        # Clear auth status cache
        if hasattr(self, "_auth_cache"):
            self._auth_cache.clear()

        # Clear profile cache if exists
        if hasattr(self, "_profile_cache"):
            self._profile_cache = None

    # ==================== Common Utility Methods ====================

    async def is_authenticated(self) -> bool:
        """Check if current authentication is valid.

        Returns:
            True if authenticated, False otherwise
        """
        credentials = await self.load_credentials()
        if not credentials:
            return False

        return not self.is_expired(credentials)

    async def get_access_token(self) -> str | None:
        """Get valid access token from credentials.

        Returns:
            Access token if available and valid, None otherwise
        """
        credentials = await self.load_credentials()
        if not credentials:
            return None

        if self.is_expired(credentials):
            logger.info("token_expired")
            return None

        # Get access_token attribute from credentials
        if hasattr(credentials, "access_token"):
            return str(credentials.access_token)
        elif hasattr(credentials, "claude_ai_oauth"):
            # Handle Claude credentials format
            return str(credentials.claude_ai_oauth.access_token.get_secret_value())

        return None

    @async_ttl_cache(ttl=60.0)  # Cache auth status for 1 minute
    async def get_cached_auth_status(self) -> dict[str, Any]:
        """Get current authentication status with caching.

        This is a convenience method that wraps get_auth_status() with caching.

        Returns:
            Dictionary with authentication status information
        """
        return await self.get_auth_status()
