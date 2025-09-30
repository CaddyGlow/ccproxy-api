"""Base token manager for all authentication providers."""

import json
import os
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from typing import Any, Generic, TypeVar

from pydantic import ValidationError

from ccproxy.auth.exceptions import (
    CredentialsInvalidError,
    CredentialsStorageError,
)
from ccproxy.auth.managers.token_snapshot import TokenSnapshot
from ccproxy.auth.models.credentials import BaseCredentials
from ccproxy.auth.storage.base import TokenStorage
from ccproxy.core.logging import get_logger
from ccproxy.utils.caching import AuthStatusCache, async_ttl_cache


logger = get_logger(__name__)

# Type variable for credentials
CredentialsT = TypeVar("CredentialsT", bound=BaseCredentials)


class BaseTokenManager(ABC, Generic[CredentialsT]):
    """Base manager for token storage and refresh operations.

    This generic base class provides common functionality for managing
    authentication tokens across different providers (OpenAI, Claude, etc.).

    Type Parameters:
        CredentialsT: The specific credential type (e.g., OpenAICredentials, ClaudeCredentials)
    """

    def __init__(
        self,
        storage: TokenStorage[CredentialsT],
        credentials_ttl: float | None = None,
        refresh_grace_seconds: float | None = None,
    ):
        """Initialize token manager.

        Args:
            storage: Token storage backend that matches the credential type
        """
        self.storage = storage
        self._auth_cache = AuthStatusCache(ttl=60.0)  # 1 minute TTL for auth status
        self._profile_cache: Any = None  # For subclasses that cache profiles
        # In-memory credentials cache to reduce file checks
        self._credentials_cache: CredentialsT | None = None
        self._credentials_loaded_at: float | None = None
        # TTL for rechecking credentials from storage (config-driven)
        # Prefer explicit parameter; fallback to environment; then default.
        if credentials_ttl is not None:
            try:
                ttl_val = float(credentials_ttl)
                self._credentials_ttl = ttl_val if ttl_val >= 0 else 30.0
            except Exception:
                self._credentials_ttl = 30.0
        else:
            env_val = os.getenv("AUTH__CREDENTIALS_TTL_SECONDS")
            try:
                self._credentials_ttl = float(env_val) if env_val is not None else 30.0
                if self._credentials_ttl < 0:
                    self._credentials_ttl = 30.0
            except Exception:
                self._credentials_ttl = 30.0

        # Grace period before expiry to trigger proactive refresh
        if refresh_grace_seconds is not None:
            try:
                grace_val = float(refresh_grace_seconds)
                self._refresh_grace_seconds = grace_val if grace_val >= 0 else 0.0
            except Exception:
                self._refresh_grace_seconds = 120.0
        else:
            env_grace = os.getenv("AUTH__REFRESH_GRACE_SECONDS")
            try:
                grace_val = float(env_grace) if env_grace is not None else 120.0
                if grace_val < 0:
                    grace_val = 0.0
                self._refresh_grace_seconds = grace_val
            except Exception:
                self._refresh_grace_seconds = 120.0

    # ==================== Core Operations ====================

    async def load_credentials(self) -> CredentialsT | None:
        """Load credentials from storage.

        Returns:
            Credentials if found and valid, None otherwise
        """
        try:
            # Serve from cache when fresh and not expired
            if self._credentials_cache is not None and self._credentials_loaded_at:
                from time import time as _now

                age = _now() - self._credentials_loaded_at
                if age < self._credentials_ttl and not self.is_expired(
                    self._credentials_cache
                ):
                    logger.debug(
                        "credentials_cache_hit",
                        age_seconds=round(age, 2),
                        ttl_seconds=self._credentials_ttl,
                    )
                    return self._credentials_cache

            # Otherwise, reload from storage (also triggers on expired or stale cache)
            creds = await self.storage.load()
            # Update cache regardless of result (None clears cache)
            self._credentials_cache = creds
            from time import time as _now

            self._credentials_loaded_at = _now()
            logger.debug(
                "credentials_cache_refreshed",
                has_credentials=bool(creds),
                ttl_seconds=self._credentials_ttl,
            )
            return creds
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
            ok = await self.storage.save(credentials)
            if ok:
                # Update cache immediately
                self._credentials_cache = credentials
                from time import time as _now

                self._credentials_loaded_at = _now()
            return ok
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
            # Clear the caches
            self._auth_cache.clear()
            self._credentials_cache = None
            self._credentials_loaded_at = None

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

    @abstractmethod
    def _build_token_snapshot(self, credentials: CredentialsT) -> TokenSnapshot:
        """Construct a token snapshot for the given credentials."""

    def _safe_token_snapshot(self, credentials: CredentialsT) -> TokenSnapshot | None:
        """Safely build a token snapshot with defensive logging."""
        try:
            return self._build_token_snapshot(credentials)
        except NotImplementedError:
            raise
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.debug(
                "token_snapshot_failed",
                error=str(exc),
                credentials_type=type(credentials).__name__,
                category="auth",
            )
            return None

    async def get_token_snapshot(self) -> TokenSnapshot | None:
        """Return a lightweight snapshot of stored token metadata."""
        credentials = await self.load_credentials()
        if not credentials:
            return None
        try:
            return self._build_token_snapshot(credentials)
        except NotImplementedError:
            raise
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.debug(
                "token_snapshot_failed",
                error=str(exc),
                credentials_type=type(credentials).__name__,
                category="auth",
            )
            return None

    # ==================== Common Implementations ====================

    @property
    def refresh_grace_seconds(self) -> float:
        """Seconds before expiry when tokens should be proactively refreshed."""

        return self._refresh_grace_seconds

    def seconds_until_expiration(self, credentials: CredentialsT) -> float | None:
        """Return seconds until the access token expires, if available."""

        expires_at = self.get_expiration_time(credentials)
        if not isinstance(expires_at, datetime):
            return None

        # Normalise naive datetimes to UTC to avoid comparison issues
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=UTC)

        delta = expires_at - datetime.now(UTC)
        return delta.total_seconds()

    def should_refresh(
        self, credentials: CredentialsT, grace_seconds: float | None = None
    ) -> bool:
        """Determine whether credentials should be refreshed."""

        seconds_remaining = self.seconds_until_expiration(credentials)
        if seconds_remaining is None:
            return False

        grace = (
            self.refresh_grace_seconds
            if grace_seconds is None
            else max(grace_seconds, 0.0)
        )

        return seconds_remaining <= grace

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

    # Subclasses should implement protocol methods

    @abstractmethod
    async def refresh_token(self) -> CredentialsT | None:
        """Refresh the access token using the refresh token.

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
            status["expires_at"] = expires_at.isoformat()
            seconds_remaining = self.seconds_until_expiration(credentials)
            if seconds_remaining is not None:
                status["expires_in"] = max(0, int(seconds_remaining))

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
        snapshot = self._safe_token_snapshot(credentials)
        if snapshot:
            return snapshot.expires_at
        return None

    # ==================== Unified Profile Support ====================

    async def get_profile(self) -> Any:
        """Get profile information.

        To be implemented by provider-specific managers.
        Returns provider-specific profile model.
        """
        return None

    async def get_profile_quick(self) -> Any:
        """Get profile information without performing I/O or network when possible.

        Default behavior returns any cached profile stored on the manager.
        Provider implementations may override to derive lightweight profiles
        directly from credentials (e.g., JWT claims) without remote calls.

        Returns:
            Provider-specific profile model or None if unavailable
        """
        # Return cached profile if a subclass maintains one
        return getattr(self, "_profile_cache", None)

    async def get_unified_profile(self) -> dict[str, Any]:
        """Get profile in a unified format across all providers.

        Returns:
            Dictionary with standardized fields plus provider-specific extras
        """
        profile = await self.get_profile()
        if not profile:
            return {}

        extras = getattr(profile, "extras", None)
        if extras is None:
            extras = getattr(profile, "features", {}) or {}

        return {
            "account_id": profile.account_id,
            "email": profile.email,
            "display_name": profile.display_name,
            "provider": profile.provider_type,
            "extras": extras,
        }

    async def get_unified_profile_quick(self) -> dict[str, Any]:
        """Get a lightweight unified profile across providers.

        Uses cached or locally derivable data only. Implementations can
        override get_profile_quick() to provide provider-specific logic.

        Returns:
            Dictionary with standardized fields or empty dict if unavailable
        """
        profile = await self.get_profile_quick()
        if not profile:
            return {}

        extras = getattr(profile, "extras", None)
        if extras is None:
            extras = getattr(profile, "features", {}) or {}

        return {
            "account_id": getattr(profile, "account_id", ""),
            "email": getattr(profile, "email", ""),
            "display_name": getattr(profile, "display_name", None),
            "provider": getattr(profile, "provider_type", "unknown"),
            "extras": extras,
        }

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

        # Clear credentials cache so next access rechecks storage
        self._credentials_cache = None
        self._credentials_loaded_at = None

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

        snapshot = self._safe_token_snapshot(credentials)
        if snapshot and snapshot.access_token:
            return snapshot.access_token

        return None

    @async_ttl_cache(ttl=60.0)  # Cache auth status for 1 minute
    async def get_cached_auth_status(self) -> dict[str, Any]:
        """Get current authentication status with caching.

        This is a convenience method that wraps get_auth_status() with caching.

        Returns:
            Dictionary with authentication status information
        """
        return await self.get_auth_status()
