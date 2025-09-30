"""File-backed AuthManager adapter for legacy token snapshot files."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Any, cast

from ccproxy.auth.exceptions import AuthenticationError
from ccproxy.auth.managers.token_snapshot import TokenSnapshot
from ccproxy.auth.models.credentials import BaseCredentials
from ccproxy.auth.oauth.protocol import StandardProfileFields
from ccproxy.core.logging import TraceBoundLogger, get_plugin_logger


logger = get_plugin_logger(__name__)


def _parse_datetime(value: Any) -> datetime | None:
    """Parse various datetime formats into timezone-aware datetime."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, int | float):
        return datetime.fromtimestamp(value, tz=UTC)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            return datetime.fromisoformat(text)
        except ValueError:
            return None
    return None


async def _read_snapshot_from_file(path: Path) -> dict[str, Any] | None:
    """Read token snapshot from JSON file."""

    def _load() -> dict[str, Any] | None:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            logger.warning("snapshot_not_mapping", path=str(path))
            return None
        return cast(dict[str, Any], data)

    try:
        return await asyncio.to_thread(_load)
    except FileNotFoundError:
        return None
    except json.JSONDecodeError as exc:
        logger.warning("invalid_json", path=str(path), error=str(exc))
        return None
    except PermissionError as exc:
        logger.warning("permission_denied", path=str(path), error=str(exc))
        return None
    except OSError as exc:
        logger.warning("read_failed", path=str(path), error=str(exc))
        return None


class FileBackedAuthManager:
    """AuthManager implementation that reads tokens from JSON snapshot files.

    This adapter provides backward compatibility with file-based token snapshots
    while implementing the AuthManager protocol for composition.
    """

    def __init__(
        self,
        path: Path,
        provider: str,
        *,
        logger: TraceBoundLogger | None = None,
    ) -> None:
        """Initialize file-backed auth manager.

        Args:
            path: Path to JSON token snapshot file
            provider: Provider name (e.g., "claude-api")
            logger: Optional logger for this manager
        """
        self.path = path
        self.provider = provider
        self.logger = (logger or get_plugin_logger(__name__)).bind(
            path=str(path), provider=provider
        )
        self._snapshot: TokenSnapshot | None = None
        self._mtime_ns: int | None = None

    async def _load_snapshot(self, *, force: bool = False) -> TokenSnapshot | None:
        """Load token snapshot from file, with mtime-based caching."""
        try:
            stat_result = await asyncio.to_thread(self.path.stat)
            mtime_ns = stat_result.st_mtime_ns
        except FileNotFoundError:
            self._snapshot = None
            self._mtime_ns = None
            return None
        except PermissionError as exc:
            self.logger.warning("stat_permission_denied", error=str(exc))
            return self._snapshot
        except OSError as exc:
            self.logger.warning("stat_failed", error=str(exc))
            return self._snapshot

        need_reload = force or self._snapshot is None or mtime_ns != self._mtime_ns
        if not need_reload:
            return self._snapshot

        raw = await _read_snapshot_from_file(self.path)
        if raw is None:
            self._snapshot = None
            self._mtime_ns = mtime_ns
            return None

        # Parse snapshot data
        extras = (
            dict(raw.get("extras", {})) if isinstance(raw.get("extras"), dict) else {}
        )

        tokens = raw.get("tokens")
        token_data = tokens if isinstance(tokens, dict) else {}

        provider = str(raw.get("provider", self.provider))
        account_id = raw.get("account_id") or token_data.get("account_id")
        access_token = raw.get("access_token") or token_data.get("access_token")
        refresh_token = raw.get("refresh_token") or token_data.get("refresh_token")

        expires_at = _parse_datetime(raw.get("expires_at"))
        if expires_at is None:
            expires_at = _parse_datetime(token_data.get("expires_at"))

        scopes = raw.get("scopes")
        if not scopes and isinstance(token_data.get("scopes"), list):
            scopes = token_data.get("scopes")

        if token_data and "tokens" not in extras:
            extras.setdefault("tokens", token_data)

        snapshot = TokenSnapshot(
            provider=provider,
            account_id=account_id,
            access_token=access_token,
            refresh_token=refresh_token,
            expires_at=expires_at,
            scopes=tuple(scopes if scopes else ()),
            extras=extras,
        )

        self._snapshot = snapshot
        self._mtime_ns = mtime_ns
        self.logger.debug("snapshot_loaded", provider=snapshot.provider)
        return snapshot

    async def get_token_snapshot(self) -> TokenSnapshot | None:
        """Get token snapshot (for compatibility with should_refresh checks).

        Returns:
            TokenSnapshot if available, None otherwise
        """
        return await self._load_snapshot()

    # ==================== AuthManager Protocol Implementation ====================

    async def get_access_token(self) -> str:
        """Get valid access token.

        Returns:
            Access token string

        Raises:
            AuthenticationError: If no valid token available
        """
        snapshot = await self._load_snapshot()
        if not snapshot or not snapshot.access_token:
            raise AuthenticationError(f"No valid access token in file: {self.path}")
        return str(snapshot.access_token)

    async def get_credentials(self) -> BaseCredentials:
        """Get credentials (not supported for file-backed managers).

        Raises:
            AuthenticationError: Always, as file-backed managers don't expose credentials
        """
        raise AuthenticationError(
            "File-backed manager does not expose provider-specific credential models"
        )

    async def is_authenticated(self) -> bool:
        """Check if current authentication is valid.

        Returns:
            True if a valid token is present, False otherwise
        """
        snapshot = await self._load_snapshot()
        return bool(snapshot and snapshot.access_token)

    async def get_user_profile(self) -> StandardProfileFields | None:
        """Get user profile (not available for file-backed managers).

        Returns:
            None, as file-backed managers don't provide profile information
        """
        return None

    async def validate_credentials(self) -> bool:
        """Validate that credentials are available and valid.

        Returns:
            True if valid token present, False otherwise
        """
        return await self.is_authenticated()

    def get_provider_name(self) -> str:
        """Get the provider name.

        Returns:
            Provider name string
        """
        return self.provider

    async def __aenter__(self) -> FileBackedAuthManager:
        """Async context manager entry."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Async context manager exit."""
        return None


__all__ = ["FileBackedAuthManager"]
