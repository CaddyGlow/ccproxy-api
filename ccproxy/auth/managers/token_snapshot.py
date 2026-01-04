"""Shared token snapshot model for credential managers."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


@dataclass(slots=True)
class TokenSnapshot:
    """Immutable view over sensitive token metadata.

    Token managers return this lightweight structure to share
    credential state without exposing implementation details.
    Secrets should only appear in the access/refresh token fields
    and remain masked when rendered via the helper methods.
    """

    provider: str | None = None
    account_id: str | None = None
    access_token: str | None = None
    refresh_token: str | None = None
    expires_at: datetime | None = None
    scopes: tuple[str, ...] = ()
    extras: dict[str, Any] = field(default_factory=dict)

    def has_access_token(self) -> bool:
        """Whether an access token is present."""
        return bool(self.access_token)

    def has_refresh_token(self) -> bool:
        """Whether a refresh token is present."""
        return bool(self.refresh_token)

    def access_token_preview(self, visible: int = 8) -> str | None:
        """Return a masked preview of the access token."""
        token = self.access_token
        if not token:
            return None
        if visible <= 0 or len(token) <= visible * 2:
            return "*" * len(token)
        return f"{token[:visible]}...{token[-visible:]}"

    def refresh_token_preview(self, visible: int = 4) -> str | None:
        """Return a masked preview of the refresh token."""
        token = self.refresh_token
        if not token:
            return None
        if visible <= 0 or len(token) <= visible * 2:
            return "*" * len(token)
        return f"{token[:visible]}...{token[-visible:]}"

    def expires_in_seconds(self) -> int | None:
        """Return seconds until expiration when available."""
        if not self.expires_at:
            return None
        now = datetime.now(UTC)
        delta = self.expires_at - now
        return max(0, int(delta.total_seconds()))

    def with_scopes(self, scopes: Iterable[str]) -> TokenSnapshot:
        """Return a copy with the provided scopes tuple."""
        scope_tuple = tuple(scope for scope in scopes if scope)
        return TokenSnapshot(
            provider=self.provider,
            account_id=self.account_id,
            access_token=self.access_token,
            refresh_token=self.refresh_token,
            expires_at=self.expires_at,
            scopes=scope_tuple,
            extras=dict(self.extras),
        )


__all__ = ["TokenSnapshot"]
