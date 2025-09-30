"""Tests for the Codex/OpenAI token manager."""

from __future__ import annotations

import base64
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from pydantic import SecretStr

from ccproxy.auth.exceptions import OAuthTokenRefreshError
from ccproxy.auth.storage.generic import GenericJsonStorage
from ccproxy.plugins.oauth_codex.manager import CodexTokenManager
from ccproxy.plugins.oauth_codex.models import OpenAICredentials, OpenAITokens


def _forge_jwt(expiry: datetime) -> str:
    """Create a minimal unsigned JWT with the given expiration time."""
    header = base64.urlsafe_b64encode(
        json.dumps({"alg": "HS256", "typ": "JWT"}).encode()
    ).rstrip(b"=")
    payload = base64.urlsafe_b64encode(
        json.dumps({"exp": int(expiry.timestamp()), "sub": "user@test"}).encode()
    ).rstrip(b"=")
    signature = base64.urlsafe_b64encode(b"signature").rstrip(b"=")
    return f"{header.decode()}.{payload.decode()}.{signature.decode()}"


@pytest.mark.asyncio
async def test_codex_manager_refreshes_before_expiry(tmp_path: Path) -> None:
    """Manager refreshes when the token is within the grace period."""
    storage_path = tmp_path / "codex_refresh.json"
    storage = GenericJsonStorage(storage_path, OpenAICredentials)
    manager = CodexTokenManager(storage=storage)

    soon = datetime.now(UTC) + timedelta(seconds=45)
    initial_token = _forge_jwt(soon)
    initial_credentials = OpenAICredentials(
        OPENAI_API_KEY=None,
        tokens=OpenAITokens(
            id_token=SecretStr(initial_token),
            access_token=SecretStr(initial_token),
            refresh_token=SecretStr("refresh_token"),
            account_id="acct-123",
        ),
        last_refresh=datetime.now(UTC).isoformat(),
        active=True,
    )
    await manager.save_credentials(initial_credentials)

    assert manager.should_refresh(initial_credentials) is True

    refreshed_token = _forge_jwt(datetime.now(UTC) + timedelta(hours=1))
    refreshed_credentials = OpenAICredentials(
        OPENAI_API_KEY=None,
        tokens=OpenAITokens(
            id_token=SecretStr(refreshed_token),
            access_token=SecretStr(refreshed_token),
            refresh_token=SecretStr("refresh_token"),
            account_id="acct-123",
        ),
        last_refresh=datetime.now(UTC).isoformat(),
        active=True,
    )

    async def _refresh() -> OpenAICredentials:
        await manager.save_credentials(refreshed_credentials)
        return refreshed_credentials

    manager.refresh_token = AsyncMock(side_effect=_refresh)

    token = await manager.get_access_token_with_refresh()

    assert token != initial_token
    assert manager.refresh_token.await_count == 1
    stored = await manager.load_credentials()
    assert stored is not None
    assert stored.access_token == token
    ttl = manager.seconds_until_expiration(stored)
    assert ttl is not None and ttl > 3000


@pytest.mark.asyncio
async def test_codex_manager_raises_on_refresh_failure(tmp_path: Path) -> None:
    """Manager raises when refresh fails, leaving stored token untouched."""
    storage_path = tmp_path / "codex_refresh_fail.json"
    storage = GenericJsonStorage(storage_path, OpenAICredentials)
    manager = CodexTokenManager(storage=storage)

    soon = datetime.now(UTC) + timedelta(seconds=45)
    initial_token = _forge_jwt(soon)
    credentials = OpenAICredentials(
        OPENAI_API_KEY=None,
        tokens=OpenAITokens(
            id_token=SecretStr(initial_token),
            access_token=SecretStr(initial_token),
            refresh_token=SecretStr("refresh_token"),
            account_id="acct-123",
        ),
        last_refresh=datetime.now(UTC).isoformat(),
        active=True,
    )
    await manager.save_credentials(credentials)

    manager.refresh_token = AsyncMock(return_value=None)

    with pytest.raises(OAuthTokenRefreshError):
        await manager.get_access_token_with_refresh()

    stored = await manager.load_credentials()
    assert stored is not None
    assert stored.access_token == initial_token
