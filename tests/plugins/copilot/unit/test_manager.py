"""Unit tests for Copilot token manager refresh behaviour."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

from pydantic import SecretStr

from ccproxy.plugins.copilot.manager import CopilotTokenManager
from ccproxy.plugins.copilot.oauth.models import (
    CopilotCredentials,
    CopilotOAuthToken,
    CopilotTokenResponse,
)
from ccproxy.plugins.copilot.oauth.storage import CopilotOAuthStorage


def _build_oauth_token(now: datetime) -> CopilotOAuthToken:
    """Helper to create a valid OAuth token scoped for tests."""

    return CopilotOAuthToken(
        access_token=SecretStr("oauth-token"),
        token_type="bearer",
        scope="read:user",
        created_at=int(now.timestamp()),
        expires_in=3600,
    )


def _build_credentials(
    *,
    now: datetime,
    refresh_in: int,
    updated_offset: timedelta,
) -> CopilotCredentials:
    """Create Copilot credentials with consistent timestamps for assertions."""

    oauth_token = _build_oauth_token(now)
    copilot_token = CopilotTokenResponse(
        token=SecretStr("copilot-token"),
        expires_at=now + timedelta(hours=2),
        refresh_in=refresh_in,
    )
    updated_at = int((now - updated_offset).timestamp())
    return CopilotCredentials(
        oauth_token=oauth_token,
        copilot_token=copilot_token,
        account_type="individual",
        created_at=updated_at,
        updated_at=updated_at,
    )


def test_is_expired_uses_refresh_window(tmp_path: Path) -> None:
    """Manager should treat refresh window crossing as expiration trigger."""

    storage = CopilotOAuthStorage(credentials_path=tmp_path / "credentials.json")
    manager = CopilotTokenManager(storage=storage)

    now = datetime.now(UTC)

    within_window = _build_credentials(
        now=now,
        refresh_in=1500,
        updated_offset=timedelta(seconds=10),
    )
    assert manager.is_expired(within_window) is False

    past_window = within_window.model_copy(
        update={
            "updated_at": int((now - timedelta(seconds=2000)).timestamp()),
        }
    )
    assert manager.is_expired(past_window) is True


def test_get_expiration_time_prefers_refresh_deadline(tmp_path: Path) -> None:
    """Earliest refresh deadline should be reported as expiration time."""

    storage = CopilotOAuthStorage(credentials_path=tmp_path / "credentials.json")
    manager = CopilotTokenManager(storage=storage)

    now = datetime.now(UTC)
    credentials = _build_credentials(
        now=now,
        refresh_in=900,
        updated_offset=timedelta(seconds=60),
    )

    expected_deadline = datetime.fromtimestamp(
        credentials.updated_at + 900,
        tz=UTC,
    )

    expiration_time = manager.get_expiration_time(credentials)
    assert expiration_time == expected_deadline
