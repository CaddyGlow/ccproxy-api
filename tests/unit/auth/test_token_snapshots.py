"""Tests for token snapshot helpers and utilities."""

from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

from pydantic import SecretStr

from ccproxy.cli.commands.auth import _token_snapshot_from_credentials
from ccproxy.plugins.copilot.oauth.models import (
    CopilotCredentials,
    CopilotOAuthToken,
    CopilotTokenResponse,
)
from ccproxy.plugins.oauth_claude.models import ClaudeCredentials, ClaudeOAuthToken
from ccproxy.plugins.oauth_codex.models import OpenAICredentials, OpenAITokens


def test_snapshot_from_claude_credentials(tmp_path: Path) -> None:
    """Ensure Claude credentials produce a populated snapshot."""
    oauth = ClaudeOAuthToken(
        accessToken=SecretStr("claude_access"),
        refreshToken=SecretStr("claude_refresh"),
        expiresAt=int(datetime.now(UTC).timestamp() * 1000) + 60000,
        scopes=["read", "write"],
        subscriptionType="pro",
    )
    credentials = ClaudeCredentials(claudeAiOauth=oauth)

    with patch("pathlib.Path.home", return_value=tmp_path):
        snapshot = _token_snapshot_from_credentials(credentials, "claude-api")
    assert snapshot is not None
    assert snapshot.provider == "claude-api"
    assert snapshot.access_token == "claude_access"
    assert snapshot.refresh_token == "claude_refresh"
    assert snapshot.scopes == ("read", "write")
    assert snapshot.extras.get("subscription_type") == "pro"


def test_snapshot_from_openai_credentials() -> None:
    """Ensure OpenAI credentials populate expected snapshot fields."""
    tokens = OpenAITokens(
        id_token=SecretStr("id.jwt"),
        access_token=SecretStr("access.jwt"),
        refresh_token=SecretStr("refresh-token"),
        account_id="acct-123",
    )
    credentials = OpenAICredentials(
        OPENAI_API_KEY=None,
        tokens=tokens,
        last_refresh=datetime.now(UTC).isoformat(),
        active=True,
    )

    snapshot = _token_snapshot_from_credentials(credentials, "codex")
    assert snapshot is not None
    assert snapshot.provider == "codex"
    assert snapshot.account_id == "acct-123"
    assert snapshot.access_token == "access.jwt"
    assert snapshot.refresh_token == "refresh-token"
    assert snapshot.extras.get("id_token_present") is True


def test_snapshot_from_copilot_credentials() -> None:
    """Ensure Copilot credentials derive access and refresh tokens."""
    oauth_token = CopilotOAuthToken(
        access_token=SecretStr("gh_oauth"),
        token_type="bearer",
        expires_in=3600,
        refresh_token=SecretStr("gh_refresh"),
        scope="read:user gist",
        created_at=int(datetime.now(UTC).timestamp()),
    )
    copilot_token = CopilotTokenResponse(
        token=SecretStr("copilot_service"),
        expires_at=datetime.now(UTC) + timedelta(hours=1),
    )
    credentials = CopilotCredentials(
        oauth_token=oauth_token,
        copilot_token=copilot_token,
        account_type="individual",
    )

    snapshot = _token_snapshot_from_credentials(credentials, "copilot")
    assert snapshot is not None
    assert snapshot.provider == "copilot"
    assert snapshot.access_token == "copilot_service"
    assert snapshot.refresh_token == "gh_refresh"
    assert snapshot.scopes == ("read:user", "gist")
    assert snapshot.extras.get("has_copilot_token") is True
