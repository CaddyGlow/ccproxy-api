"""Tests for auth validate --renew functionality."""

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from typer.testing import CliRunner

from ccproxy.cli.commands.auth import app
from ccproxy.services.credentials.models import ClaudeCredentials, OAuthToken


class TestAuthValidateRenew:
    """Test auth validate --renew command."""

    @pytest.fixture
    def runner(self):
        """Create a CLI runner."""
        return CliRunner()

    @pytest.fixture
    def expired_credentials(self):
        """Create expired credentials."""
        past_time = datetime.now(UTC) - timedelta(hours=1)
        past_ms = int(past_time.timestamp() * 1000)

        return ClaudeCredentials(
            claudeAiOauth=OAuthToken(
                accessToken="expired-token",
                refreshToken="test-refresh-token",
                expiresAt=past_ms,
                scopes=["user:inference"],
                subscriptionType="pro",
            )
        )

    @pytest.fixture
    def expiring_credentials(self):
        """Create credentials expiring in 30 minutes."""
        future_time = datetime.now(UTC) + timedelta(minutes=30)
        future_ms = int(future_time.timestamp() * 1000)

        return ClaudeCredentials(
            claudeAiOauth=OAuthToken(
                accessToken="expiring-token",
                refreshToken="test-refresh-token",
                expiresAt=future_ms,
                scopes=["user:inference"],
                subscriptionType="pro",
            )
        )

    @pytest.fixture
    def valid_credentials(self):
        """Create valid credentials with plenty of time left."""
        future_time = datetime.now(UTC) + timedelta(hours=12)
        future_ms = int(future_time.timestamp() * 1000)

        return ClaudeCredentials(
            claudeAiOauth=OAuthToken(
                accessToken="valid-token",
                refreshToken="test-refresh-token",
                expiresAt=future_ms,
                scopes=["user:inference"],
                subscriptionType="pro",
            )
        )

    def test_validate_expired_without_renew(self, runner, expired_credentials):
        """Test validate shows expired message without --renew."""
        with patch(
            "ccproxy.cli.commands.auth.get_credentials_manager"
        ) as mock_get_manager:
            mock_manager = MagicMock()
            mock_manager.validate = AsyncMock(
                return_value={
                    "valid": True,
                    "expired": True,
                    "subscription_type": "pro",
                    "expires_at": expired_credentials.claude_ai_oauth.expires_at_datetime.isoformat(),
                    "scopes": ["user:inference"],
                }
            )
            mock_get_manager.return_value = mock_manager

            result = runner.invoke(app, ["validate"])

            assert result.exit_code == 0
            assert "expired" in result.output.lower()
            assert (
                "Use --renew flag to automatically refresh the token" in result.output
            )

    def test_validate_expired_with_renew_success(
        self, runner, expired_credentials, valid_credentials
    ):
        """Test validate with --renew successfully refreshes expired token."""
        with patch(
            "ccproxy.cli.commands.auth.get_credentials_manager"
        ) as mock_get_manager:
            mock_manager = MagicMock()

            # First validation returns expired
            mock_manager.validate = AsyncMock(
                side_effect=[
                    {
                        "valid": True,
                        "expired": True,
                        "subscription_type": "pro",
                        "expires_at": expired_credentials.claude_ai_oauth.expires_at_datetime.isoformat(),
                        "scopes": ["user:inference"],
                    },
                    # After refresh, return valid
                    {
                        "valid": True,
                        "expired": False,
                        "subscription_type": "pro",
                        "expires_at": valid_credentials.claude_ai_oauth.expires_at_datetime.isoformat(),
                        "scopes": ["user:inference"],
                    },
                ]
            )

            # get_valid_credentials triggers refresh
            mock_manager.get_valid_credentials = AsyncMock(
                return_value=valid_credentials
            )
            mock_get_manager.return_value = mock_manager

            result = runner.invoke(app, ["validate", "--renew"])

            assert result.exit_code == 0
            assert "Token expired or expiring soon, refreshing..." in result.output
            assert "Token refreshed successfully" in result.output
            assert "Valid Claude credentials found" in result.output

    def test_validate_expiring_with_renew(
        self, runner, expiring_credentials, valid_credentials
    ):
        """Test validate with --renew refreshes token expiring soon."""
        with patch(
            "ccproxy.cli.commands.auth.get_credentials_manager"
        ) as mock_get_manager:
            mock_manager = MagicMock()

            # First validation returns expiring (not expired)
            mock_manager.validate = AsyncMock(
                side_effect=[
                    {
                        "valid": True,
                        "expired": False,
                        "subscription_type": "pro",
                        "expires_at": expiring_credentials.claude_ai_oauth.expires_at_datetime.isoformat(),
                        "scopes": ["user:inference"],
                    },
                    # After refresh, return valid with more time
                    {
                        "valid": True,
                        "expired": False,
                        "subscription_type": "pro",
                        "expires_at": valid_credentials.claude_ai_oauth.expires_at_datetime.isoformat(),
                        "scopes": ["user:inference"],
                    },
                ]
            )

            mock_manager.get_valid_credentials = AsyncMock(
                return_value=valid_credentials
            )
            mock_get_manager.return_value = mock_manager

            result = runner.invoke(app, ["validate", "--renew"])

            assert result.exit_code == 0
            assert "Token expired or expiring soon, refreshing..." in result.output
            assert "Token refreshed successfully" in result.output

    def test_validate_valid_with_renew_no_refresh(self, runner, valid_credentials):
        """Test validate with --renew doesn't refresh valid token with plenty of time."""
        with patch(
            "ccproxy.cli.commands.auth.get_credentials_manager"
        ) as mock_get_manager:
            mock_manager = MagicMock()

            # Validation returns valid with plenty of time left
            mock_manager.validate = AsyncMock(
                return_value={
                    "valid": True,
                    "expired": False,
                    "subscription_type": "pro",
                    "expires_at": valid_credentials.claude_ai_oauth.expires_at_datetime.isoformat(),
                    "scopes": ["user:inference"],
                }
            )

            mock_manager.get_valid_credentials = AsyncMock()
            mock_get_manager.return_value = mock_manager

            result = runner.invoke(app, ["validate", "--renew"])

            assert result.exit_code == 0
            # Should not attempt refresh
            assert "refreshing" not in result.output.lower()
            assert "Valid Claude credentials found" in result.output
            mock_manager.get_valid_credentials.assert_not_called()

    def test_validate_renew_failure(self, runner, expired_credentials):
        """Test validate with --renew handles refresh failure gracefully."""
        with patch(
            "ccproxy.cli.commands.auth.get_credentials_manager"
        ) as mock_get_manager:
            mock_manager = MagicMock()

            mock_manager.validate = AsyncMock(
                return_value={
                    "valid": True,
                    "expired": True,
                    "subscription_type": "pro",
                    "expires_at": expired_credentials.claude_ai_oauth.expires_at_datetime.isoformat(),
                    "scopes": ["user:inference"],
                }
            )

            # Refresh fails
            mock_manager.get_valid_credentials = AsyncMock(
                side_effect=Exception("Network error")
            )
            mock_get_manager.return_value = mock_manager

            result = runner.invoke(app, ["validate", "--renew"])

            assert result.exit_code == 0
            assert "Failed to refresh token: Network error" in result.output
            # Should still show the expired credentials info
            assert "expired" in result.output.lower()
