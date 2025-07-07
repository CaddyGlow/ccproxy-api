"""Tests for auth CLI commands."""

from datetime import UTC, datetime, timedelta, timezone
from unittest.mock import patch

from typer.testing import CliRunner

from claude_code_proxy.cli.commands.auth import app


class TestAuthCommands:
    """Test auth CLI commands."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    @patch(
        "claude_code_proxy.cli.commands.auth.CredentialsService.validate_credentials"
    )
    def test_validate_command_success(self, mock_validate):
        """Test successful credential validation."""
        future_time = datetime.now(UTC) + timedelta(days=7)

        mock_validate.return_value = {
            "valid": True,
            "expired": False,
            "subscription_type": "max",
            "expires_at": future_time.isoformat(),
            "scopes": ["user:inference", "user:profile"],
        }

        result = self.runner.invoke(app, ["validate"])

        assert result.exit_code == 0
        assert "Valid" in result.stdout
        assert "max" in result.stdout
        assert "Valid Claude credentials found" in result.stdout

    @patch(
        "claude_code_proxy.cli.commands.auth.CredentialsService.validate_credentials"
    )
    def test_validate_command_expired(self, mock_validate):
        """Test validation with expired credentials."""
        past_time = datetime.now(UTC) - timedelta(days=1)

        mock_validate.return_value = {
            "valid": True,
            "expired": True,
            "subscription_type": "pro",
            "expires_at": past_time.isoformat(),
            "scopes": ["user:inference"],
        }

        result = self.runner.invoke(app, ["validate"])

        assert result.exit_code == 0
        assert "Expired" in result.stdout
        assert "pro" in result.stdout
        assert "credentials found but expired" in result.stdout

    @patch(
        "claude_code_proxy.cli.commands.auth.CredentialsService.validate_credentials"
    )
    def test_validate_command_not_found(self, mock_validate):
        """Test validation when credentials not found."""
        mock_validate.return_value = {
            "valid": False,
            "error": "No credentials file found in ~/.claude/credentials.json",
        }

        result = self.runner.invoke(app, ["validate"])

        assert result.exit_code == 0
        assert "No credentials file found" in result.stdout
        assert "claude login" in result.stdout

    @patch(
        "claude_code_proxy.cli.commands.auth.CredentialsService.validate_credentials"
    )
    def test_validate_command_error(self, mock_validate):
        """Test validation with exception."""
        mock_validate.side_effect = Exception("Test error")

        result = self.runner.invoke(app, ["validate"])

        assert result.exit_code == 1
        assert "Error validating credentials" in result.output

    @patch(
        "claude_code_proxy.cli.commands.auth.CredentialsService.find_credentials_file"
    )
    @patch("claude_code_proxy.cli.commands.auth.CredentialsService.load_credentials")
    def test_info_command_success(self, mock_load, mock_find):
        """Test info command with valid credentials."""
        from pathlib import Path

        from claude_code_proxy.services.credentials import ClaudeCredentials, OAuthToken

        mock_find.return_value = Path("/home/test/.claude/credentials.json")

        future_time = datetime.now(UTC) + timedelta(days=7)
        future_ms = int(future_time.timestamp() * 1000)

        mock_creds = ClaudeCredentials.model_validate(
            {
                "claudeAiOauth": {
                    "accessToken": "test-access-token-very-long",
                    "refreshToken": "refresh-token",
                    "expiresAt": future_ms,
                    "scopes": ["user:inference", "user:profile"],
                    "subscriptionType": "max",
                }
            }
        )
        mock_load.return_value = mock_creds

        result = self.runner.invoke(app, ["info"])

        assert result.exit_code == 0
        assert "/home/test/.claude/credentials.json" in result.stdout
        assert "max" in result.stdout
        assert "No" in result.stdout  # Not expired
        assert "test-acc...ery-long" in result.stdout  # Token preview

    @patch(
        "claude_code_proxy.cli.commands.auth.CredentialsService.find_credentials_file"
    )
    def test_info_command_not_found(self, mock_find):
        """Test info command when credentials not found."""
        mock_find.return_value = None

        result = self.runner.invoke(app, ["info"])

        assert result.exit_code == 1
        assert "No credential file found" in result.output
        assert "Expected locations:" in result.stdout

    @patch(
        "claude_code_proxy.cli.commands.auth.CredentialsService.find_credentials_file"
    )
    @patch("claude_code_proxy.cli.commands.auth.CredentialsService.load_credentials")
    def test_info_command_load_error(self, mock_load, mock_find):
        """Test info command with load error."""
        from pathlib import Path

        mock_find.return_value = Path("/home/test/.claude/credentials.json")
        mock_load.return_value = None

        result = self.runner.invoke(app, ["info"])

        assert result.exit_code == 1
        assert "Failed to load credentials" in result.output
