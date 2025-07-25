"""Tests for CLI authentication commands.

This module tests the CLI authentication commands in ccproxy/cli/commands/auth.py,
including validate, info, login, and renew commands with proper type safety.
"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from typer.testing import CliRunner

from ccproxy.auth.models import (
    AccountInfo,
    ClaudeCredentials,
    OAuthToken,
    OrganizationInfo,
    UserProfile,
    ValidationResult,
)
from ccproxy.cli.commands.auth import (
    app,
    credential_info,
    get_credentials_manager,
    get_docker_credential_paths,
    login_command,
    renew,
    validate_credentials,
)
from ccproxy.services.credentials.manager import CredentialsManager


class TestAuthCLICommands:
    """Test CLI authentication commands."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create CLI test runner."""
        return CliRunner(env={"NO_COLOR": "1"})

    @pytest.fixture
    def mock_credentials_manager(self) -> AsyncMock:
        """Create mock credentials manager."""
        mock = AsyncMock(spec=CredentialsManager)
        return mock

    @pytest.fixture
    def mock_oauth_token(self) -> OAuthToken:
        """Create mock OAuth token."""
        return OAuthToken(
            accessToken="sk-test-token-123",
            refreshToken="refresh-token-456",
            expiresAt=None,
            tokenType="Bearer",
            subscriptionType="pro",
            scopes=["chat", "completions"],
        )

    @pytest.fixture
    def mock_credentials(self, mock_oauth_token: OAuthToken) -> ClaudeCredentials:
        """Create mock Claude credentials."""
        return ClaudeCredentials(claudeAiOauth=mock_oauth_token)

    @pytest.fixture
    def mock_validation_result_valid(
        self, mock_credentials: ClaudeCredentials
    ) -> ValidationResult:
        """Create valid validation result."""
        return ValidationResult(
            valid=True,
            expired=False,
            path="/home/user/.claude/credentials.json",
            credentials=mock_credentials,
        )

    @pytest.fixture
    def mock_validation_result_invalid(self) -> ValidationResult:
        """Create invalid validation result."""
        return ValidationResult(
            valid=False,
            expired=False,
            path=None,
            credentials=None,
        )

    @pytest.fixture
    def mock_user_profile(self) -> UserProfile:
        """Create mock user profile."""
        account = AccountInfo(
            uuid="user-123",
            email="test@example.com",
            full_name="Test User",
            display_name="testuser",
            has_claude_pro=True,
            has_claude_max=False,
        )
        organization = OrganizationInfo(
            uuid="org-456",
            name="Test Organization",
            organization_type="business",
            billing_type="monthly",
            rate_limit_tier="tier1",
        )
        return UserProfile(account=account, organization=organization)


class TestGetCredentialsManager(TestAuthCLICommands):
    """Test get_credentials_manager helper function."""

    @patch("ccproxy.cli.commands.auth.get_settings")
    def test_get_credentials_manager_default_paths(
        self, mock_get_settings: MagicMock
    ) -> None:
        """Test get_credentials_manager with default paths."""
        mock_settings = MagicMock()
        mock_get_settings.return_value = mock_settings

        with patch("ccproxy.cli.commands.auth.CredentialsManager") as mock_cm:
            manager = get_credentials_manager()

            mock_get_settings.assert_called_once()
            mock_cm.assert_called_once_with(config=mock_settings.auth)

    @patch("ccproxy.cli.commands.auth.get_settings")
    def test_get_credentials_manager_custom_paths(
        self, mock_get_settings: MagicMock
    ) -> None:
        """Test get_credentials_manager with custom paths."""
        mock_settings = MagicMock()
        mock_get_settings.return_value = mock_settings
        custom_paths = [Path("/custom/path/credentials.json")]

        with patch("ccproxy.cli.commands.auth.CredentialsManager") as mock_cm:
            manager = get_credentials_manager(custom_paths)

            mock_get_settings.assert_called_once()
            assert mock_settings.auth.storage.storage_paths == custom_paths
            mock_cm.assert_called_once_with(config=mock_settings.auth)


class TestGetDockerCredentialPaths(TestAuthCLICommands):
    """Test get_docker_credential_paths helper function."""

    @patch("ccproxy.cli.commands.auth.get_claude_docker_home_dir")
    def test_get_docker_credential_paths(self, mock_get_docker_home: MagicMock) -> None:
        """Test Docker credential paths generation."""
        mock_get_docker_home.return_value = "/docker/home"

        paths = get_docker_credential_paths()

        expected_paths = [
            Path("/docker/home/.claude/.credentials.json"),
            Path("/docker/home/.config/claude/.credentials.json"),
            Path(".credentials.json"),
        ]
        assert paths == expected_paths
        mock_get_docker_home.assert_called_once()


class TestValidateCredentialsCommand(TestAuthCLICommands):
    """Test validate credentials CLI command."""

    @patch("ccproxy.cli.commands.auth.get_credentials_manager")
    def test_validate_credentials_valid(
        self,
        mock_get_manager: MagicMock,
        runner: CliRunner,
        mock_credentials_manager: AsyncMock,
        mock_validation_result_valid: ValidationResult,
    ) -> None:
        """Test validate command with valid credentials."""
        mock_get_manager.return_value = mock_credentials_manager
        mock_credentials_manager.validate.return_value = mock_validation_result_valid

        result = runner.invoke(app, ["validate"])

        assert result.exit_code == 0
        assert "Valid Claude credentials found" in result.stdout
        mock_credentials_manager.validate.assert_called_once()

    @patch("ccproxy.cli.commands.auth.get_credentials_manager")
    def test_validate_credentials_invalid(
        self,
        mock_get_manager: MagicMock,
        runner: CliRunner,
        mock_credentials_manager: AsyncMock,
        mock_validation_result_invalid: ValidationResult,
    ) -> None:
        """Test validate command with invalid credentials."""
        mock_get_manager.return_value = mock_credentials_manager
        mock_credentials_manager.validate.return_value = mock_validation_result_invalid

        result = runner.invoke(app, ["validate"])

        assert result.exit_code == 0
        assert "No credentials file found" in result.stdout
        mock_credentials_manager.validate.assert_called_once()

    @patch("ccproxy.cli.commands.auth.get_credentials_manager")
    def test_validate_credentials_docker_flag(
        self,
        mock_get_manager: MagicMock,
        runner: CliRunner,
        mock_credentials_manager: AsyncMock,
        mock_validation_result_valid: ValidationResult,
    ) -> None:
        """Test validate command with --docker flag."""
        mock_get_manager.return_value = mock_credentials_manager
        mock_credentials_manager.validate.return_value = mock_validation_result_valid

        result = runner.invoke(app, ["validate", "--docker"])

        assert result.exit_code == 0
        # Check that get_credentials_manager was called with Docker paths
        mock_get_manager.assert_called_once()
        call_args = mock_get_manager.call_args
        # Check if custom_paths was passed as positional or keyword argument
        if len(call_args[0]) > 0:
            custom_paths = call_args[0][0]
        else:
            custom_paths = call_args.kwargs.get("custom_paths")
        assert custom_paths is not None
        assert any(".claude" in str(path) for path in custom_paths)

    @patch("ccproxy.cli.commands.auth.get_credentials_manager")
    def test_validate_credentials_custom_file(
        self,
        mock_get_manager: MagicMock,
        runner: CliRunner,
        mock_credentials_manager: AsyncMock,
        mock_validation_result_valid: ValidationResult,
    ) -> None:
        """Test validate command with --credential-file flag."""
        mock_get_manager.return_value = mock_credentials_manager
        mock_credentials_manager.validate.return_value = mock_validation_result_valid
        custom_file = "/custom/credentials.json"

        result = runner.invoke(app, ["validate", "--credential-file", custom_file])

        assert result.exit_code == 0
        # Check that get_credentials_manager was called with custom file path
        mock_get_manager.assert_called_once()
        call_args = mock_get_manager.call_args
        # Check if custom_paths was passed as positional or keyword argument
        if len(call_args[0]) > 0:
            custom_paths = call_args[0][0]
        else:
            custom_paths = call_args.kwargs.get("custom_paths")
        assert custom_paths == [Path(custom_file)]

    @patch("ccproxy.cli.commands.auth.get_credentials_manager")
    def test_validate_credentials_exception(
        self,
        mock_get_manager: MagicMock,
        runner: CliRunner,
        mock_credentials_manager: AsyncMock,
    ) -> None:
        """Test validate command with exception."""
        mock_get_manager.return_value = mock_credentials_manager
        mock_credentials_manager.validate.side_effect = Exception("Test error")

        result = runner.invoke(app, ["validate"])

        assert result.exit_code == 1
        assert "Error validating credentials: Test error" in result.stdout


class TestCredentialInfoCommand(TestAuthCLICommands):
    """Test credential info CLI command."""

    @patch("ccproxy.cli.commands.auth.get_credentials_manager")
    def test_credential_info_success(
        self,
        mock_get_manager: MagicMock,
        runner: CliRunner,
        mock_credentials_manager: AsyncMock,
        mock_credentials: ClaudeCredentials,
        mock_user_profile: UserProfile,
    ) -> None:
        """Test info command with successful credential loading."""
        mock_get_manager.return_value = mock_credentials_manager
        mock_credentials_manager.load.return_value = mock_credentials
        mock_credentials_manager.get_account_profile.return_value = mock_user_profile
        mock_credentials_manager.find_credentials_file.return_value = Path(
            "/home/user/.claude/credentials.json"
        )

        result = runner.invoke(app, ["info"])

        assert result.exit_code == 0
        assert "Claude Credential Information" in result.stdout
        assert "test@example.com" in result.stdout
        assert "Test Organization" in result.stdout
        mock_credentials_manager.load.assert_called_once()

    @patch("ccproxy.cli.commands.auth.get_credentials_manager")
    def test_credential_info_no_credentials(
        self,
        mock_get_manager: MagicMock,
        runner: CliRunner,
        mock_credentials_manager: AsyncMock,
    ) -> None:
        """Test info command with no credentials found."""
        mock_get_manager.return_value = mock_credentials_manager
        mock_credentials_manager.load.return_value = None

        result = runner.invoke(app, ["info"])

        assert result.exit_code == 1
        assert "No credential file found" in result.stdout
        mock_credentials_manager.load.assert_called_once()

    @patch("ccproxy.cli.commands.auth.get_credentials_manager")
    def test_credential_info_docker_flag(
        self,
        mock_get_manager: MagicMock,
        runner: CliRunner,
        mock_credentials_manager: AsyncMock,
        mock_credentials: ClaudeCredentials,
    ) -> None:
        """Test info command with --docker flag."""
        mock_get_manager.return_value = mock_credentials_manager
        mock_credentials_manager.load.return_value = mock_credentials
        mock_credentials_manager.get_account_profile.return_value = None
        mock_credentials_manager.find_credentials_file.return_value = Path(
            "/docker/home/.claude/credentials.json"
        )

        result = runner.invoke(app, ["info", "--docker"])

        assert result.exit_code == 0
        # Check that get_credentials_manager was called with Docker paths
        mock_get_manager.assert_called_once()
        call_args = mock_get_manager.call_args
        # Check if custom_paths was passed as positional or keyword argument
        if len(call_args[0]) > 0:
            custom_paths = call_args[0][0]
        else:
            custom_paths = call_args.kwargs.get("custom_paths")
        assert custom_paths is not None


class TestLoginCommand(TestAuthCLICommands):
    """Test login CLI command."""

    @patch("ccproxy.cli.commands.auth.get_credentials_manager")
    def test_login_command_success(
        self,
        mock_get_manager: MagicMock,
        runner: CliRunner,
        mock_credentials_manager: AsyncMock,
        mock_validation_result_invalid: ValidationResult,
        mock_validation_result_valid: ValidationResult,
    ) -> None:
        """Test successful login command."""
        mock_get_manager.return_value = mock_credentials_manager
        # First validation returns invalid (not logged in)
        # Second validation returns valid (after login)
        mock_credentials_manager.validate.side_effect = [
            mock_validation_result_invalid,
            mock_validation_result_valid,
        ]
        mock_credentials_manager.login.return_value = None

        result = runner.invoke(app, ["login"])

        assert result.exit_code == 0
        assert "Successfully logged in to Claude!" in result.stdout
        mock_credentials_manager.login.assert_called_once()

    @patch("ccproxy.cli.commands.auth.get_credentials_manager")
    def test_login_command_already_logged_in_cancel(
        self,
        mock_get_manager: MagicMock,
        runner: CliRunner,
        mock_credentials_manager: AsyncMock,
        mock_validation_result_valid: ValidationResult,
    ) -> None:
        """Test login command when already logged in and user cancels."""
        mock_get_manager.return_value = mock_credentials_manager
        mock_credentials_manager.validate.return_value = mock_validation_result_valid

        # Simulate user saying "no" to overwrite
        result = runner.invoke(app, ["login"], input="n\n")

        assert result.exit_code == 0
        assert "Login cancelled" in result.stdout
        mock_credentials_manager.login.assert_not_called()

    @patch("ccproxy.cli.commands.auth.get_credentials_manager")
    def test_login_command_exception(
        self,
        mock_get_manager: MagicMock,
        runner: CliRunner,
        mock_credentials_manager: AsyncMock,
        mock_validation_result_invalid: ValidationResult,
    ) -> None:
        """Test login command with exception during login."""
        mock_get_manager.return_value = mock_credentials_manager
        mock_credentials_manager.validate.return_value = mock_validation_result_invalid
        mock_credentials_manager.login.side_effect = Exception("Login failed")

        result = runner.invoke(app, ["login"])

        assert result.exit_code == 1
        assert "Login failed. Please try again." in result.stdout


class TestRenewCommand(TestAuthCLICommands):
    """Test renew CLI command."""

    @patch("ccproxy.cli.commands.auth.get_credentials_manager")
    def test_renew_command_success(
        self,
        mock_get_manager: MagicMock,
        runner: CliRunner,
        mock_credentials_manager: AsyncMock,
        mock_validation_result_valid: ValidationResult,
        mock_credentials: ClaudeCredentials,
    ) -> None:
        """Test successful renew command."""
        mock_get_manager.return_value = mock_credentials_manager
        mock_credentials_manager.validate.return_value = mock_validation_result_valid
        mock_credentials_manager.refresh_token.return_value = mock_credentials

        result = runner.invoke(app, ["renew"])

        assert result.exit_code == 0
        assert "Successfully renewed credentials!" in result.stdout
        mock_credentials_manager.refresh_token.assert_called_once()

    @patch("ccproxy.cli.commands.auth.get_credentials_manager")
    def test_renew_command_no_credentials(
        self,
        mock_get_manager: MagicMock,
        runner: CliRunner,
        mock_credentials_manager: AsyncMock,
        mock_validation_result_invalid: ValidationResult,
    ) -> None:
        """Test renew command with no credentials found."""
        mock_get_manager.return_value = mock_credentials_manager
        mock_credentials_manager.validate.return_value = mock_validation_result_invalid

        result = runner.invoke(app, ["renew"])

        assert result.exit_code == 1
        assert "No credentials found to renew" in result.stdout
        mock_credentials_manager.refresh_token.assert_not_called()

    @patch("ccproxy.cli.commands.auth.get_credentials_manager")
    def test_renew_command_refresh_fails(
        self,
        mock_get_manager: MagicMock,
        runner: CliRunner,
        mock_credentials_manager: AsyncMock,
        mock_validation_result_valid: ValidationResult,
    ) -> None:
        """Test renew command when token refresh fails."""
        mock_get_manager.return_value = mock_credentials_manager
        mock_credentials_manager.validate.return_value = mock_validation_result_valid
        mock_credentials_manager.refresh_token.return_value = None

        result = runner.invoke(app, ["renew"])

        assert result.exit_code == 1
        assert "Failed to renew credentials" in result.stdout

    @patch("ccproxy.cli.commands.auth.get_credentials_manager")
    def test_renew_command_docker_flag(
        self,
        mock_get_manager: MagicMock,
        runner: CliRunner,
        mock_credentials_manager: AsyncMock,
        mock_validation_result_valid: ValidationResult,
        mock_credentials: ClaudeCredentials,
    ) -> None:
        """Test renew command with --docker flag."""
        mock_get_manager.return_value = mock_credentials_manager
        mock_credentials_manager.validate.return_value = mock_validation_result_valid
        mock_credentials_manager.refresh_token.return_value = mock_credentials

        result = runner.invoke(app, ["renew", "--docker"])

        assert result.exit_code == 0
        # Check that get_credentials_manager was called with Docker paths
        mock_get_manager.assert_called_once()
        call_args = mock_get_manager.call_args
        # Check if custom_paths was passed as positional or keyword argument
        if len(call_args[0]) > 0:
            custom_paths = call_args[0][0]
        else:
            custom_paths = call_args.kwargs.get("custom_paths")
        assert custom_paths is not None

    @patch("ccproxy.cli.commands.auth.get_credentials_manager")
    def test_renew_command_custom_file(
        self,
        mock_get_manager: MagicMock,
        runner: CliRunner,
        mock_credentials_manager: AsyncMock,
        mock_validation_result_valid: ValidationResult,
        mock_credentials: ClaudeCredentials,
    ) -> None:
        """Test renew command with custom credential file."""
        mock_get_manager.return_value = mock_credentials_manager
        mock_credentials_manager.validate.return_value = mock_validation_result_valid
        mock_credentials_manager.refresh_token.return_value = mock_credentials
        custom_file = "/custom/credentials.json"

        result = runner.invoke(app, ["renew", "--credential-file", custom_file])

        assert result.exit_code == 0
        # Check that get_credentials_manager was called with custom file path
        mock_get_manager.assert_called_once()
        call_args = mock_get_manager.call_args
        # Check if custom_paths was passed as positional or keyword argument
        if len(call_args[0]) > 0:
            custom_paths = call_args[0][0]
        else:
            custom_paths = call_args.kwargs.get("custom_paths")
        assert custom_paths == [Path(custom_file)]


class TestAuthCLIIntegration(TestAuthCLICommands):
    """Test CLI authentication integration scenarios."""

    def test_app_structure(self) -> None:
        """Test that the auth app is properly structured."""
        assert app.info.name == "auth"
        assert app.info.help == "Authentication and credential management"

    @patch("ccproxy.cli.commands.auth.get_credentials_manager")
    def test_all_commands_available(
        self,
        mock_get_manager: MagicMock,
        runner: CliRunner,
        mock_credentials_manager: AsyncMock,
    ) -> None:
        """Test that all auth commands are available."""
        mock_get_manager.return_value = mock_credentials_manager

        # Test help shows all commands
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "validate" in result.stdout
        assert "info" in result.stdout
        assert "login" in result.stdout
        assert "renew" in result.stdout

    def test_command_structure_types(self) -> None:
        """Test that command functions have proper type annotations."""
        # Verify that our command functions have proper type hints
        import inspect

        # Check validate_credentials function signature
        sig = inspect.signature(validate_credentials)
        assert sig.return_annotation is None  # typer commands return None

        # Check credential_info function signature
        sig = inspect.signature(credential_info)
        assert sig.return_annotation is None

        # Check login_command function signature
        sig = inspect.signature(login_command)
        assert sig.return_annotation is None

        # Check renew function signature
        sig = inspect.signature(renew)
        assert sig.return_annotation is None
