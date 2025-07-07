"""Tests for the Claude command."""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from claude_code_proxy.cli.commands.claude import claude
from claude_code_proxy.config.settings import Settings


@pytest.fixture
def runner():
    """Create a Typer test runner."""
    return CliRunner()


@pytest.fixture
def mock_settings():
    """Create mock settings."""
    settings = MagicMock(spec=Settings)
    settings.claude_cli_path = "/usr/local/bin/claude"

    # Create docker_settings as a separate mock
    docker_settings = MagicMock()
    docker_settings.docker_image = "claude-proxy:latest"
    docker_settings.docker_volumes = []
    docker_settings.docker_environment = {}
    docker_settings.docker_additional_args = []
    docker_settings.docker_home_directory = None
    docker_settings.docker_workspace_directory = None
    docker_settings.user_mapping_enabled = False
    docker_settings.user_uid = None
    docker_settings.user_gid = None

    settings.docker_settings = docker_settings
    return settings


class TestClaudeCommand:
    """Test cases for the Claude command."""

    @patch("claude_code_proxy.cli.commands.claude.config_manager")
    @patch("claude_code_proxy.cli.commands.claude.os.execvp")
    def test_claude_local_execution(
        self, mock_execvp, mock_config_manager, mock_settings
    ):
        """Test executing claude command locally."""
        mock_config_manager.load_settings.return_value = mock_settings

        # Create a test app with the claude command
        from typer import Typer

        app = Typer()
        app.command()(claude)

        runner = CliRunner()
        result = runner.invoke(app, ["--", "--version"])

        # execvp doesn't return, so we check it was called
        mock_execvp.assert_called_once_with(
            "/usr/local/bin/claude", ["/usr/local/bin/claude", "--version"]
        )

    @patch("claude_code_proxy.cli.commands.claude.config_manager")
    def test_claude_no_cli_path(self, mock_config_manager):
        """Test claude command when CLI path is not configured."""
        mock_settings = MagicMock()
        mock_settings.claude_cli_path = None
        mock_config_manager.load_settings.return_value = mock_settings

        # Create a test app with the claude command
        from typer import Typer

        app = Typer()
        app.command()(claude)

        runner = CliRunner()
        result = runner.invoke(app, ["--", "--version"])

        assert result.exit_code == 1
        assert "Claude CLI not found" in result.output

    @patch("claude_code_proxy.cli.commands.claude.config_manager")
    @patch("claude_code_proxy.cli.commands.claude.create_docker_adapter")
    def test_claude_docker_execution(
        self, mock_docker_adapter, mock_config_manager, mock_settings
    ):
        """Test executing claude command with Docker."""
        mock_config_manager.load_settings.return_value = mock_settings

        # Mock Docker adapter
        mock_adapter = MagicMock()
        mock_docker_adapter.return_value = mock_adapter

        # Create a test app with the claude command
        from typer import Typer

        app = Typer()
        app.command()(claude)

        runner = CliRunner()
        result = runner.invoke(app, ["--docker", "--", "--version"])

        assert result.exit_code == 0
        mock_docker_adapter.assert_called_once()
        mock_adapter.exec_container.assert_called_once()

        # Verify command was passed correctly
        call_args = mock_adapter.exec_container.call_args
        command = call_args.kwargs["command"]
        assert command == ["claude", "--version"]

    @patch("claude_code_proxy.cli.commands.claude.config_manager")
    @patch("claude_code_proxy.cli.commands.claude.create_docker_adapter")
    def test_claude_docker_with_custom_image(
        self, mock_docker_adapter, mock_config_manager, mock_settings
    ):
        """Test claude command with custom Docker image."""
        mock_config_manager.load_settings.return_value = mock_settings
        mock_adapter = MagicMock()
        mock_docker_adapter.return_value = mock_adapter

        # Create a test app with the claude command
        from typer import Typer

        app = Typer()
        app.command()(claude)

        runner = CliRunner()
        result = runner.invoke(
            app, ["--docker", "--docker-image", "custom:v2", "--", "doctor"]
        )

        assert result.exit_code == 0
        call_args = mock_adapter.exec_container.call_args
        assert call_args.kwargs["image"] == "custom:v2"
        assert call_args.kwargs["command"] == ["claude", "doctor"]

    @patch("claude_code_proxy.cli.commands.claude.config_manager")
    @patch("claude_code_proxy.cli.commands.claude.create_docker_adapter")
    def test_claude_docker_with_volumes_and_env(
        self, mock_docker_adapter, mock_config_manager, mock_settings, tmp_path
    ):
        """Test claude command with Docker volumes and environment variables."""
        mock_config_manager.load_settings.return_value = mock_settings
        mock_adapter = MagicMock()
        mock_docker_adapter.return_value = mock_adapter

        # Create temporary directory for volume
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        # Create a test app with the claude command
        from typer import Typer

        app = Typer()
        app.command()(claude)

        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "--docker",
                "--docker-volume",
                f"{data_dir}:/data",
                "--docker-env",
                "CLAUDE_API_KEY=sk-123",
                "--",
                "chat",
            ],
        )

        assert result.exit_code == 0
        call_args = mock_adapter.exec_container.call_args

        # Check volumes
        volumes = call_args.kwargs["volumes"]
        assert any(v[1] == "/data" for v in volumes)

        # Check environment
        environment = call_args.kwargs["environment"]
        assert "CLAUDE_API_KEY" in environment
        assert environment["CLAUDE_API_KEY"] == "sk-123"

    @patch("claude_code_proxy.cli.commands.claude.config_manager")
    @patch("claude_code_proxy.cli.commands.claude.create_docker_adapter")
    def test_claude_docker_with_user_mapping(
        self, mock_docker_adapter, mock_config_manager, mock_settings
    ):
        """Test claude command with Docker user mapping."""
        mock_config_manager.load_settings.return_value = mock_settings
        mock_adapter = MagicMock()
        mock_docker_adapter.return_value = mock_adapter

        # Create a test app with the claude command
        from typer import Typer

        app = Typer()
        app.command()(claude)

        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "--docker",
                "--user-mapping",
                "--user-uid",
                "1000",
                "--user-gid",
                "1000",
                "--",
                "config",
            ],
        )

        assert result.exit_code == 0
        call_args = mock_adapter.exec_container.call_args
        user_context = call_args.kwargs["user_context"]

        assert user_context is not None
        assert user_context.uid == 1000
        assert user_context.gid == 1000

    @patch("claude_code_proxy.cli.commands.claude.config_manager")
    @patch("claude_code_proxy.cli.commands.claude.os.execvp")
    def test_claude_with_multiple_args(
        self, mock_execvp, mock_config_manager, mock_settings
    ):
        """Test claude command with multiple arguments."""
        mock_config_manager.load_settings.return_value = mock_settings

        # Create a test app with the claude command
        from typer import Typer

        app = Typer()
        app.command()(claude)

        runner = CliRunner()
        result = runner.invoke(
            app, ["--", "chat", "--model", "claude-3", "--max-tokens", "1000"]
        )

        mock_execvp.assert_called_once_with(
            "/usr/local/bin/claude",
            [
                "/usr/local/bin/claude",
                "chat",
                "--model",
                "claude-3",
                "--max-tokens",
                "1000",
            ],
        )

    @patch("claude_code_proxy.cli.commands.claude.config_manager")
    def test_claude_no_args(self, mock_config_manager, mock_settings):
        """Test claude command with no arguments."""
        mock_config_manager.load_settings.return_value = mock_settings

        # Create a test app with the claude command
        from typer import Typer

        app = Typer()
        app.command()(claude)

        with patch("claude_code_proxy.cli.commands.claude.os.execvp") as mock_execvp:
            runner = CliRunner()
            result = runner.invoke(app, [])

            # Should execute with no additional args
            mock_execvp.assert_called_once_with(
                "/usr/local/bin/claude", ["/usr/local/bin/claude"]
            )

    @patch("claude_code_proxy.cli.commands.claude.config_manager")
    @patch("claude_code_proxy.cli.commands.claude.Path.is_absolute")
    @patch("claude_code_proxy.cli.commands.claude.Path.resolve")
    @patch("claude_code_proxy.cli.commands.claude.os.execvp")
    def test_claude_relative_path_resolution(
        self, mock_execvp, mock_resolve, mock_is_absolute, mock_config_manager
    ):
        """Test claude command resolves relative paths to absolute."""
        mock_settings = MagicMock()
        mock_settings.claude_cli_path = "claude"
        mock_config_manager.load_settings.return_value = mock_settings

        mock_is_absolute.return_value = False
        mock_resolve.return_value = Path("/usr/local/bin/claude")

        # Create a test app with the claude command
        from typer import Typer

        app = Typer()
        app.command()(claude)

        runner = CliRunner()
        result = runner.invoke(app, ["--", "--version"])

        mock_execvp.assert_called_once_with(
            "/usr/local/bin/claude", ["/usr/local/bin/claude", "--version"]
        )

    @patch("claude_code_proxy.cli.commands.claude.config_manager")
    def test_claude_configuration_error(self, mock_config_manager):
        """Test claude command with configuration error."""
        from claude_code_proxy.config.settings import ConfigurationError

        mock_config_manager.load_settings.side_effect = ConfigurationError(
            "Invalid config"
        )

        # Create a test app with the claude command
        from typer import Typer

        app = Typer()
        app.command()(claude)

        runner = CliRunner()
        result = runner.invoke(app, ["--", "--version"])

        assert result.exit_code == 1
        assert "Configuration error" in result.output

    @patch("claude_code_proxy.cli.commands.claude.config_manager")
    @patch("claude_code_proxy.cli.commands.claude.os.execvp")
    def test_claude_os_error(self, mock_execvp, mock_config_manager, mock_settings):
        """Test claude command with OS error during execution."""
        mock_config_manager.load_settings.return_value = mock_settings
        mock_execvp.side_effect = OSError("Command not found")

        # Create a test app with the claude command
        from typer import Typer

        app = Typer()
        app.command()(claude)

        runner = CliRunner()
        result = runner.invoke(app, ["--", "--version"])

        assert result.exit_code == 1
        assert "Failed to execute command" in result.output
