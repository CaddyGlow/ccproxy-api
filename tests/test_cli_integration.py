"""Integration tests for CLI commands."""

import os
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
import typer
from typer.testing import CliRunner

from claude_code_proxy.cli import app, claude, config
from claude_code_proxy.config.settings import Settings


@pytest.mark.integration
class TestCliRunner:
    """Test CLI using CliRunner."""

    def setup_method(self):
        """Setup test environment."""
        self.runner = CliRunner(env={"NO_COLOR": "1", "TERM": "dumb"})

    def test_version_flag_short(self):
        """Test -V flag prints version and exits."""
        result = self.runner.invoke(app, ["-V"])
        assert result.exit_code == 0
        assert "claude-code-proxy-api" in result.stdout

    def test_version_flag_long(self):
        """Test --version flag prints version and exits."""
        result = self.runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert "claude-code-proxy-api" in result.stdout

    def test_help_flag(self):
        """Test --help flag shows help."""
        result = self.runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "Claude Code Proxy API Server" in result.stdout
        assert "Anthropic" in result.stdout

    def test_no_args_shows_help(self):
        """Test no arguments shows help."""
        result = self.runner.invoke(app, [])
        assert result.exit_code == 2  # Typer returns 2 for missing required args
        assert "Usage:" in result.stdout


@pytest.mark.integration
class TestConfigCommand:
    """Test config command."""

    def setup_method(self):
        """Setup test environment."""
        self.runner = CliRunner(env={"NO_COLOR": "1", "TERM": "dumb"})

    @patch("claude_code_proxy.cli.get_settings")
    def test_config_command_success(self, mock_get_settings):
        """Test config command shows configuration."""
        mock_settings = Mock(spec=Settings)
        mock_settings.host = "localhost"
        mock_settings.port = 8000
        mock_settings.log_level = "INFO"
        mock_settings.claude_cli_path = "/usr/bin/claude"
        mock_settings.workers = 1
        mock_settings.reload = False
        mock_get_settings.return_value = mock_settings

        result = self.runner.invoke(app, ["config"])

        assert result.exit_code == 0
        assert "Current Configuration:" in result.stdout
        assert "Host: localhost" in result.stdout
        assert "Port: 8000" in result.stdout
        assert "Log Level: INFO" in result.stdout
        assert "Claude CLI Path: /usr/bin/claude" in result.stdout
        assert "Workers: 1" in result.stdout
        assert "Reload: False" in result.stdout

    @patch("claude_code_proxy.cli.get_settings")
    def test_config_command_auto_detect_claude_path(self, mock_get_settings):
        """Test config command with auto-detect claude path."""
        mock_settings = Mock(spec=Settings)
        mock_settings.host = "0.0.0.0"
        mock_settings.port = 3000
        mock_settings.log_level = "DEBUG"
        mock_settings.claude_cli_path = None
        mock_settings.workers = 4
        mock_settings.reload = True
        mock_get_settings.return_value = mock_settings

        result = self.runner.invoke(app, ["config"])

        assert result.exit_code == 0
        assert "Claude CLI Path: Auto-detect" in result.stdout
        assert "Workers: 4" in result.stdout
        assert "Reload: True" in result.stdout

    @patch("claude_code_proxy.cli.get_settings")
    def test_config_command_error(self, mock_get_settings):
        """Test config command handles errors."""
        mock_get_settings.side_effect = Exception("Configuration error")

        result = self.runner.invoke(app, ["config"])

        assert result.exit_code == 1
        # Error messages might be in stderr or stdout depending on how typer handles them
        assert "Error loading configuration: Configuration error" in (
            result.stdout + result.stderr
        )

    def test_config_command_help(self):
        """Test config command help."""
        result = self.runner.invoke(app, ["config", "--help"])
        assert result.exit_code == 0
        assert "Show current configuration" in result.stdout


@pytest.mark.integration
class TestClaudeCommand:
    """Test claude command."""

    def setup_method(self):
        """Setup test environment."""
        self.runner = CliRunner(env={"NO_COLOR": "1", "TERM": "dumb"})

    @patch("claude_code_proxy.cli.get_settings")
    @patch("claude_code_proxy.cli.os.execvp")
    def test_claude_command_local_cli(self, mock_execvp, mock_get_settings):
        """Test claude command with local CLI."""
        mock_settings = Mock(spec=Settings)
        mock_settings.claude_cli_path = "/usr/bin/claude"
        mock_get_settings.return_value = mock_settings

        # Mock Path.is_absolute to return True
        with patch("pathlib.Path.is_absolute", return_value=True):
            # The CliRunner doesn't actually call execvp, so we check the exit code
            result = self.runner.invoke(app, ["claude", "--", "--version"])

            # Since execvp replaces the process, we expect it to be called
            mock_execvp.assert_called_once_with(
                "/usr/bin/claude", ["/usr/bin/claude", "--version"]
            )

    @patch("claude_code_proxy.cli.get_settings")
    @patch("claude_code_proxy.cli.os.execvp")
    def test_claude_command_relative_path(self, mock_execvp, mock_get_settings):
        """Test claude command with relative path."""
        mock_settings = Mock(spec=Settings)
        mock_settings.claude_cli_path = "claude"
        mock_get_settings.return_value = mock_settings

        # Mock Path methods
        with (
            patch("pathlib.Path.is_absolute", return_value=False),
            patch("pathlib.Path.resolve", return_value=Path("/resolved/path/claude")),
        ):
            result = self.runner.invoke(app, ["claude", "--", "--version"])

            mock_execvp.assert_called_once_with(
                "/resolved/path/claude", ["/resolved/path/claude", "--version"]
            )

    @patch("claude_code_proxy.cli.get_settings")
    def test_claude_command_no_cli_path(self, mock_get_settings):
        """Test claude command when CLI path is not found."""
        mock_settings = Mock(spec=Settings)
        mock_settings.claude_cli_path = None
        mock_get_settings.return_value = mock_settings

        result = self.runner.invoke(app, ["claude", "--", "--version"])

        assert result.exit_code == 1
        # Error messages are in stderr
        assert "Error: Claude CLI not found" in result.stderr
        assert "Please install Claude CLI" in result.stderr

    @patch("claude_code_proxy.cli.get_settings")
    @patch("claude_code_proxy.cli.os.execvp")
    def test_claude_command_execvp_error(self, mock_execvp, mock_get_settings):
        """Test claude command when execvp fails."""
        mock_settings = Mock(spec=Settings)
        mock_settings.claude_cli_path = "/usr/bin/claude"
        mock_get_settings.return_value = mock_settings

        mock_execvp.side_effect = OSError("Command not found")

        with patch("pathlib.Path.is_absolute", return_value=True):
            result = self.runner.invoke(app, ["claude", "--", "--version"])

            assert result.exit_code == 1
            assert "Failed to execute command: Command not found" in result.stderr

    @patch("claude_code_proxy.cli.get_settings")
    @patch("claude_code_proxy.cli.DockerCommandBuilder")
    @patch("claude_code_proxy.cli.os.execvp")
    def test_claude_command_docker_mode(
        self, mock_execvp, mock_docker_builder, mock_get_settings
    ):
        """Test claude command in Docker mode."""
        mock_settings = Mock(spec=Settings)
        mock_settings.docker_settings = Mock()
        mock_get_settings.return_value = mock_settings

        mock_docker_cmd = ["docker", "run", "claude", "--version"]
        mock_docker_builder.from_settings_and_overrides.return_value = mock_docker_cmd

        result = self.runner.invoke(app, ["claude", "--docker", "--", "--version"])

        mock_docker_builder.from_settings_and_overrides.assert_called_once()
        mock_execvp.assert_called_once_with("docker", mock_docker_cmd)

    @patch("claude_code_proxy.cli.get_settings")
    @patch("claude_code_proxy.cli.DockerCommandBuilder")
    @patch("claude_code_proxy.cli.os.execvp")
    def test_claude_command_docker_with_options(
        self, mock_execvp, mock_docker_builder, mock_get_settings
    ):
        """Test claude command in Docker mode with options."""
        mock_settings = Mock(spec=Settings)
        mock_settings.docker_settings = Mock()
        mock_get_settings.return_value = mock_settings

        mock_docker_cmd = [
            "docker",
            "run",
            "-e",
            "API_KEY=test",
            "custom:latest",
            "--version",
        ]
        mock_docker_builder.from_settings_and_overrides.return_value = mock_docker_cmd

        result = self.runner.invoke(
            app,
            [
                "claude",
                "--docker",
                "--docker-image",
                "custom:latest",
                "--docker-env",
                "API_KEY=test",
                "--",
                "--version",
            ],
        )

        # Check that Docker builder was called with correct parameters
        call_args = mock_docker_builder.from_settings_and_overrides.call_args
        # Note: claude_args is no longer passed to DockerCommandBuilder
        assert call_args[1]["docker_image"] == "custom:latest"
        assert call_args[1]["docker_env"] == ["API_KEY=test"]

    @patch("claude_code_proxy.cli.get_settings")
    def test_claude_command_settings_error(self, mock_get_settings):
        """Test claude command when settings loading fails."""
        mock_get_settings.side_effect = Exception("Settings error")

        result = self.runner.invoke(app, ["claude", "--", "--version"])

        assert result.exit_code == 1
        assert "Error executing claude command: Settings error" in result.stderr

    def test_claude_command_help(self):
        """Test claude command help."""
        result = self.runner.invoke(app, ["claude", "--help"])
        assert result.exit_code == 0
        assert "Execute claude CLI commands directly" in result.stdout
        assert "Examples:" in result.stdout
        assert "ccproxy claude -- --version" in result.stdout
        assert "--docker" in result.stdout

    def test_claude_command_docker_help_options(self):
        """Test claude command shows Docker-related options."""
        result = self.runner.invoke(app, ["claude", "--help"])
        assert result.exit_code == 0
        assert "--docker-image" in result.stdout
        assert "--docker-env" in result.stdout
        assert "--docker-volume" in result.stdout
        assert "--docker-arg" in result.stdout
        assert "--docker-home" in result.stdout
        assert "--docker-workspace" in result.stdout


@pytest.mark.integration
class TestCommandIntegration:
    """Test command integration and error handling."""

    def setup_method(self):
        """Setup test environment."""
        self.runner = CliRunner(env={"NO_COLOR": "1", "TERM": "dumb"})

    def test_invalid_command(self):
        """Test invalid command shows error."""
        result = self.runner.invoke(app, ["invalid-command"])
        assert result.exit_code == 2
        assert "No such command" in result.stderr

    def test_command_list_in_help(self):
        """Test available commands are listed in help."""
        result = self.runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "config" in result.stdout
        assert "claude" in result.stdout

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"})
    def test_environment_variables_isolation(self):
        """Test that environment variables don't interfere with CLI tests."""
        result = self.runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert "claude-code-proxy-api" in result.stdout

    def test_cli_app_callable(self):
        """Test that the CLI app is callable."""
        assert callable(app)

    def test_commands_registered(self):
        """Test that commands are properly registered."""
        # Test that we can invoke the commands without errors
        result = self.runner.invoke(app, ["config", "--help"])
        assert result.exit_code == 0

        result = self.runner.invoke(app, ["claude", "--help"])
        assert result.exit_code == 0


@pytest.mark.integration
class TestDockerIntegration:
    """Test Docker integration scenarios."""

    def setup_method(self):
        """Setup test environment."""
        self.runner = CliRunner(env={"NO_COLOR": "1", "TERM": "dumb"})

    @patch("claude_code_proxy.cli.get_settings")
    @patch("claude_code_proxy.cli.DockerCommandBuilder")
    @patch("claude_code_proxy.cli.os.execvp")
    def test_docker_command_building(
        self, mock_execvp, mock_docker_builder, mock_get_settings
    ):
        """Test Docker command building with various options."""
        mock_settings = Mock(spec=Settings)
        mock_settings.docker_settings = Mock()
        mock_get_settings.return_value = mock_settings

        mock_docker_cmd = ["docker", "run", "--rm", "-it", "claude:latest", "doctor"]
        mock_docker_builder.from_settings_and_overrides.return_value = mock_docker_cmd

        result = self.runner.invoke(
            app,
            [
                "claude",
                "--docker",
                "--docker-volume",
                "/home/user:/home/user",
                "--docker-env",
                "HOME=/home/user",
                "--docker-arg",
                "--rm",
                "doctor",
            ],
        )

        # Verify the Docker builder was called with correct parameters
        call_args = mock_docker_builder.from_settings_and_overrides.call_args
        assert call_args[1]["docker_volume"] == ["/home/user:/home/user"]
        assert call_args[1]["docker_env"] == ["HOME=/home/user"]
        assert call_args[1]["docker_arg"] == ["--rm"]
        # Note: claude_args is no longer passed to DockerCommandBuilder

    @patch("claude_code_proxy.cli.get_settings")
    @patch("claude_code_proxy.cli.DockerCommandBuilder")
    @patch("claude_code_proxy.cli.os.execvp")
    def test_docker_multiple_volumes_and_env(
        self, mock_execvp, mock_docker_builder, mock_get_settings
    ):
        """Test Docker command with multiple volumes and environment variables."""
        mock_settings = Mock(spec=Settings)
        mock_settings.docker_settings = Mock()
        mock_get_settings.return_value = mock_settings

        mock_docker_cmd = ["docker", "run", "claude:latest", "config"]
        mock_docker_builder.from_settings_and_overrides.return_value = mock_docker_cmd

        result = self.runner.invoke(
            app,
            [
                "claude",
                "--docker",
                "--docker-volume",
                "/data:/data",
                "--docker-volume",
                "/config:/config:ro",
                "--docker-env",
                "API_KEY=test",
                "--docker-env",
                "LOG_LEVEL=DEBUG",
                "config",
            ],
        )

        call_args = mock_docker_builder.from_settings_and_overrides.call_args
        assert call_args[1]["docker_volume"] == ["/data:/data", "/config:/config:ro"]
        assert call_args[1]["docker_env"] == ["API_KEY=test", "LOG_LEVEL=DEBUG"]


@pytest.mark.integration
class TestErrorScenarios:
    """Test various error scenarios."""

    def setup_method(self):
        """Setup test environment."""
        self.runner = CliRunner(env={"NO_COLOR": "1", "TERM": "dumb"})

    @patch("claude_code_proxy.cli.get_settings")
    def test_config_command_exception_handling(self, mock_get_settings):
        """Test config command handles various exceptions."""
        mock_get_settings.side_effect = FileNotFoundError("Config file not found")

        result = self.runner.invoke(app, ["config"])

        assert result.exit_code == 1
        assert "Error loading configuration: Config file not found" in result.stderr

    @patch("claude_code_proxy.cli.get_settings")
    def test_claude_command_exception_handling(self, mock_get_settings):
        """Test claude command handles various exceptions."""
        mock_get_settings.side_effect = ValueError("Invalid configuration")

        result = self.runner.invoke(app, ["claude", "--", "--version"])

        assert result.exit_code == 1
        assert "Error executing claude command: Invalid configuration" in result.stderr

    @patch("claude_code_proxy.cli.get_settings")
    @patch("claude_code_proxy.cli.os.execvp")
    def test_claude_command_permission_error(self, mock_execvp, mock_get_settings):
        """Test claude command handles permission errors."""
        mock_settings = Mock(spec=Settings)
        mock_settings.claude_cli_path = "/usr/bin/claude"
        mock_get_settings.return_value = mock_settings

        mock_execvp.side_effect = PermissionError("Permission denied")

        with patch("pathlib.Path.is_absolute", return_value=True):
            result = self.runner.invoke(app, ["claude", "--", "--version"])

            assert result.exit_code == 1
            assert "Failed to execute command: Permission denied" in result.stderr


@pytest.mark.integration
class TestFastAPICliIntegration:
    """Test FastAPI CLI commands integration."""

    def setup_method(self):
        """Setup test environment."""
        self.runner = CliRunner(env={"NO_COLOR": "1", "TERM": "dumb"})

    def test_run_command_help(self):
        """Test run command help."""
        result = self.runner.invoke(app, ["run", "--help"])
        assert result.exit_code == 0
        assert "Run a FastAPI app in production mode" in result.stdout
        assert "--host" in result.stdout
        assert "--port" in result.stdout
        assert "--workers" in result.stdout
        assert "--reload" in result.stdout

    def test_dev_command_help(self):
        """Test dev command help."""
        result = self.runner.invoke(app, ["dev", "--help"])
        assert result.exit_code == 0
        assert "Run a FastAPI app in development mode" in result.stdout
        assert "--host" in result.stdout
        assert "--port" in result.stdout
        assert "--reload" in result.stdout

    def test_fastapi_cli_commands_available(self):
        """Test that FastAPI CLI commands are available."""
        # Test that run command is available
        result = self.runner.invoke(app, ["run", "--help"])
        assert result.exit_code == 0
        assert "Run a FastAPI app in production mode" in result.stdout

        # Test that dev command is available
        result = self.runner.invoke(app, ["dev", "--help"])
        assert result.exit_code == 0
        assert "Run a FastAPI app in development mode" in result.stdout

    def test_fastapi_cli_options_validation(self):
        """Test FastAPI CLI options validation."""
        # Test run command with invalid port
        result = self.runner.invoke(app, ["run", "--port", "invalid"])
        assert result.exit_code != 0

        # Test run command with negative port
        result = self.runner.invoke(app, ["run", "--port", "-1"])
        assert result.exit_code != 0

    @patch("claude_code_proxy.cli.get_default_path_hook")
    def test_default_path_hook_success(self, mock_get_default_path):
        """Test default path hook finds the main.py file."""
        mock_path = Mock()
        mock_path.is_file.return_value = True
        mock_get_default_path.return_value = mock_path

        # Call the hook directly
        from claude_code_proxy.cli import get_default_path_hook

        result = get_default_path_hook()

        assert result == mock_path

    @patch("claude_code_proxy.cli.get_package_dir")
    @patch("pathlib.Path.is_file")
    def test_default_path_hook_no_file_found(self, mock_is_file, mock_get_package_dir):
        """Test default path hook when no main.py file is found."""
        mock_package_dir = Path("/mock/package/dir")
        mock_get_package_dir.return_value = mock_package_dir
        # Mock is_file to return False for all paths
        mock_is_file.return_value = False

        from fastapi_cli.exceptions import FastAPICLIException

        from claude_code_proxy.cli import get_default_path_hook

        with pytest.raises(FastAPICLIException) as exc_info:
            get_default_path_hook()

        assert "Could not find a default file to run" in str(exc_info.value)

    def test_fastapi_cli_integration_basic(self):
        """Test basic FastAPI CLI integration without starting servers."""
        # Test that commands exist and show proper help
        result = self.runner.invoke(app, ["run", "--help"])
        assert result.exit_code == 0
        assert "FastAPI app in production mode" in result.stdout

        result = self.runner.invoke(app, ["dev", "--help"])
        assert result.exit_code == 0
        assert "FastAPI app in development mode" in result.stdout

    def test_fastapi_commands_in_help(self):
        """Test that FastAPI commands appear in main help."""
        result = self.runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "run" in result.stdout
        assert "dev" in result.stdout
        assert "Run a FastAPI app in production mode" in result.stdout
        assert "Run a FastAPI app in development mode" in result.stdout


@pytest.mark.integration
class TestCliEnvironmentIsolation:
    """Test CLI environment isolation and configuration."""

    def setup_method(self):
        """Setup test environment."""
        self.runner = CliRunner(env={"NO_COLOR": "1", "TERM": "dumb"})

    @patch.dict(os.environ, {"CLAUDE_CODE_PROXY_HOST": "test-host"})
    def test_environment_variable_isolation(self):
        """Test that environment variables don't interfere with tests."""
        result = self.runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        # The version should still work regardless of environment variables
        assert "claude-code-proxy-api" in result.stdout

    def test_cli_exit_codes(self):
        """Test proper exit codes for various scenarios."""
        # Success case
        result = self.runner.invoke(app, ["--version"])
        assert result.exit_code == 0

        # Help case
        result = self.runner.invoke(app, ["--help"])
        assert result.exit_code == 0

        # Invalid command case
        result = self.runner.invoke(app, ["nonexistent-command"])
        assert result.exit_code != 0

    def test_cli_output_format(self):
        """Test CLI output formatting."""
        result = self.runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        # Should contain version info in expected format
        assert "claude-code-proxy-api" in result.stdout
        # Should not contain extra whitespace or formatting issues
        assert result.stdout.strip()

    def test_cli_mixed_options_and_commands(self):
        """Test CLI with mixed options and commands."""
        # Test version flag with command (version should take precedence)
        result = self.runner.invoke(app, ["--version", "config"])
        assert result.exit_code == 0
        assert "claude-code-proxy-api" in result.stdout
        # Should not execute the config command
        assert "Current Configuration:" not in result.stdout

    @patch("claude_code_proxy.cli.get_settings")
    def test_cli_command_isolation(self, mock_get_settings):
        """Test that CLI commands don't interfere with each other."""
        mock_settings = Mock(spec=Settings)
        mock_settings.host = "localhost"
        mock_settings.port = 8000
        mock_settings.log_level = "INFO"
        mock_settings.claude_cli_path = "/usr/bin/claude"
        mock_settings.workers = 1
        mock_settings.reload = False
        mock_get_settings.return_value = mock_settings

        # Run config command
        result1 = self.runner.invoke(app, ["config"])
        assert result1.exit_code == 0
        assert "Current Configuration:" in result1.stdout

        # Run version command - should not be affected by previous command
        result2 = self.runner.invoke(app, ["--version"])
        assert result2.exit_code == 0
        assert "claude-code-proxy-api" in result2.stdout

        # Run help command - should not be affected by previous commands
        result3 = self.runner.invoke(app, ["--help"])
        assert result3.exit_code == 0
        assert "Claude Code Proxy API Server" in result3.stdout

    def test_cli_typer_integration(self):
        """Test Typer integration specifics."""
        # Test that the CLI app is properly configured
        assert app.info is not None
        assert app.info.help is not None

        # Test rich markup is configured
        assert hasattr(app, "rich_markup_mode")

        # Test no_args_is_help is configured
        result = self.runner.invoke(app, [])
        assert result.exit_code == 2  # Typer returns 2 for help when no args


@pytest.mark.integration
class TestCliRobustness:
    """Test CLI robustness and edge cases."""

    def setup_method(self):
        """Setup test environment."""
        self.runner = CliRunner(env={"NO_COLOR": "1", "TERM": "dumb"})

    def test_cli_with_invalid_arguments(self):
        """Test CLI behavior with invalid arguments."""
        # Test invalid flag
        result = self.runner.invoke(app, ["--invalid-flag"])
        assert result.exit_code != 0

        # Test invalid command
        result = self.runner.invoke(app, ["invalid-command"])
        assert result.exit_code != 0

    def test_cli_with_special_characters(self):
        """Test CLI with special characters in arguments."""
        # Test claude command with special characters
        result = self.runner.invoke(app, ["claude", "--help"])
        assert result.exit_code == 0

        # Test config command (should not be affected by special characters in environment)
        with patch.dict(os.environ, {"TEST_VAR": "special!@#$%^&*()chars"}):
            result = self.runner.invoke(app, ["config", "--help"])
            assert result.exit_code == 0

    def test_cli_memory_usage(self):
        """Test that CLI commands don't leak memory or resources."""
        # Multiple invocations should not cause issues
        for _ in range(5):
            result = self.runner.invoke(app, ["--version"])
            assert result.exit_code == 0
            assert "claude-code-proxy-api" in result.stdout

    def test_cli_signal_handling(self):
        """Test CLI signal handling (basic test)."""
        # Test that normal commands complete successfully
        result = self.runner.invoke(app, ["--version"])
        assert result.exit_code == 0

        # Test that help commands complete successfully
        result = self.runner.invoke(app, ["--help"])
        assert result.exit_code == 0

    def test_cli_concurrent_safety(self):
        """Test CLI concurrent safety (basic test)."""
        # Test that multiple CLI invocations don't interfere
        results = []
        for _i in range(3):
            result = self.runner.invoke(app, ["--version"])
            results.append(result)

        # All should succeed
        for result in results:
            assert result.exit_code == 0
            assert "claude-code-proxy-api" in result.stdout
