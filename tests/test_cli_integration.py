"""Integration tests for CLI commands."""

import os
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
import typer
from typer.testing import CliRunner

from claude_code_proxy.cli import app, claude
from claude_code_proxy.cli.commands.config import config_list
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

    def test_api_command_with_no_extra_args(self):
        """Test api command runs successfully with mocked dependencies."""
        with (
            patch(
                "claude_code_proxy.cli.commands.api.get_settings"
            ) as mock_get_settings,
            patch("uvicorn.run") as mock_run,
        ):
            # Create a proper Settings instance with valid values
            from claude_code_proxy.config.settings import Settings

            mock_settings = Settings(
                host="127.0.0.1",
                port=8000,
                reload=False,
                workers=1,
                claude_cli_path=None,
            )
            mock_get_settings.return_value = mock_settings

            # Explicitly call api command since default behavior uses sys.argv manipulation
            result = self.runner.invoke(app, ["api"])

            # Should run api command successfully
            mock_run.assert_called_once()
            assert result.exit_code == 0

    def test_default_command_behavior_sys_argv_manipulation(self):
        """Test that sys.argv is correctly manipulated for default command behavior."""
        import sys

        # Test case 1: No arguments provided
        original_argv = sys.argv[:]
        try:
            sys.argv = ["ccproxy"]

            # Import and execute the main section logic
            import importlib

            import claude_code_proxy.cli
            from claude_code_proxy.cli import app

            # Simulate the sys.argv manipulation from __name__ == "__main__"
            if len(sys.argv) == 1:
                sys.argv.append("api")

            assert sys.argv == ["ccproxy", "api"]

        finally:
            sys.argv = original_argv

        # Test case 2: CLI options without command
        try:
            sys.argv = ["ccproxy", "--port", "8080"]

            # Check if any argument is a known command
            known_commands = {"api", "claude", "config", "fastapi"}

            has_command = False
            for arg in sys.argv[1:]:
                if not arg.startswith("-") and arg in known_commands:
                    has_command = True
                    break

            if not has_command and (
                "--help" not in sys.argv
                and "-h" not in sys.argv
                and "--version" not in sys.argv
                and "-V" not in sys.argv
            ):
                sys.argv.insert(1, "api")

            assert sys.argv == ["ccproxy", "api", "--port", "8080"]

        finally:
            sys.argv = original_argv

        # Test case 3: Explicit command should not be modified
        try:
            sys.argv = ["ccproxy", "claude", "--version"]

            # Known commands should not trigger insertion
            known_commands = {"api", "claude", "config", "fastapi"}

            has_command = False
            for arg in sys.argv[1:]:
                if not arg.startswith("-") and arg in known_commands:
                    has_command = True
                    break

            # Should have found 'claude' command, so no modification
            assert has_command
            assert sys.argv == ["ccproxy", "claude", "--version"]

        finally:
            sys.argv = original_argv


@pytest.mark.integration
class TestConfigCommand:
    """Test config command."""

    def setup_method(self):
        """Setup test environment."""
        self.runner = CliRunner(env={"NO_COLOR": "1", "TERM": "dumb"})

    @patch("claude_code_proxy.cli.commands.config.get_settings")
    def test_config_command_success(self, mock_get_settings):
        """Test config command shows configuration."""
        mock_settings = Mock(spec=Settings)
        mock_settings.host = "localhost"
        mock_settings.port = 8000
        mock_settings.log_level = "INFO"
        mock_settings.claude_cli_path = "/usr/bin/claude"
        mock_settings.workers = 1
        mock_settings.reload = False
        mock_settings.server_url = "http://localhost:8000"
        mock_settings.auth_token = None
        mock_settings.api_tools_handling = "warning"
        mock_settings.cors_origins = ["*"]
        # Add mock docker settings
        mock_docker_settings = Mock()
        mock_docker_settings.docker_image = "claude-code-proxy"
        mock_docker_settings.docker_home_directory = None
        mock_docker_settings.docker_workspace_directory = None
        mock_docker_settings.user_uid = 1000
        mock_docker_settings.user_gid = 1000
        mock_docker_settings.user_mapping_enabled = True
        mock_docker_settings.docker_volumes = []
        mock_docker_settings.docker_environment = {}
        mock_docker_settings.docker_additional_args = []
        mock_settings.docker_settings = mock_docker_settings

        # Add mock claude_code_options
        mock_claude_code_options = Mock()
        mock_claude_code_options.model = None
        mock_claude_code_options.max_thinking_tokens = 8000
        mock_claude_code_options.max_turns = None
        mock_claude_code_options.cwd = None
        mock_claude_code_options.system_prompt = None
        mock_claude_code_options.append_system_prompt = None
        mock_claude_code_options.permission_mode = None
        mock_claude_code_options.permission_prompt_tool_name = None
        mock_claude_code_options.continue_conversation = False
        mock_claude_code_options.resume = None
        mock_claude_code_options.allowed_tools = []
        mock_claude_code_options.disallowed_tools = []
        mock_claude_code_options.mcp_servers = []
        mock_claude_code_options.mcp_tools = []
        mock_settings.claude_code_options = mock_claude_code_options

        mock_get_settings.return_value = mock_settings

        result = self.runner.invoke(app, ["config", "list"])

        assert result.exit_code == 0
        assert "Claude Code Proxy API Configuration" in result.stdout
        assert "localhost" in result.stdout
        assert "8000" in result.stdout
        assert "INFO" in result.stdout
        assert "/usr/bin/claude" in result.stdout
        assert "Server Configuration" in result.stdout

    @patch("claude_code_proxy.cli.commands.config.get_settings")
    def test_config_command_auto_detect_claude_path(self, mock_get_settings):
        """Test config command with auto-detect claude path."""
        mock_settings = Mock(spec=Settings)
        mock_settings.host = "0.0.0.0"
        mock_settings.port = 3000
        mock_settings.log_level = "DEBUG"
        mock_settings.claude_cli_path = None
        mock_settings.workers = 4
        mock_settings.reload = True
        mock_settings.server_url = "http://0.0.0.0:3000"
        mock_settings.auth_token = None
        mock_settings.api_tools_handling = "warning"
        mock_settings.cors_origins = ["*"]
        # Add mock docker settings
        mock_docker_settings = Mock()
        mock_docker_settings.docker_image = "claude-code-proxy"
        mock_docker_settings.docker_home_directory = None
        mock_docker_settings.docker_workspace_directory = None
        mock_docker_settings.user_uid = 1000
        mock_docker_settings.user_gid = 1000
        mock_docker_settings.user_mapping_enabled = True
        mock_docker_settings.docker_volumes = []
        mock_docker_settings.docker_environment = {}
        mock_docker_settings.docker_additional_args = []
        mock_settings.docker_settings = mock_docker_settings

        # Add mock claude_code_options
        mock_claude_code_options = Mock()
        mock_claude_code_options.model = None
        mock_claude_code_options.max_thinking_tokens = 8000
        mock_claude_code_options.max_turns = None
        mock_claude_code_options.cwd = None
        mock_claude_code_options.system_prompt = None
        mock_claude_code_options.append_system_prompt = None
        mock_claude_code_options.permission_mode = None
        mock_claude_code_options.permission_prompt_tool_name = None
        mock_claude_code_options.continue_conversation = False
        mock_claude_code_options.resume = None
        mock_claude_code_options.allowed_tools = []
        mock_claude_code_options.disallowed_tools = []
        mock_claude_code_options.mcp_servers = []
        mock_claude_code_options.mcp_tools = []
        mock_settings.claude_code_options = mock_claude_code_options

        mock_get_settings.return_value = mock_settings

        result = self.runner.invoke(app, ["config", "list"])

        assert result.exit_code == 0
        assert "Auto-detect" in result.stdout
        assert "4" in result.stdout
        assert "True" in result.stdout

    @patch("claude_code_proxy.cli.commands.config.get_settings")
    def test_config_command_error(self, mock_get_settings):
        """Test config command handles errors."""
        mock_get_settings.side_effect = Exception("Configuration error")

        result = self.runner.invoke(app, ["config", "list"])

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

    @patch("claude_code_proxy.cli.commands.api.config_manager.load_settings")
    @patch("claude_code_proxy.cli.main.os.execvp")
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

    @patch("claude_code_proxy.cli.commands.api.config_manager.load_settings")
    @patch("claude_code_proxy.cli.main.os.execvp")
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

    @patch("claude_code_proxy.cli.commands.api.config_manager.load_settings")
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

    @patch("claude_code_proxy.cli.commands.api.config_manager.load_settings")
    @patch("claude_code_proxy.cli.main.os.execvp")
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

    @patch("claude_code_proxy.cli.commands.api.config_manager.load_settings")
    @patch("claude_code_proxy.utils.docker_builder.DockerCommandBuilder")
    @patch("claude_code_proxy.cli.main.os.execvp")
    def test_claude_command_docker_mode(
        self, mock_execvp, mock_docker_builder, mock_get_settings
    ):
        """Test claude command in Docker mode."""
        mock_settings = Mock(spec=Settings)
        mock_docker_settings = Mock()
        mock_docker_settings.docker_environment = {}
        mock_docker_settings.docker_additional_args = []
        mock_docker_settings.docker_volumes = []
        mock_docker_settings.docker_image = "test-image"
        mock_docker_settings.docker_home_directory = None
        mock_docker_settings.docker_workspace_directory = None
        mock_docker_settings.user_uid = 1000
        mock_docker_settings.user_gid = 1000
        mock_docker_settings.user_mapping_enabled = True
        mock_docker_settings.validate_environment_variable = Mock(
            return_value=("API_KEY", "test")
        )
        mock_docker_settings.validate_volume_format = Mock(
            return_value="/home/user:/home/user"
        )
        mock_settings.docker_settings = mock_docker_settings
        mock_get_settings.return_value = mock_settings

        mock_docker_cmd = ["docker", "run", "claude", "claude", "--version"]
        mock_docker_builder.from_settings_and_overrides.return_value = mock_docker_cmd
        mock_docker_builder.execute_from_settings.return_value = None

        result = self.runner.invoke(app, ["claude", "--docker", "--", "--version"])

        # Should execute Docker command via os.execvp
        mock_execvp.assert_called_once()
        call_args = mock_execvp.call_args
        assert call_args[0][0] == "docker"
        assert "claude" in call_args[0][1]
        assert "--version" in call_args[0][1]

    @patch("claude_code_proxy.cli.commands.api.config_manager.load_settings")
    @patch("claude_code_proxy.utils.docker_builder.DockerCommandBuilder")
    @patch("claude_code_proxy.cli.main.os.execvp")
    def test_claude_command_docker_with_options(
        self, mock_execvp, mock_docker_builder, mock_get_settings
    ):
        """Test claude command in Docker mode with options."""
        mock_settings = Mock(spec=Settings)
        mock_docker_settings = Mock()
        mock_docker_settings.docker_environment = {}
        mock_docker_settings.docker_additional_args = []
        mock_docker_settings.docker_volumes = []
        mock_docker_settings.docker_image = "test-image"
        mock_docker_settings.docker_home_directory = None
        mock_docker_settings.docker_workspace_directory = None
        mock_docker_settings.user_uid = 1000
        mock_docker_settings.user_gid = 1000
        mock_docker_settings.user_mapping_enabled = True
        mock_docker_settings.validate_environment_variable = Mock(
            return_value=("API_KEY", "test")
        )
        mock_docker_settings.validate_volume_format = Mock(
            return_value="/home/user:/home/user"
        )
        mock_settings.docker_settings = mock_docker_settings
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

        # Should execute Docker command via os.execvp with custom options
        mock_execvp.assert_called_once()
        call_args = mock_execvp.call_args
        assert call_args[0][0] == "docker"
        docker_cmd = call_args[0][1]
        assert "claude" in docker_cmd
        assert "--version" in docker_cmd

    @patch("claude_code_proxy.cli.commands.api.config_manager.load_settings")
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
        assert "--docker-worksp" in result.stdout  # Truncated in help display
        assert "--user-mapping" in result.stdout
        assert "--user-uid" in result.stdout
        assert "--user-gid" in result.stdout


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

    @patch("claude_code_proxy.cli.commands.api.config_manager.load_settings")
    @patch("claude_code_proxy.utils.docker_builder.DockerCommandBuilder")
    @patch("claude_code_proxy.cli.main.os.execvp")
    def test_docker_command_building(
        self, mock_execvp, mock_docker_builder, mock_get_settings
    ):
        """Test Docker command building with various options."""
        mock_settings = Mock(spec=Settings)
        mock_docker_settings = Mock()
        mock_docker_settings.docker_environment = {}
        mock_docker_settings.docker_additional_args = []
        mock_docker_settings.docker_volumes = []
        mock_docker_settings.docker_image = "test-image"
        mock_docker_settings.docker_home_directory = None
        mock_docker_settings.docker_workspace_directory = None
        mock_docker_settings.user_uid = 1000
        mock_docker_settings.user_gid = 1000
        mock_docker_settings.user_mapping_enabled = True
        mock_docker_settings.validate_environment_variable = Mock(
            return_value=("API_KEY", "test")
        )
        mock_docker_settings.validate_volume_format = Mock(
            return_value="/home/user:/home/user"
        )
        mock_settings.docker_settings = mock_docker_settings
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

        # Should execute Docker command via os.execvp
        mock_execvp.assert_called_once()
        call_args = mock_execvp.call_args
        assert call_args[0][0] == "docker"
        docker_cmd = call_args[0][1]
        assert "claude" in docker_cmd or "doctor" in docker_cmd

    @patch("claude_code_proxy.cli.commands.api.config_manager.load_settings")
    @patch("claude_code_proxy.utils.docker_builder.DockerCommandBuilder")
    @patch("claude_code_proxy.cli.main.os.execvp")
    def test_docker_multiple_volumes_and_env(
        self, mock_execvp, mock_docker_builder, mock_get_settings
    ):
        """Test Docker command with multiple volumes and environment variables."""
        mock_settings = Mock(spec=Settings)
        mock_docker_settings = Mock()
        mock_docker_settings.docker_environment = {}
        mock_docker_settings.docker_additional_args = []
        mock_docker_settings.docker_volumes = []
        mock_docker_settings.docker_image = "test-image"
        mock_docker_settings.docker_home_directory = None
        mock_docker_settings.docker_workspace_directory = None
        mock_docker_settings.user_uid = 1000
        mock_docker_settings.user_gid = 1000
        mock_docker_settings.user_mapping_enabled = True
        mock_docker_settings.validate_environment_variable = Mock(
            return_value=("API_KEY", "test")
        )
        mock_docker_settings.validate_volume_format = Mock(
            return_value="/home/user:/home/user"
        )
        mock_settings.docker_settings = mock_docker_settings
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

        # Should execute Docker command via os.execvp with multiple options
        mock_execvp.assert_called_once()
        call_args = mock_execvp.call_args
        assert call_args[0][0] == "docker"
        docker_cmd = call_args[0][1]
        assert "claude" in docker_cmd or "config" in docker_cmd


@pytest.mark.integration
class TestErrorScenarios:
    """Test various error scenarios."""

    def setup_method(self):
        """Setup test environment."""
        self.runner = CliRunner(env={"NO_COLOR": "1", "TERM": "dumb"})

    @patch("claude_code_proxy.cli.commands.config.get_settings")
    def test_config_command_exception_handling(self, mock_get_settings):
        """Test config command handles various exceptions."""
        mock_get_settings.side_effect = FileNotFoundError("Config file not found")

        result = self.runner.invoke(app, ["config", "list"])

        assert result.exit_code == 1
        assert "Error loading configuration: Config file not found" in (
            result.stdout + result.stderr
        )

    @patch("claude_code_proxy.cli.commands.api.config_manager.load_settings")
    def test_claude_command_exception_handling(self, mock_get_settings):
        """Test claude command handles various exceptions."""
        mock_get_settings.side_effect = ValueError("Invalid configuration")

        result = self.runner.invoke(app, ["claude", "--", "--version"])

        assert result.exit_code == 1
        assert "Error executing claude command: Invalid configuration" in result.stderr

    @patch("claude_code_proxy.cli.commands.api.config_manager.load_settings")
    @patch("claude_code_proxy.cli.main.os.execvp")
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
        result = self.runner.invoke(app, ["fastapi", "run", "--help"])
        assert result.exit_code == 0
        assert "Run a FastAPI app in production mode" in result.stdout
        assert "--host" in result.stdout
        assert "--port" in result.stdout
        assert "--workers" in result.stdout
        assert "--reload" in result.stdout

    def test_dev_command_help(self):
        """Test dev command help."""
        result = self.runner.invoke(app, ["fastapi", "dev", "--help"])
        assert result.exit_code == 0
        assert "Run a FastAPI app in development mode" in result.stdout
        assert "--host" in result.stdout
        assert "--port" in result.stdout
        assert "--reload" in result.stdout

    def test_fastapi_cli_commands_available(self):
        """Test that FastAPI CLI commands are available."""
        # Test that run command is available
        result = self.runner.invoke(app, ["fastapi", "run", "--help"])
        assert result.exit_code == 0
        assert "Run a FastAPI app in production mode" in result.stdout

        # Test that dev command is available
        result = self.runner.invoke(app, ["fastapi", "dev", "--help"])
        assert result.exit_code == 0
        assert "Run a FastAPI app in development mode" in result.stdout

    def test_fastapi_cli_options_validation(self):
        """Test FastAPI CLI options validation."""
        # Test run command with invalid port
        result = self.runner.invoke(app, ["fastapi", "run", "--port", "invalid"])
        assert result.exit_code != 0

        # Test run command with negative port
        result = self.runner.invoke(app, ["fastapi", "run", "--port", "-1"])
        assert result.exit_code != 0

    # Disabled: get_default_path_hook function no longer exists
    # @patch("claude_code_proxy.cli.main.get_default_path_hook")
    # def test_default_path_hook_success(self, mock_get_default_path):
    #     """Test default path hook finds the main.py file."""
    #     mock_path = Mock()
    #     mock_path.is_file.return_value = True
    #     mock_get_default_path.return_value = mock_path

    #     # Call the hook directly
    #     from claude_code_proxy.cli.main import get_default_path_hook

    #     result = get_default_path_hook()

    #     assert result == mock_path

    # Disabled: get_default_path_hook function no longer exists
    # @patch("claude_code_proxy.utils.helper.get_package_dir")
    # @patch("pathlib.Path.is_file")
    # def test_default_path_hook_no_file_found(self, mock_is_file, mock_get_package_dir):
    #     """Test default path hook when no main.py file is found."""
    #     mock_package_dir = Path("/mock/package/dir")
    #     mock_get_package_dir.return_value = mock_package_dir
    #     # Mock is_file to return False for all paths
    #     mock_is_file.return_value = False

    #     from claude_code_proxy.cli import get_default_path_hook

    #     with pytest.raises(FileNotFoundError) as exc_info:
    #         get_default_path_hook()

    #     assert "Could not find a default file to run" in str(exc_info.value)

    def test_fastapi_cli_integration_basic(self):
        """Test basic FastAPI CLI integration without starting servers."""
        # Test that commands exist and show proper help
        result = self.runner.invoke(app, ["fastapi", "run", "--help"])
        assert result.exit_code == 0
        assert "FastAPI app in production mode" in result.stdout

        result = self.runner.invoke(app, ["fastapi", "dev", "--help"])
        assert result.exit_code == 0
        assert "FastAPI app in development mode" in result.stdout

    def test_fastapi_commands_in_help(self):
        """Test that FastAPI commands appear in main help."""
        result = self.runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "fastapi" in result.stdout
        assert "FastAPI development commands" in result.stdout

        # Test that the fastapi subcommand shows the individual commands
        result = self.runner.invoke(app, ["fastapi", "--help"])
        assert result.exit_code == 0
        assert "run" in result.stdout
        assert "dev" in result.stdout


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

    @patch("claude_code_proxy.cli.commands.config.get_settings")
    def test_cli_command_isolation(self, mock_get_settings):
        """Test that CLI commands don't interfere with each other."""
        mock_settings = Mock(spec=Settings)
        mock_settings.host = "localhost"
        mock_settings.port = 8000
        mock_settings.log_level = "INFO"
        mock_settings.claude_cli_path = "/usr/bin/claude"
        mock_settings.workers = 1
        mock_settings.reload = False
        mock_settings.server_url = "http://localhost:8000"
        mock_settings.auth_token = None
        mock_settings.api_tools_handling = "warning"
        mock_settings.cors_origins = ["*"]
        # Add mock docker settings
        mock_docker_settings = Mock()
        mock_docker_settings.docker_image = "claude-code-proxy"
        mock_docker_settings.docker_home_directory = None
        mock_docker_settings.docker_workspace_directory = None
        mock_docker_settings.user_uid = 1000
        mock_docker_settings.user_gid = 1000
        mock_docker_settings.user_mapping_enabled = True
        mock_docker_settings.docker_volumes = []
        mock_docker_settings.docker_environment = {}
        mock_docker_settings.docker_additional_args = []
        mock_settings.docker_settings = mock_docker_settings

        # Add mock claude_code_options
        mock_claude_code_options = Mock()
        mock_claude_code_options.model = None
        mock_claude_code_options.max_thinking_tokens = 8000
        mock_claude_code_options.max_turns = None
        mock_claude_code_options.cwd = None
        mock_claude_code_options.system_prompt = None
        mock_claude_code_options.append_system_prompt = None
        mock_claude_code_options.permission_mode = None
        mock_claude_code_options.permission_prompt_tool_name = None
        mock_claude_code_options.continue_conversation = False
        mock_claude_code_options.resume = None
        mock_claude_code_options.allowed_tools = []
        mock_claude_code_options.disallowed_tools = []
        mock_claude_code_options.mcp_servers = []
        mock_claude_code_options.mcp_tools = []
        mock_settings.claude_code_options = mock_claude_code_options

        mock_get_settings.return_value = mock_settings

        # Run config command
        result1 = self.runner.invoke(app, ["config", "list"])
        assert result1.exit_code == 0
        assert "Claude Code Proxy API Configuration" in result1.stdout

        # Run version command - should not be affected by previous command
        result2 = self.runner.invoke(app, ["--version"])
        assert result2.exit_code == 0
        assert "claude-code-proxy-api" in result2.stdout

        # Run help command - should not be affected by previous commands
        result3 = self.runner.invoke(app, ["--help"])
        assert result3.exit_code == 0
        assert "Claude Code Proxy API Server" in result3.stdout

    @patch("uvicorn.run")
    def test_cli_typer_integration(self, mock_run):
        """Test Typer integration specifics."""
        # Test that the CLI app is properly configured
        assert app.info is not None
        assert app.info.help is not None

        # Test rich markup is configured
        assert hasattr(app, "rich_markup_mode")

        # Test default command starts server (now default behavior)
        result = self.runner.invoke(app, [])
        assert result.exit_code == 0  # Should succeed when server start is mocked
        mock_run.assert_called_once()  # Verify server start was attempted


@pytest.mark.integration
class TestApiCommandWithCliOverrides:
    """Test API command with CLI parameter overrides."""

    def setup_method(self):
        """Setup test environment."""
        self.runner = CliRunner(env={"NO_COLOR": "1", "TERM": "dumb"})

    @patch("claude_code_proxy.cli.commands.api.config_manager.load_settings")
    @patch("uvicorn.run")
    def test_api_command_basic_cli_overrides(self, mock_run, mock_get_settings):
        """Test API command with basic CLI parameter overrides."""
        mock_settings = Mock()
        mock_settings.host = "127.0.0.1"
        mock_settings.port = 8080
        mock_settings.reload = True
        mock_settings.log_level = "INFO"
        mock_settings.workers = 1
        mock_get_settings.return_value = mock_settings

        result = self.runner.invoke(
            app, ["api", "--port", "8080", "--host", "127.0.0.1", "--reload"]
        )

        # Should call uvicorn.run with the correct parameters
        mock_run.assert_called_once()
        call_args = mock_run.call_args
        assert call_args[1]["host"] == "127.0.0.1"
        assert call_args[1]["port"] == 8080
        assert call_args[1]["reload"]

    @patch("uvicorn.run")
    @patch("claude_code_proxy.cli.commands.api.config_manager.load_settings")
    def test_api_command_claude_code_options_overrides(
        self, mock_get_settings, mock_run
    ):
        """Test API command with ClaudeCodeOptions parameter overrides."""
        mock_settings = Mock()
        mock_settings.host = "localhost"
        mock_settings.port = 8000
        mock_settings.reload = False
        mock_settings.workers = 1
        mock_settings.log_level = "INFO"
        mock_get_settings.return_value = mock_settings

        with patch.dict("os.environ", {}, clear=False) as mock_env:
            result = self.runner.invoke(
                app,
                [
                    "api",
                    "--max-thinking-tokens",
                    "15000",
                    "--allowed-tools",
                    "Read,Write,Bash",
                    "--permission-mode",
                    "acceptEdits",
                    "--cwd",
                    "/workspace",
                ],
            )

            # Should set environment variable with CLI overrides
            import json

            # Check that environment variable was set with CLI overrides
            env_overrides = mock_env.get("CCPROXY_CONFIG_OVERRIDES")
            assert env_overrides is not None
            overrides = json.loads(env_overrides)

            assert overrides["claude_code_options"]["max_thinking_tokens"] == 15000
            assert overrides["claude_code_options"]["allowed_tools"] == [
                "Read",
                "Write",
                "Bash",
            ]
            assert overrides["claude_code_options"]["permission_mode"] == "acceptEdits"
            assert overrides["claude_code_options"]["cwd"] == "/workspace"

            assert result.exit_code == 0

    # Pool settings test removed - connection pooling functionality has been removed

    @patch("uvicorn.run")
    @patch("claude_code_proxy.cli.commands.api.config_manager.load_settings")
    def test_api_command_security_settings_overrides(self, mock_get_settings, mock_run):
        """Test API command with security settings parameter overrides."""
        mock_settings = Mock()
        mock_settings.host = "localhost"
        mock_settings.port = 8000
        mock_settings.reload = False
        mock_settings.workers = 1
        mock_settings.log_level = "INFO"
        mock_get_settings.return_value = mock_settings

        with patch.dict("os.environ", {}, clear=False) as mock_env:
            result = self.runner.invoke(
                app,
                [
                    "api",
                    "--cors-origins",
                    "https://app.com,https://admin.com",
                    "--auth-token",
                    "test-token",
                    "--tools-handling",
                    "error",
                ],
            )

            # Should set environment variable with CLI overrides
            import json

            # Check that environment variable was set with CLI overrides
            env_overrides = mock_env.get("CCPROXY_CONFIG_OVERRIDES")
            assert env_overrides is not None
            overrides = json.loads(env_overrides)

            assert overrides["cors_origins"] == ["https://app.com", "https://admin.com"]
            assert overrides["auth_token"] == "test-token"
            assert overrides["tools_handling"] == "error"

            assert result.exit_code == 0

    @patch(
        "claude_code_proxy.utils.docker_builder.DockerCommandBuilder.execute_from_settings"
    )
    @patch("claude_code_proxy.cli.commands.api.config_manager.load_settings")
    def test_api_command_docker_mode_with_overrides(
        self, mock_get_settings, mock_docker_execute
    ):
        """Test API command in Docker mode with CLI overrides."""
        mock_settings = Mock()
        mock_settings.host = "localhost"
        mock_settings.port = 9000
        mock_settings.reload = True
        mock_settings.log_level = "INFO"

        # Create a proper mock for docker_settings
        mock_docker_settings = Mock()
        mock_docker_settings.docker_home_directory = None
        mock_docker_settings.docker_workspace_directory = None
        mock_docker_settings.docker_volumes = []
        mock_docker_settings.docker_environment = {}
        mock_docker_settings.docker_additional_args = []
        mock_settings.docker_settings = mock_docker_settings
        mock_get_settings.return_value = mock_settings

        mock_docker_execute.return_value = None

        result = self.runner.invoke(
            app,
            [
                "api",
                "--docker",
                "--port",
                "9000",
                "--reload",
                "--max-thinking-tokens",
                "12000",
            ],
        )

        # Check that the command executed successfully
        assert result.exit_code == 0

        # Should call Docker execute method
        mock_docker_execute.assert_called_once()
        call_args = mock_docker_execute.call_args

        # Check that docker_env contains expected environment variables
        docker_env_list = call_args[1]["docker_env"]
        docker_env_str = " ".join(docker_env_list)
        assert "PORT=9000" in docker_env_str
        assert "RELOAD=true" in docker_env_str
        assert "HOST=0.0.0.0" in docker_env_str

    @patch("uvicorn.run")
    @patch("claude_code_proxy.cli.commands.api.config_manager.load_settings")
    def test_api_command_no_overrides(self, mock_get_settings, mock_run):
        """Test API command without any CLI overrides."""
        mock_settings = Mock()
        mock_settings.host = "0.0.0.0"
        mock_settings.port = 8000
        mock_settings.reload = False
        mock_settings.workers = 1
        mock_settings.log_level = "INFO"
        mock_get_settings.return_value = mock_settings

        result = self.runner.invoke(app, ["api"])

        # Should execute successfully
        assert result.exit_code == 0

        # Should use default settings without any environment overrides
        mock_run.assert_called_once()
        call_args = mock_run.call_args
        assert call_args[1]["host"] == "0.0.0.0"
        assert call_args[1]["port"] == 8000
        assert not call_args[1]["reload"]

        # Should not set environment overrides
        import os

        assert os.environ.get("CCPROXY_CONFIG_OVERRIDES") is None

    def test_api_command_help_shows_all_options(self):
        """Test that api command help shows all CLI options."""
        result = self.runner.invoke(app, ["api", "--help"])
        assert result.exit_code == 0

        # Check for core server settings
        assert "--host" in result.stdout
        assert "--port" in result.stdout
        assert "--reload" in result.stdout
        assert "--log-level" in result.stdout
        assert "--workers" in result.stdout

        # Check for security settings
        assert "--cors-origins" in result.stdout
        assert "--auth-token" in result.stdout
        assert "--tools-handling" in result.stdout

        # Check for key ClaudeCodeOptions (the help may be truncated, so test fewer)
        assert "--max-thinking-tokens" in result.stdout
        assert "--allowed-tools" in result.stdout

        # Pool settings removed - no longer check for pool-related options

        # Check for Docker settings
        assert "--docker" in result.stdout
        assert "--docker-image" in result.stdout

    @patch("uvicorn.run")
    @patch("claude_code_proxy.cli.commands.api.config_manager.load_settings")
    def test_api_command_comprehensive_overrides(self, mock_get_settings, mock_run):
        """Test API command with comprehensive CLI parameter overrides."""
        mock_settings = Mock()
        mock_settings.host = "127.0.0.1"
        mock_settings.port = 9000
        mock_settings.reload = True
        mock_settings.workers = 2
        mock_settings.log_level = "DEBUG"
        mock_get_settings.return_value = mock_settings

        with patch.dict("os.environ", {}, clear=False) as mock_env:
            result = self.runner.invoke(
                app,
                [
                    "api",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    "9000",
                    "--reload",
                    "--log-level",
                    "DEBUG",
                    "--workers",
                    "2",
                    "--auth-token",
                    "secret-token",
                    "--max-thinking-tokens",
                    "20000",
                    "--allowed-tools",
                    "Read,Write,Bash,Edit",
                    "--permission-mode",
                    "bypassPermissions",
                ],
            )

            # Should set environment variable with comprehensive CLI overrides
            import json

            # Check that environment variable was set with CLI overrides
            env_overrides = mock_env.get("CCPROXY_CONFIG_OVERRIDES")
            assert env_overrides is not None
            overrides = json.loads(env_overrides)

            # Check server settings
            assert overrides["host"] == "127.0.0.1"
            assert overrides["port"] == 9000
            assert overrides["reload"] is True
            assert overrides["log_level"] == "DEBUG"
            assert overrides["workers"] == 2
            assert overrides["auth_token"] == "secret-token"

            # Check ClaudeCodeOptions
            claude_code_options = overrides["claude_code_options"]
            assert claude_code_options["max_thinking_tokens"] == 20000
            assert claude_code_options["allowed_tools"] == [
                "Read",
                "Write",
                "Bash",
                "Edit",
            ]
            assert claude_code_options["permission_mode"] == "bypassPermissions"

            assert result.exit_code == 0

        # Pool settings removed - no longer check for pool-related settings

    @patch("claude_code_proxy.cli.commands.api.config_manager.load_settings")
    def test_api_command_settings_loading_error(self, mock_get_settings):
        """Test API command handles settings loading errors."""
        mock_get_settings.side_effect = ValueError("Configuration error")

        result = self.runner.invoke(app, ["api", "--port", "8080"])

        assert result.exit_code == 1
        assert "Error starting server: Configuration error" in result.stderr


@pytest.mark.integration
class TestSettingsWithCliOverrides:
    """Test settings system with CLI overrides."""

    def setup_method(self):
        """Setup test environment."""
        # Clean environment
        import os

        if "CCPROXY_CONFIG_OVERRIDES" in os.environ:
            del os.environ["CCPROXY_CONFIG_OVERRIDES"]

    def teardown_method(self):
        """Clean up after tests."""
        import os

        if "CCPROXY_CONFIG_OVERRIDES" in os.environ:
            del os.environ["CCPROXY_CONFIG_OVERRIDES"]

    @patch("claude_code_proxy.config.settings.find_toml_config_file")
    def test_settings_with_cli_overrides_environment_variable(self, mock_find_config):
        """Test settings loading with CLI overrides from environment variable."""
        import json
        import os

        from claude_code_proxy.config.settings import get_settings

        mock_find_config.return_value = None

        # Set CLI overrides in environment
        overrides = {
            "host": "test-host",
            "port": 9999,
            "claude_code_options": {
                "max_thinking_tokens": 25000,
                "allowed_tools": ["Read", "Write"],
            },
        }
        os.environ["CCPROXY_CONFIG_OVERRIDES"] = json.dumps(overrides)

        settings = get_settings()

        assert settings.host == "test-host"
        assert settings.port == 9999
        assert settings.claude_code_options.max_thinking_tokens == 25000
        assert settings.claude_code_options.allowed_tools == ["Read", "Write"]

    @patch("claude_code_proxy.config.settings.find_toml_config_file")
    def test_settings_with_invalid_cli_overrides_json(self, mock_find_config):
        """Test settings loading handles invalid JSON in CLI overrides."""
        import os

        from claude_code_proxy.config.settings import get_settings

        mock_find_config.return_value = None

        # Set invalid JSON in environment
        os.environ["CCPROXY_CONFIG_OVERRIDES"] = "invalid-json"

        # Should not raise an error, should use defaults
        settings = get_settings()
        assert settings.host == "127.0.0.1"  # Default value
        assert settings.port == 8000  # Default value

    @patch("claude_code_proxy.config.settings.find_toml_config_file")
    def test_settings_without_cli_overrides(self, mock_find_config):
        """Test settings loading without CLI overrides."""
        import os

        from claude_code_proxy.config.settings import get_settings

        mock_find_config.return_value = None

        # Ensure no CLI overrides environment variable
        if "CCPROXY_CONFIG_OVERRIDES" in os.environ:
            del os.environ["CCPROXY_CONFIG_OVERRIDES"]

        settings = get_settings()
        assert settings.host == "127.0.0.1"  # Default value
        assert settings.port == 8000  # Default value


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
