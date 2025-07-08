"""Integration tests for CLI commands."""

from unittest.mock import Mock, patch

import pytest
from typer.testing import CliRunner

from ccproxy.cli.main import app
from ccproxy.config.settings import Settings


@pytest.mark.integration
class TestCLIIntegration:
    """Integration tests for CLI commands."""

    def setup_method(self):
        """Setup test environment."""
        self.runner = CliRunner(env={"NO_COLOR": "1", "TERM": "dumb"})

    def test_version_flag(self):
        """Test version flag works correctly."""
        result = self.runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert "ccproxy" in result.stdout

    def test_help_flag(self):
        """Test help flag shows usage information."""
        result = self.runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "Claude Code Proxy API Server" in result.stdout

    def test_api_command_help(self):
        """Test api command help."""
        result = self.runner.invoke(app, ["api", "--help"])
        assert result.exit_code == 0
        assert "Start the Claude Code Proxy API server" in result.stdout

    def test_claude_command_help(self):
        """Test claude command help."""
        result = self.runner.invoke(app, ["claude", "--help"])
        assert result.exit_code == 0
        assert "Execute claude CLI commands" in result.stdout

    def test_config_command_help(self):
        """Test config command help."""
        result = self.runner.invoke(app, ["config", "--help"])
        assert result.exit_code == 0

    def test_fastapi_command_help(self):
        """Test fastapi command help."""
        result = self.runner.invoke(app, ["fastapi", "--help"])
        assert result.exit_code == 0

    @patch("uvicorn.run")
    @patch("ccproxy.cli.commands.api.config_manager.load_settings")
    def test_api_command_basic(self, mock_load_settings, mock_uvicorn_run):
        """Test api command with mocked dependencies."""
        # Create a comprehensive mock settings object
        mock_settings = Mock(spec=Settings)
        mock_settings.host = "127.0.0.1"
        mock_settings.port = 8000
        mock_settings.workers = 1
        mock_settings.reload = False
        mock_settings.log_level = "INFO"
        mock_settings.auth_token = None
        mock_settings.server_url = "http://127.0.0.1:8000"
        mock_settings.cors_origins = ["*"]
        mock_settings.cors_credentials = True
        mock_settings.cors_methods = ["*"]
        mock_settings.cors_headers = ["*"]
        mock_settings.cors_origin_regex = None
        mock_settings.cors_expose_headers = []
        mock_settings.cors_max_age = 600
        mock_settings.api_tools_handling = "warning"
        mock_load_settings.return_value = mock_settings

        result = self.runner.invoke(app, ["api"])

        # Should run without error
        assert result.exit_code == 0
        # Should call uvicorn.run
        mock_uvicorn_run.assert_called_once()

    @patch("ccproxy.cli.commands.config.commands.get_settings")
    def test_config_list_command(self, mock_get_settings):
        """Test config list command with mocked settings."""
        # Create a comprehensive mock settings object
        mock_settings = Mock(spec=Settings)
        mock_settings.host = "127.0.0.1"
        mock_settings.port = 8000
        mock_settings.log_level = "INFO"
        mock_settings.workers = 1
        mock_settings.reload = False
        mock_settings.server_url = "http://127.0.0.1:8000"
        mock_settings.claude_cli_path = "/usr/bin/claude"
        mock_settings.auth_token = None
        mock_settings.api_tools_handling = "warning"
        mock_settings.cors_origins = ["*"]

        # Mock docker settings
        mock_docker_settings = Mock()
        mock_docker_settings.docker_image = "claude-proxy:latest"
        mock_docker_settings.docker_home_directory = None
        mock_docker_settings.docker_workspace_directory = None
        mock_docker_settings.docker_volumes = []
        mock_docker_settings.docker_environment = {}
        mock_docker_settings.docker_additional_args = []
        mock_docker_settings.user_mapping_enabled = True
        mock_docker_settings.user_uid = 1000
        mock_docker_settings.user_gid = 1000
        mock_settings.docker_settings = mock_docker_settings

        mock_get_settings.return_value = mock_settings

        result = self.runner.invoke(app, ["config", "list"])

        assert result.exit_code == 0
        assert "Claude Code Proxy API Configuration" in result.stdout

    @patch("os.execvp")
    @patch("ccproxy.cli.commands.claude.config_manager.load_settings")
    def test_claude_command_basic(self, mock_load_settings, mock_execvp):
        """Test claude command with mocked dependencies."""
        # Create a mock settings object
        mock_settings = Mock(spec=Settings)
        mock_settings.claude_cli_path = "/usr/bin/claude"
        mock_load_settings.return_value = mock_settings

        # Test should not actually execute - mock will prevent it
        result = self.runner.invoke(app, ["claude", "--", "--version"])

        # Command should prepare to execute
        assert result.exit_code == 0
        mock_execvp.assert_called_once()

    def test_permission_tool_basic(self):
        """Test permission tool command with safe input."""
        result = self.runner.invoke(
            app, ["permission-tool", "bash", '{"command": "ls -la"}']
        )

        assert result.exit_code == 0
        assert '"behavior":"allow"' in result.stdout
        assert '"updatedInput"' in result.stdout

    def test_permission_tool_dangerous_input(self):
        """Test permission tool rejects dangerous commands."""
        result = self.runner.invoke(
            app, ["permission-tool", "bash", '{"command": "rm -rf /"}']
        )

        assert result.exit_code == 0
        assert '"behavior":"deny"' in result.stdout


@pytest.mark.integration
class TestCLIDefaultBehavior:
    """Test CLI default behavior and argument handling."""

    def setup_method(self):
        """Setup test environment."""
        self.runner = CliRunner(env={"NO_COLOR": "1", "TERM": "dumb"})

    @patch("uvicorn.run")
    @patch("ccproxy.cli.commands.api.config_manager.load_settings")
    def test_no_command_defaults_to_api(self, mock_load_settings, mock_uvicorn_run):
        """Test that running with no command defaults to API server."""
        # Create a mock settings object
        mock_settings = Mock(spec=Settings)
        mock_settings.host = "127.0.0.1"
        mock_settings.port = 8000
        mock_settings.workers = 1
        mock_settings.reload = False
        mock_settings.log_level = "INFO"
        mock_settings.auth_token = None
        mock_settings.cors_origins = None
        mock_settings.claude_cli_path = None
        mock_settings.docker = False
        mock_settings.openai_compatibility = True
        mock_load_settings.return_value = mock_settings

        # Run with just flags, should default to api command
        result = self.runner.invoke(app, ["--port", "9000"])

        assert result.exit_code == 0
        mock_uvicorn_run.assert_called_once()
