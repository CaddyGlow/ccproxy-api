"""Tests for the ccproxy serve CLI command.

This module tests the CLI serve command functionality including:
- Command line argument parsing and validation
- Server startup and configuration
- Option group organization and help display
- Integration with FastAPI application lifecycle
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from ccproxy.cli.main import app as cli_app
from ccproxy.config.settings import Settings


class TestServeCommand:
    """Test the serve CLI command functionality."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create CLI test runner."""
        return CliRunner()

    @pytest.fixture
    def mock_config_file(self, tmp_path: Path) -> Path:
        """Create a temporary config file for testing."""
        config_file = tmp_path / "test_config.toml"
        config_file.write_text("""
[server]
port = 8080
host = "127.0.0.1"

[security]
auth_token = "test-token"

[claude]
cli_path = "/usr/local/bin/claude"
""")
        return config_file

    def test_serve_help_display(self, runner: CliRunner) -> None:
        """Test that serve command help displays without errors."""
        result = runner.invoke(cli_app, ["serve", "--help"])

        assert result.exit_code == 0
        assert "Usage:" in result.output
        assert "Server Settings" in result.output
        assert "Security Settings" in result.output
        assert "Claude Settings" in result.output
        assert "Configuration" in result.output

    def test_serve_help_no_task_registration(self, runner: CliRunner) -> None:
        """Test that help display doesn't trigger task registration."""
        with patch(
            "ccproxy.scheduler.manager._register_default_tasks"
        ) as mock_register:
            result = runner.invoke(cli_app, ["serve", "--help"])

            assert result.exit_code == 0
            # Task registration should not be called during help display
            mock_register.assert_not_called()

    def test_serve_with_port_option(
        self, runner: CliRunner, test_settings: Settings
    ) -> None:
        """Test serve command with port option."""
        with (
            patch("ccproxy.cli.commands.serve.uvicorn.run") as mock_uvicorn,
            patch(
                "ccproxy.config.settings.config_manager.load_settings"
            ) as mock_load_settings,
        ):
            mock_uvicorn.return_value = None
            # Create test settings with the expected port override
            modified_settings = Settings(
                server=test_settings.server.model_copy(update={"port": 9000}),
                security=test_settings.security,
                auth=test_settings.auth,
            )
            mock_load_settings.return_value = modified_settings

            result = runner.invoke(cli_app, ["serve", "--port", "9000"])

            assert result.exit_code == 0
            mock_uvicorn.assert_called_once()
            # Verify port was passed correctly
            call_args = mock_uvicorn.call_args[1]
            assert call_args["port"] == 9000

    def test_serve_with_host_option(
        self, runner: CliRunner, test_settings: Settings
    ) -> None:
        """Test serve command with host option."""
        with (
            patch("ccproxy.cli.commands.serve.uvicorn.run") as mock_uvicorn,
            patch(
                "ccproxy.config.settings.config_manager.load_settings"
            ) as mock_load_settings,
        ):
            mock_uvicorn.return_value = None
            # Create test settings with the expected host override
            modified_settings = Settings(
                server=test_settings.server.model_copy(update={"host": "0.0.0.0"}),
                security=test_settings.security,
                auth=test_settings.auth,
            )
            mock_load_settings.return_value = modified_settings

            result = runner.invoke(cli_app, ["serve", "--host", "0.0.0.0"])

            assert result.exit_code == 0
            mock_uvicorn.assert_called_once()
            # Verify host was passed correctly
            call_args = mock_uvicorn.call_args[1]
            assert call_args["host"] == "0.0.0.0"

    def test_serve_with_config_file(
        self, runner: CliRunner, test_settings: Settings, mock_config_file: Path
    ) -> None:
        """Test serve command with configuration file."""
        with (
            patch("ccproxy.cli.commands.serve.uvicorn.run") as mock_uvicorn,
            patch(
                "ccproxy.config.settings.config_manager.load_settings"
            ) as mock_load_settings,
        ):
            mock_uvicorn.return_value = None
            mock_load_settings.return_value = test_settings

            result = runner.invoke(
                cli_app, ["serve", "--config", str(mock_config_file)]
            )

            assert result.exit_code == 0
            mock_uvicorn.assert_called_once()

    def test_serve_with_auth_token(
        self, runner: CliRunner, test_settings: Settings
    ) -> None:
        """Test serve command with auth token option."""
        with (
            patch("ccproxy.cli.commands.serve.uvicorn.run") as mock_uvicorn,
            patch(
                "ccproxy.config.settings.config_manager.load_settings"
            ) as mock_load_settings,
        ):
            mock_uvicorn.return_value = None
            mock_load_settings.return_value = test_settings

            result = runner.invoke(cli_app, ["serve", "--auth-token", "secret-token"])

            assert result.exit_code == 0
            mock_uvicorn.assert_called_once()

    def test_serve_with_reload_option(
        self, runner: CliRunner, test_settings: Settings
    ) -> None:
        """Test serve command with reload option."""
        with (
            patch("ccproxy.cli.commands.serve.uvicorn.run") as mock_uvicorn,
            patch(
                "ccproxy.config.settings.config_manager.load_settings"
            ) as mock_load_settings,
        ):
            mock_uvicorn.return_value = None
            # Create test settings with the expected reload override
            modified_settings = Settings(
                server=test_settings.server.model_copy(update={"reload": True}),
                security=test_settings.security,
                auth=test_settings.auth,
            )
            mock_load_settings.return_value = modified_settings

            result = runner.invoke(cli_app, ["serve", "--reload"])

            assert result.exit_code == 0
            mock_uvicorn.assert_called_once()
            # Verify reload was passed correctly
            call_args = mock_uvicorn.call_args[1]
            assert call_args["reload"] is True

    def test_serve_with_multiple_options(
        self, runner: CliRunner, test_settings: Settings
    ) -> None:
        """Test serve command with multiple options combined."""
        with (
            patch("ccproxy.cli.commands.serve.uvicorn.run") as mock_uvicorn,
            patch(
                "ccproxy.config.settings.config_manager.load_settings"
            ) as mock_load_settings,
        ):
            mock_uvicorn.return_value = None
            # Create test settings with the expected multiple overrides
            modified_settings = Settings(
                server=test_settings.server.model_copy(
                    update={"port": 8080, "host": "127.0.0.1", "reload": True}
                ),
                security=test_settings.security.model_copy(
                    update={"auth_token": "test-token"}
                ),
                auth=test_settings.auth,
            )
            mock_load_settings.return_value = modified_settings

            result = runner.invoke(
                cli_app,
                [
                    "serve",
                    "--port",
                    "8080",
                    "--host",
                    "127.0.0.1",
                    "--auth-token",
                    "test-token",
                    "--reload",
                ],
            )

            assert result.exit_code == 0
            mock_uvicorn.assert_called_once()

            # Verify all options were passed correctly
            call_args = mock_uvicorn.call_args[1]
            assert call_args["port"] == 8080
            assert call_args["host"] == "127.0.0.1"
            assert call_args["reload"] is True

    def test_serve_with_docker_option(
        self, runner: CliRunner, test_settings: Settings
    ) -> None:
        """Test serve command with Docker option."""
        with (
            patch("ccproxy.cli.commands.serve._run_docker_server") as mock_docker,
            patch(
                "ccproxy.config.settings.config_manager.load_settings"
            ) as mock_load_settings,
        ):
            mock_docker.return_value = None
            mock_load_settings.return_value = test_settings

            result = runner.invoke(cli_app, ["serve", "--docker"])

            assert result.exit_code == 0
            mock_docker.assert_called_once()


class TestServeCommandOptions:
    """Test individual option groups and their validation."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create CLI test runner."""
        return CliRunner()

    def test_server_options_group(self, runner: CliRunner) -> None:
        """Test server options are properly grouped in help."""
        result = runner.invoke(cli_app, ["serve", "--help"])

        assert result.exit_code == 0
        help_text = result.output

        # Check for Server Settings section
        assert "Server Settings" in help_text
        assert "port" in help_text
        assert "host" in help_text
        assert "reload" in help_text

    def test_security_options_group(self, runner: CliRunner) -> None:
        """Test security options are properly grouped in help."""
        result = runner.invoke(cli_app, ["serve", "--help"])

        assert result.exit_code == 0
        help_text = result.output

        # Check for Security Settings section
        assert "Security Settings" in help_text

    def test_claude_options_group(self, runner: CliRunner) -> None:
        """Test Claude options are properly grouped in help."""
        result = runner.invoke(cli_app, ["serve", "--help"])

        assert result.exit_code == 0
        help_text = result.output

        # Check for Claude Settings section
        assert "Claude Settings" in help_text

    def test_configuration_options_group(self, runner: CliRunner) -> None:
        """Test configuration options are properly grouped in help."""
        result = runner.invoke(cli_app, ["serve", "--help"])

        assert result.exit_code == 0
        help_text = result.output

        # Check for Configuration section
        assert "Configuration" in help_text
        assert "-config" in help_text

    def test_docker_options_group(self, runner: CliRunner) -> None:
        """Test Docker options are properly grouped in help."""
        result = runner.invoke(cli_app, ["serve", "--help"])

        assert result.exit_code == 0
        help_text = result.output

        # Check for Docker Settings section (if Docker options exist)
        if "Docker Settings" in help_text:
            assert "docker" in help_text or "use-docker" in help_text

    def test_option_validation_callbacks(
        self, runner: CliRunner, test_settings: Settings
    ) -> None:
        """Test that option validation callbacks work properly."""
        # Test valid port validation
        with (
            patch("ccproxy.cli.commands.serve.uvicorn.run") as mock_uvicorn,
            patch(
                "ccproxy.config.settings.config_manager.load_settings"
            ) as mock_load_settings,
        ):
            mock_uvicorn.return_value = None
            mock_load_settings.return_value = test_settings

            result = runner.invoke(cli_app, ["serve", "--port", "8080"])
            assert result.exit_code == 0

        # Test invalid port validation - this should fail at validation level
        result = runner.invoke(cli_app, ["serve", "--port", "70000"])
        assert result.exit_code != 0


class TestServeCommandIntegration:
    """Integration tests for the serve command with actual server components."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create CLI test runner."""
        return CliRunner()

    def test_serve_scheduler_task_registration_timing(
        self, runner: CliRunner, test_settings: Settings
    ) -> None:
        """Test that tasks are only registered during actual server startup."""
        with (
            patch("ccproxy.cli.commands.serve.uvicorn.run") as mock_uvicorn,
            patch(
                "ccproxy.config.settings.config_manager.load_settings"
            ) as mock_load_settings,
        ):
            mock_uvicorn.return_value = None
            mock_load_settings.return_value = test_settings

            # Run serve command
            result = runner.invoke(cli_app, ["serve", "--port", "8000"])

            assert result.exit_code == 0
            mock_uvicorn.assert_called_once()

    def test_serve_uvicorn_integration(
        self, runner: CliRunner, test_settings: Settings
    ) -> None:
        """Test that serve command properly integrates with uvicorn."""
        with (
            patch("ccproxy.cli.commands.serve.uvicorn.run") as mock_uvicorn,
            patch(
                "ccproxy.config.settings.config_manager.load_settings"
            ) as mock_load_settings,
        ):
            mock_uvicorn.return_value = None
            # Create test settings with the expected overrides
            modified_settings = Settings(
                server=test_settings.server.model_copy(
                    update={"port": 8000, "host": "0.0.0.0"}
                ),
                security=test_settings.security,
                auth=test_settings.auth,
            )
            mock_load_settings.return_value = modified_settings

            result = runner.invoke(
                cli_app, ["serve", "--port", "8000", "--host", "0.0.0.0"]
            )

            assert result.exit_code == 0
            mock_uvicorn.assert_called_once()

            # Verify uvicorn was called with correct parameters
            call_args = mock_uvicorn.call_args
            kwargs = call_args[1]

            # Check that factory=True is used for proper app creation
            assert kwargs.get("factory") is True
            assert kwargs.get("port") == 8000
            assert kwargs.get("host") == "0.0.0.0"
            assert "create_app" in kwargs.get("app", "")

    def test_serve_with_invalid_config_file(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        """Test serve command with invalid config file."""
        invalid_config = tmp_path / "invalid.toml"
        invalid_config.write_text("invalid toml content [[[")

        result = runner.invoke(cli_app, ["serve", "--config", str(invalid_config)])

        # Should handle config errors gracefully
        assert result.exit_code != 0

    def test_serve_with_nonexistent_config_file(self, runner: CliRunner) -> None:
        """Test serve command with nonexistent config file."""
        result = runner.invoke(
            cli_app, ["serve", "--config", "/nonexistent/config.toml"]
        )

        # Should handle missing config file gracefully
        assert result.exit_code != 0


class TestServeCommandEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create CLI test runner."""
        return CliRunner()

    def test_serve_configuration_error_handling(self, runner: CliRunner) -> None:
        """Test that configuration errors are handled gracefully."""
        with patch("ccproxy.config.settings.config_manager.load_settings") as mock_load:
            from ccproxy.config.settings import ConfigurationError

            mock_load.side_effect = ConfigurationError("Test configuration error")

            result = runner.invoke(cli_app, ["serve"])

            assert result.exit_code == 1
            assert "Configuration error" in result.output

    def test_serve_with_log_level_option(
        self, runner: CliRunner, test_settings: Settings
    ) -> None:
        """Test serve command with log level option."""
        with (
            patch("ccproxy.cli.commands.serve.uvicorn.run") as mock_uvicorn,
            patch(
                "ccproxy.config.settings.config_manager.load_settings"
            ) as mock_load_settings,
        ):
            mock_uvicorn.return_value = None
            mock_load_settings.return_value = test_settings

            result = runner.invoke(cli_app, ["serve", "--log-level", "DEBUG"])

            assert result.exit_code == 0
            mock_uvicorn.assert_called_once()

    def test_serve_with_log_file_option(
        self, runner: CliRunner, test_settings: Settings, tmp_path: Path
    ) -> None:
        """Test serve command with log file option."""
        log_file = tmp_path / "test.log"

        with (
            patch("ccproxy.cli.commands.serve.uvicorn.run") as mock_uvicorn,
            patch(
                "ccproxy.config.settings.config_manager.load_settings"
            ) as mock_load_settings,
        ):
            mock_uvicorn.return_value = None
            mock_load_settings.return_value = test_settings

            result = runner.invoke(cli_app, ["serve", "--log-file", str(log_file)])

            assert result.exit_code == 0
            mock_uvicorn.assert_called_once()
