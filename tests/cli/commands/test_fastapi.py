"""Tests for FastAPI CLI commands."""

import sys
from unittest.mock import MagicMock, Mock, patch

import pytest
from typer.testing import CliRunner


def _setup_mocks():
    """Set up mocks for imports."""
    # Mock the toolkit before any imports
    mock_toolkit = Mock()
    mock_toolkit.print = Mock()
    sys.modules["claude_code_proxy.utils.cli"] = Mock(
        get_rich_toolkit=Mock(return_value=mock_toolkit),
        get_uvicorn_log_config=Mock(return_value={"version": 1}),
        RichToolkit=Mock(return_value=mock_toolkit),
    )


# Set up mocks
_setup_mocks()

# Now we can safely import
from claude_code_proxy.cli.commands.fastapi import _run, app  # noqa: E402
from claude_code_proxy.config.settings import ConfigurationError  # noqa: E402


@pytest.fixture
def runner():
    """Create a CLI test runner."""
    return CliRunner()


class TestFastAPICommands:
    """Test FastAPI CLI commands."""

    @patch("claude_code_proxy.cli.commands.fastapi.uvicorn.run")
    @patch("claude_code_proxy.cli.commands.fastapi.config_manager")
    @patch("claude_code_proxy.cli.commands.fastapi.get_config_path_from_context")
    def test_run_command(
        self, mock_get_config_path, mock_config_manager, mock_uvicorn_run, runner
    ):
        """Test the run command."""
        # Setup mocks
        mock_get_config_path.return_value = None
        mock_settings = MagicMock()
        mock_config_manager.load_settings.return_value = mock_settings

        # Run command
        result = runner.invoke(app, ["run", "--host", "0.0.0.0", "--port", "8080"])

        # Assertions
        assert result.exit_code == 0
        mock_config_manager.load_settings.assert_called_once()
        mock_config_manager.setup_logging.assert_called_once()
        mock_uvicorn_run.assert_called_once()

        # Check uvicorn.run arguments
        call_args = mock_uvicorn_run.call_args[1]
        assert call_args["host"] == "0.0.0.0"
        assert call_args["port"] == 8080
        assert call_args["reload"] is False

    @patch("claude_code_proxy.cli.commands.fastapi.uvicorn.run")
    @patch("claude_code_proxy.cli.commands.fastapi.config_manager")
    @patch("claude_code_proxy.cli.commands.fastapi.get_config_path_from_context")
    def test_dev_command(
        self, mock_get_config_path, mock_config_manager, mock_uvicorn_run, runner
    ):
        """Test the dev command."""
        # Setup mocks
        mock_get_config_path.return_value = None
        mock_settings = MagicMock()
        mock_config_manager.load_settings.return_value = mock_settings

        # Run command
        result = runner.invoke(app, ["dev", "--host", "localhost", "--port", "3000"])

        # Assertions
        assert result.exit_code == 0
        mock_config_manager.load_settings.assert_called_once()
        mock_config_manager.setup_logging.assert_called_once()
        mock_uvicorn_run.assert_called_once()

        # Check uvicorn.run arguments
        call_args = mock_uvicorn_run.call_args[1]
        assert call_args["host"] == "localhost"
        assert call_args["port"] == 3000
        assert call_args["reload"] is True  # Dev mode has reload enabled by default

    @patch("claude_code_proxy.cli.commands.fastapi.uvicorn.run")
    @patch("claude_code_proxy.cli.commands.fastapi.config_manager")
    @patch("claude_code_proxy.cli.commands.fastapi.get_config_path_from_context")
    def test_run_command_with_reload(
        self, mock_get_config_path, mock_config_manager, mock_uvicorn_run, runner
    ):
        """Test the run command with reload enabled."""
        # Setup mocks
        mock_get_config_path.return_value = None
        mock_settings = MagicMock()
        mock_config_manager.load_settings.return_value = mock_settings

        # Run command with reload
        result = runner.invoke(app, ["run", "--reload"])

        # Assertions
        assert result.exit_code == 0
        call_args = mock_uvicorn_run.call_args[1]
        assert call_args["reload"] is True

    @patch("claude_code_proxy.cli.commands.fastapi.uvicorn.run")
    @patch("claude_code_proxy.cli.commands.fastapi.config_manager")
    @patch("claude_code_proxy.cli.commands.fastapi.get_config_path_from_context")
    def test_dev_command_no_reload(
        self, mock_get_config_path, mock_config_manager, mock_uvicorn_run, runner
    ):
        """Test the dev command with reload disabled."""
        # Setup mocks
        mock_get_config_path.return_value = None
        mock_settings = MagicMock()
        mock_config_manager.load_settings.return_value = mock_settings

        # Run command with no-reload
        result = runner.invoke(app, ["dev", "--no-reload"])

        # Assertions
        assert result.exit_code == 0
        call_args = mock_uvicorn_run.call_args[1]
        assert call_args["reload"] is False

    @patch("claude_code_proxy.cli.commands.fastapi.config_manager")
    @patch("claude_code_proxy.cli.commands.fastapi.get_config_path_from_context")
    @patch("claude_code_proxy.cli.commands.fastapi.get_rich_toolkit")
    def test_run_configuration_error(
        self, mock_get_toolkit, mock_get_config_path, mock_config_manager, runner
    ):
        """Test run command with configuration error."""
        # Setup mocks
        mock_get_config_path.return_value = None
        mock_config_manager.load_settings.side_effect = ConfigurationError(
            "Invalid config"
        )

        # Mock the toolkit to capture error output
        mock_toolkit = MagicMock()
        mock_get_toolkit.return_value = mock_toolkit

        # Run command
        result = runner.invoke(app, ["run"])

        # Assertions
        assert result.exit_code == 1
        mock_toolkit.print.assert_called_with(
            "Configuration error: Invalid config", tag="error"
        )

    @patch("claude_code_proxy.cli.commands.fastapi.uvicorn.run")
    @patch("claude_code_proxy.cli.commands.fastapi.config_manager")
    @patch("claude_code_proxy.cli.commands.fastapi.get_config_path_from_context")
    @patch("claude_code_proxy.cli.commands.fastapi.get_rich_toolkit")
    def test_run_generic_exception(
        self,
        mock_get_toolkit,
        mock_get_config_path,
        mock_config_manager,
        mock_uvicorn_run,
        runner,
    ):
        """Test run command with generic exception."""
        # Setup mocks
        mock_get_config_path.return_value = None
        mock_settings = MagicMock()
        mock_config_manager.load_settings.return_value = mock_settings
        mock_uvicorn_run.side_effect = Exception("Server error")

        # Mock the toolkit to capture error output
        mock_toolkit = MagicMock()
        mock_get_toolkit.return_value = mock_toolkit

        # Run command
        result = runner.invoke(app, ["run"])

        # Assertions
        assert result.exit_code == 1
        mock_toolkit.print.assert_called_with(
            "Error starting production server: Server error", tag="error"
        )


class TestRunHelper:
    """Test the _run helper function."""

    @patch("claude_code_proxy.cli.commands.fastapi.uvicorn.run")
    @patch("claude_code_proxy.cli.commands.fastapi.config_manager")
    @patch("claude_code_proxy.cli.commands.fastapi.get_config_path_from_context")
    @patch("claude_code_proxy.cli.commands.fastapi.get_uvicorn_log_config")
    def test_run_helper_production(
        self,
        mock_get_log_config,
        mock_get_config_path,
        mock_config_manager,
        mock_uvicorn_run,
    ):
        """Test _run helper function for production mode."""
        # Setup mocks
        mock_get_config_path.return_value = None
        mock_settings = MagicMock()
        mock_config_manager.load_settings.return_value = mock_settings
        mock_log_config = {"version": 1}
        mock_get_log_config.return_value = mock_log_config

        # Call _run
        _run("production", "0.0.0.0", 8000, False)

        # Assertions
        mock_config_manager.load_settings.assert_called_once()
        mock_config_manager.setup_logging.assert_called_once()
        mock_uvicorn_run.assert_called_once_with(
            app="claude_code_proxy:app",
            host="0.0.0.0",
            port=8000,
            reload=False,
            workers=None,
            log_config=mock_log_config,
        )

    @patch("claude_code_proxy.cli.commands.fastapi.uvicorn.run")
    @patch("claude_code_proxy.cli.commands.fastapi.config_manager")
    @patch("claude_code_proxy.cli.commands.fastapi.get_config_path_from_context")
    def test_run_helper_with_workers(
        self, mock_get_config_path, mock_config_manager, mock_uvicorn_run
    ):
        """Test _run helper function with workers."""
        # Setup mocks
        mock_get_config_path.return_value = None
        mock_settings = MagicMock()
        mock_config_manager.load_settings.return_value = mock_settings

        # Call _run with workers
        _run("production", "127.0.0.1", 8080, False, workers=4)

        # Assertions
        call_args = mock_uvicorn_run.call_args[1]
        assert call_args["workers"] == 4
