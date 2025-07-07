"""Tests for the API command."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from claude_code_proxy.cli.commands.api import api, get_config_path_from_context
from claude_code_proxy.config.settings import Settings


@pytest.fixture
def runner():
    """Create a Typer test runner."""
    return CliRunner()


@pytest.fixture
def mock_settings():
    """Create mock settings."""
    settings = MagicMock(spec=Settings)
    settings.host = "127.0.0.1"
    settings.port = 8000
    settings.reload = False
    settings.log_level = "INFO"
    settings.auth_token = "test-token"

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


class TestApiCommand:
    """Test cases for the API command."""

    @patch("claude_code_proxy.cli.commands.api.config_manager")
    @patch("claude_code_proxy.cli.commands.api.uvicorn.run")
    def test_api_local_server(self, mock_uvicorn, mock_config_manager, mock_settings):
        """Test running API server locally."""
        mock_config_manager.load_settings.return_value = mock_settings
        mock_config_manager.setup_logging.return_value = None
        mock_config_manager.get_cli_overrides_from_args.return_value = {}

        # Create a test app with the api command
        from typer import Typer

        app = Typer()
        app.command()(api)

        runner = CliRunner()
        result = runner.invoke(app, ["--port", "8080", "--host", "0.0.0.0"])

        if result.exit_code != 0:
            print(f"Exit code: {result.exit_code}")
            print(f"Output: {result.output}")
            print(f"Exception: {result.exception}")
        assert result.exit_code == 0
        mock_config_manager.load_settings.assert_called_once()
        mock_uvicorn.assert_called_once()

    @patch("claude_code_proxy.cli.commands.api.config_manager")
    @patch("claude_code_proxy.cli.commands.api.create_docker_adapter")
    def test_api_docker_server(
        self, mock_docker_adapter, mock_config_manager, mock_settings
    ):
        """Test running API server with Docker."""
        mock_config_manager.load_settings.return_value = mock_settings
        mock_config_manager.setup_logging.return_value = None

        # Mock Docker adapter
        mock_adapter = MagicMock()
        mock_docker_adapter.return_value = mock_adapter

        # Create a test app with the api command
        from typer import Typer

        app = Typer()
        app.command()(api)

        runner = CliRunner()
        result = runner.invoke(app, ["--docker", "--docker-image", "custom:latest"])

        assert result.exit_code == 0
        mock_config_manager.load_settings.assert_called_once()
        mock_docker_adapter.assert_called_once()
        mock_adapter.exec_container.assert_called_once()

    @patch("claude_code_proxy.cli.commands.api.config_manager")
    def test_api_with_cli_overrides(self, mock_config_manager, mock_settings):
        """Test API command with CLI overrides."""
        mock_config_manager.load_settings.return_value = mock_settings
        mock_config_manager.get_cli_overrides_from_args.return_value = {
            "port": 9000,
            "host": "localhost",
            "max_thinking_tokens": 5000,
        }

        # Create a test app with the api command
        from typer import Typer

        app = Typer()
        app.command()(api)

        runner = CliRunner()
        with patch("claude_code_proxy.cli.commands.api.uvicorn.run"):
            result = runner.invoke(
                app,
                [
                    "--port",
                    "9000",
                    "--host",
                    "localhost",
                    "--max-thinking-tokens",
                    "5000",
                ],
            )

        assert result.exit_code == 0
        mock_config_manager.get_cli_overrides_from_args.assert_called_once()

    @patch("claude_code_proxy.cli.commands.api.config_manager")
    def test_api_configuration_error(self, mock_config_manager):
        """Test API command with configuration error."""
        from claude_code_proxy.config.settings import ConfigurationError

        mock_config_manager.load_settings.side_effect = ConfigurationError(
            "Invalid config"
        )

        # Create a test app with the api command
        from typer import Typer

        app = Typer()
        app.command()(api)

        runner = CliRunner()
        result = runner.invoke(app, [])

        assert result.exit_code == 1
        assert "Configuration error" in result.output

    def test_get_config_path_from_context_with_path(self):
        """Test get_config_path_from_context with a valid path."""
        with patch(
            "claude_code_proxy.cli.commands.api.get_current_context"
        ) as mock_context:
            mock_ctx = MagicMock()
            mock_ctx.obj = {"config_path": "/path/to/config.toml"}
            mock_context.return_value = mock_ctx

            result = get_config_path_from_context()
            assert result == Path("/path/to/config.toml")

    def test_get_config_path_from_context_none(self):
        """Test get_config_path_from_context with None."""
        with patch(
            "claude_code_proxy.cli.commands.api.get_current_context"
        ) as mock_context:
            mock_ctx = MagicMock()
            mock_ctx.obj = {"config_path": None}
            mock_context.return_value = mock_ctx

            result = get_config_path_from_context()
            assert result is None

    def test_get_config_path_from_context_no_context(self):
        """Test get_config_path_from_context with no active context."""
        with patch(
            "claude_code_proxy.cli.commands.api.get_current_context"
        ) as mock_context:
            mock_context.side_effect = RuntimeError("No active context")

            result = get_config_path_from_context()
            assert result is None

    @patch("claude_code_proxy.cli.commands.api.config_manager")
    @patch("claude_code_proxy.cli.commands.api.create_docker_adapter")
    def test_api_docker_with_volumes(
        self, mock_docker_adapter, mock_config_manager, mock_settings, tmp_path
    ):
        """Test API command with Docker volumes."""
        mock_config_manager.load_settings.return_value = mock_settings
        mock_config_manager.get_cli_overrides_from_args.return_value = {}
        mock_config_manager.setup_logging.return_value = None
        mock_adapter = MagicMock()
        mock_docker_adapter.return_value = mock_adapter

        # Create temporary directories for volumes
        host_data = tmp_path / "data"
        host_config = tmp_path / "config"
        host_data.mkdir()
        host_config.mkdir()

        # Create a test app with the api command
        from typer import Typer

        app = Typer()
        app.command()(api)

        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "--docker",
                "--docker-volume",
                f"{host_data}:/container/data",
                "--docker-volume",
                f"{host_config}:/container/config:ro",
            ],
        )

        if result.exit_code != 0:
            print(f"Exit code: {result.exit_code}")
            print(f"Output: {result.output}")
        assert result.exit_code == 0
        # Verify volumes were passed to Docker adapter
        call_args = mock_adapter.exec_container.call_args
        volumes = call_args.kwargs["volumes"]
        assert len(volumes) >= 2  # At least our two volumes
        # Check that our volumes are in the list
        assert any(v[1] == "/container/data" for v in volumes)
        assert any(v[1] == "/container/config" for v in volumes)

    @patch("claude_code_proxy.cli.commands.api.config_manager")
    @patch("claude_code_proxy.cli.commands.api.create_docker_adapter")
    def test_api_docker_with_env_vars(
        self, mock_docker_adapter, mock_config_manager, mock_settings
    ):
        """Test API command with Docker environment variables."""
        mock_config_manager.load_settings.return_value = mock_settings
        mock_config_manager.get_cli_overrides_from_args.return_value = {}
        mock_config_manager.setup_logging.return_value = None
        mock_adapter = MagicMock()
        mock_docker_adapter.return_value = mock_adapter

        # Create a test app with the api command
        from typer import Typer

        app = Typer()
        app.command()(api)

        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "--docker",
                "--docker-env",
                "API_KEY=secret123",
                "--docker-env",
                "DEBUG=true",
            ],
        )

        assert result.exit_code == 0
        # Verify environment variables were passed to Docker adapter
        call_args = mock_adapter.exec_container.call_args
        environment = call_args.kwargs["environment"]
        assert "PORT" in environment  # Default env var
        assert "HOST" in environment  # Default env var
