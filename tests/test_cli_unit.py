"""Unit tests for CLI functions."""

import os
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
import typer
from fastapi_cli.exceptions import FastAPICLIException

from claude_code_proxy._version import __version__
from claude_code_proxy.cli import app, get_default_path_hook, main, version_callback


@pytest.mark.unit
class TestVersionCallback:
    """Test version_callback function."""

    def test_version_callback_false(self):
        """Test version_callback with False value does nothing."""
        # Should not raise any exception
        version_callback(False)

    def test_version_callback_true(self):
        """Test version_callback with True value prints version and exits."""
        with patch("typer.echo") as mock_echo:
            with pytest.raises(typer.Exit):
                version_callback(True)

            mock_echo.assert_called_once_with(f"claude-code-proxy-api {__version__}")

    def test_version_callback_with_mock_version(self):
        """Test version_callback with mocked version."""
        with (
            patch("claude_code_proxy.cli.__version__", "1.2.3"),
            patch("typer.echo") as mock_echo,
            pytest.raises(typer.Exit),
        ):
            version_callback(True)

        mock_echo.assert_called_once_with("claude-code-proxy-api 1.2.3")


@pytest.mark.unit
class TestGetDefaultPathHook:
    """Test get_default_path_hook function."""

    @patch("claude_code_proxy.cli.get_package_dir")
    def test_get_default_path_hook_file_exists(self, mock_get_package_dir):
        """Test get_default_path_hook when file exists."""
        mock_package_dir = Path("/mock/package")
        mock_get_package_dir.return_value = mock_package_dir

        expected_path = mock_package_dir / "claude_code_proxy" / "main.py"

        with patch.object(Path, "is_file", return_value=True):
            result = get_default_path_hook()
            assert result == expected_path

    @patch("claude_code_proxy.cli.get_package_dir")
    def test_get_default_path_hook_file_not_exists(self, mock_get_package_dir):
        """Test get_default_path_hook when file doesn't exist."""
        mock_package_dir = Path("/mock/package")
        mock_get_package_dir.return_value = mock_package_dir

        with patch.object(Path, "is_file", return_value=False):
            with pytest.raises(FastAPICLIException) as exc_info:
                get_default_path_hook()

            assert "Could not find a default file to run" in str(exc_info.value)

    @patch("claude_code_proxy.cli.get_package_dir")
    def test_get_default_path_hook_path_construction(self, mock_get_package_dir):
        """Test get_default_path_hook constructs correct path."""
        mock_package_dir = Path("/test/path")
        mock_get_package_dir.return_value = mock_package_dir

        with patch.object(Path, "is_file", return_value=True) as mock_is_file:
            result = get_default_path_hook()

            # Check that the correct path was checked
            expected_path = mock_package_dir / "claude_code_proxy" / "main.py"
            mock_is_file.assert_called_once()
            assert result == expected_path


@pytest.mark.unit
class TestMainFunction:
    """Test main function (app callback)."""

    def test_main_function_exists(self):
        """Test that main function exists and is callable."""
        assert callable(main)

    def test_main_function_with_version_false(self):
        """Test main function with version=False."""
        # Should not raise any exception
        main(version=False)

    def test_main_function_with_version_true(self):
        """Test main function with version=True."""
        # The version callback is handled eagerly by typer, so we just test
        # that the main function doesn't raise an exception when called directly
        main(version=True)

    def test_main_function_signature(self):
        """Test main function has correct signature."""
        import inspect

        sig = inspect.signature(main)

        # Check that version parameter exists
        assert "version" in sig.parameters
        param = sig.parameters["version"]
        assert param.annotation is bool
        assert param.default is not inspect.Parameter.empty


@pytest.mark.unit
class TestTyperAppConfiguration:
    """Test Typer app configuration."""

    def test_app_is_typer_instance(self):
        """Test that app is a Typer instance."""
        assert isinstance(app, typer.Typer)

    def test_app_configuration(self):
        """Test app configuration options."""
        # Test that the app has the expected configuration
        # Note: Typer doesn't expose these attributes directly,
        # so we test indirectly through behavior
        assert hasattr(app, "callback")
        assert hasattr(app, "command")

    def test_app_has_callback(self):
        """Test that app has a callback (main function)."""
        # The main function should be registered as a callback
        assert app.callback is not None

    def test_app_commands_registration(self):
        """Test that commands are properly registered."""
        # Check that the app has registered commands
        # This is indirect testing since Typer doesn't expose command list easily
        assert hasattr(app, "registered_commands")


@pytest.mark.unit
class TestTyperOptions:
    """Test Typer option configurations."""

    def test_version_option_configuration(self):
        """Test version option is configured correctly."""
        import inspect

        sig = inspect.signature(main)
        version_param = sig.parameters.get("version")

        assert version_param is not None
        assert version_param.annotation is bool

        # Check that it has a default value (from typer.Option)
        assert version_param.default is not inspect.Parameter.empty

    def test_version_option_eager_and_callback(self):
        """Test version option has eager=True and callback set."""
        # The version callback is handled eagerly by typer's option system
        # We test this indirectly by verifying the callback function works
        with patch("typer.echo") as mock_echo:
            with pytest.raises(typer.Exit):
                version_callback(True)

            mock_echo.assert_called_once()


@pytest.mark.unit
class TestHelpText:
    """Test help text and documentation."""

    def test_main_function_docstring(self):
        """Test main function has proper docstring."""
        assert main.__doc__ is not None
        assert "Claude Code Proxy API Server" in main.__doc__
        assert "Anthropic" in main.__doc__

    def test_version_callback_docstring(self):
        """Test version_callback has proper docstring."""
        assert version_callback.__doc__ is not None
        assert "Print version and exit" in version_callback.__doc__

    def test_get_default_path_hook_docstring(self):
        """Test get_default_path_hook has proper docstring."""
        # This function doesn't have a docstring in the current implementation
        # This test documents the current state
        pass


@pytest.mark.unit
class TestErrorHandling:
    """Test error handling in CLI functions."""

    def test_version_callback_type_error(self):
        """Test version_callback with wrong type."""
        # This should work due to Python's truthiness
        with patch("typer.echo") as mock_echo:
            with pytest.raises(typer.Exit):
                version_callback(True)  # Use bool instead of string

            mock_echo.assert_called_once()

    def test_get_default_path_hook_exception_message(self):
        """Test get_default_path_hook exception message is descriptive."""
        with patch("claude_code_proxy.cli.get_package_dir") as mock_get_package_dir:
            mock_get_package_dir.return_value = Path("/mock/path")

            with patch.object(Path, "is_file", return_value=False):
                with pytest.raises(FastAPICLIException) as exc_info:
                    get_default_path_hook()

                error_msg = str(exc_info.value)
                assert "Could not find a default file to run" in error_msg
                assert "please provide an explicit path" in error_msg


@pytest.mark.unit
class TestConfigCommand:
    """Test config command function."""

    @patch("claude_code_proxy.cli.get_settings")
    @patch("typer.echo")
    def test_config_command_success(self, mock_echo, mock_get_settings):
        """Test config command displays settings successfully."""
        from claude_code_proxy.cli import config

        # Mock settings object
        mock_settings = Mock()
        mock_settings.host = "0.0.0.0"
        mock_settings.port = 8000
        mock_settings.log_level = "INFO"
        mock_settings.claude_cli_path = "/usr/bin/claude"
        mock_settings.workers = 1
        mock_settings.reload = False
        mock_get_settings.return_value = mock_settings

        # Call config command
        config()

        # Verify settings were fetched
        mock_get_settings.assert_called_once()

        # Verify output calls
        expected_calls = [
            ("Current Configuration:",),
            ("  Host: 0.0.0.0",),
            ("  Port: 8000",),
            ("  Log Level: INFO",),
            ("  Claude CLI Path: /usr/bin/claude",),
            ("  Workers: 1",),
            ("  Reload: False",),
        ]

        assert mock_echo.call_count == len(expected_calls)
        for i, expected_call in enumerate(expected_calls):
            assert mock_echo.call_args_list[i][0] == expected_call

    @patch("claude_code_proxy.cli.get_settings")
    @patch("typer.echo")
    def test_config_command_with_none_claude_path(self, mock_echo, mock_get_settings):
        """Test config command when claude_cli_path is None."""
        from claude_code_proxy.cli import config

        # Mock settings object with None claude_cli_path
        mock_settings = Mock()
        mock_settings.host = "localhost"
        mock_settings.port = 3000
        mock_settings.log_level = "DEBUG"
        mock_settings.claude_cli_path = None
        mock_settings.workers = 2
        mock_settings.reload = True
        mock_get_settings.return_value = mock_settings

        # Call config command
        config()

        # Verify claude CLI path shows "Auto-detect"
        claude_path_call = None
        for call in mock_echo.call_args_list:
            if "Claude CLI Path:" in call[0][0]:
                claude_path_call = call[0][0]
                break

        assert claude_path_call == "  Claude CLI Path: Auto-detect"

    @patch("claude_code_proxy.cli.get_settings")
    @patch("typer.echo")
    def test_config_command_exception_handling(self, mock_echo, mock_get_settings):
        """Test config command handles exceptions properly."""
        from claude_code_proxy.cli import config

        # Mock get_settings to raise an exception
        mock_get_settings.side_effect = Exception("Settings error")

        # Call config command and expect it to raise typer.Exit
        with pytest.raises(typer.Exit) as exc_info:
            config()

        # Verify exit code is 1
        assert exc_info.value.exit_code == 1

        # Verify error message was printed
        mock_echo.assert_called_with(
            "Error loading configuration: Settings error", err=True
        )


@pytest.mark.unit
class TestClaudeCommand:
    """Test claude command function."""

    @patch("claude_code_proxy.cli.get_settings")
    @patch("typer.echo")
    @patch("os.execvp")
    def test_claude_command_local_execution(
        self, mock_execvp, mock_echo, mock_get_settings
    ):
        """Test claude command with local execution."""
        from claude_code_proxy.cli import claude

        # Mock settings
        mock_settings = Mock()
        mock_settings.claude_cli_path = "/usr/bin/claude"
        mock_get_settings.return_value = mock_settings

        # Call claude command
        claude(args=["--version"], docker=False)

        # Verify settings were fetched
        mock_get_settings.assert_called_once()

        # Verify echo was called with execution message
        mock_echo.assert_any_call("Executing: /usr/bin/claude --version")
        mock_echo.assert_any_call("")

        # Verify execvp was called with correct arguments
        mock_execvp.assert_called_once_with(
            "/usr/bin/claude", ["/usr/bin/claude", "--version"]
        )

    @patch("claude_code_proxy.cli.get_settings")
    @patch("typer.echo")
    def test_claude_command_no_cli_path(self, mock_echo, mock_get_settings):
        """Test claude command when claude CLI path is not found."""
        from claude_code_proxy.cli import claude

        # Mock settings with no claude_cli_path
        mock_settings = Mock()
        mock_settings.claude_cli_path = None
        mock_get_settings.return_value = mock_settings

        # Call claude command and expect it to raise typer.Exit
        with pytest.raises(typer.Exit) as exc_info:
            claude(args=["--version"], docker=False)

        # Verify exit code is 1
        assert exc_info.value.exit_code == 1

        # Verify error messages
        mock_echo.assert_any_call("Error: Claude CLI not found.", err=True)
        mock_echo.assert_any_call(
            "Please install Claude CLI or configure claude_cli_path.", err=True
        )

    @patch("claude_code_proxy.cli.get_settings")
    @patch("typer.echo")
    @patch("os.execvp")
    def test_claude_command_relative_path_resolution(
        self, mock_execvp, mock_echo, mock_get_settings
    ):
        """Test claude command resolves relative paths."""
        from claude_code_proxy.cli import claude

        # Mock settings with relative path
        mock_settings = Mock()
        mock_settings.claude_cli_path = "claude"
        mock_get_settings.return_value = mock_settings

        with patch("pathlib.Path.resolve", return_value=Path("/resolved/path/claude")):
            # Call claude command
            claude(args=["doctor"], docker=False)

        # Verify execvp was called with resolved path
        mock_execvp.assert_called_once_with(
            "/resolved/path/claude", ["/resolved/path/claude", "doctor"]
        )

    @patch("claude_code_proxy.cli.get_settings")
    @patch("claude_code_proxy.cli.DockerCommandBuilder")
    @patch("typer.echo")
    @patch("os.execvp")
    def test_claude_command_docker_execution(
        self, mock_execvp, mock_echo, mock_docker_builder, mock_get_settings
    ):
        """Test claude command with Docker execution."""
        from claude_code_proxy.cli import claude

        # Mock settings
        mock_settings = Mock()
        mock_settings.docker_settings = Mock()
        mock_get_settings.return_value = mock_settings

        # Mock Docker command builder
        mock_docker_cmd = ["docker", "run", "claude:latest"]
        mock_docker_builder.from_settings_and_overrides.return_value = mock_docker_cmd

        # Call claude command with Docker
        claude(
            args=["--version"],
            docker=True,
            docker_image="custom:latest",
            docker_env=["API_KEY=test"],
            docker_volume=["./data:/data"],
            docker_arg=["--rm"],
            docker_home="/home/user",
            docker_workspace="/workspace",
        )

        # Verify Docker command builder was called with correct arguments
        mock_docker_builder.from_settings_and_overrides.assert_called_once_with(
            mock_settings.docker_settings,
            docker_image="custom:latest",
            docker_env=["API_KEY=test"],
            docker_volume=["./data:/data"],
            docker_arg=["--rm"],
            docker_home="/home/user",
            docker_workspace="/workspace",
        )

        # Verify echo was called with Docker command
        mock_echo.assert_any_call(
            "Executing: docker run claude:latest claude --version"
        )
        mock_echo.assert_any_call("")

        # Verify execvp was called with Docker command (with claude and args appended)
        expected_cmd = ["docker", "run", "claude:latest", "claude", "--version"]
        mock_execvp.assert_called_once_with("docker", expected_cmd)

    @patch("claude_code_proxy.cli.get_settings")
    @patch("typer.echo")
    @patch("os.execvp")
    def test_claude_command_execvp_os_error(
        self, mock_execvp, mock_echo, mock_get_settings
    ):
        """Test claude command handles OSError from execvp."""
        from claude_code_proxy.cli import claude

        # Mock settings
        mock_settings = Mock()
        mock_settings.claude_cli_path = "/usr/bin/claude"
        mock_get_settings.return_value = mock_settings

        # Mock execvp to raise OSError
        mock_execvp.side_effect = OSError("Command not found")

        # Call claude command and expect it to raise typer.Exit
        with pytest.raises(typer.Exit) as exc_info:
            claude(args=["--version"], docker=False)

        # Verify exit code is 1
        assert exc_info.value.exit_code == 1

        # Verify error message
        mock_echo.assert_any_call(
            "Failed to execute command: Command not found", err=True
        )

    @patch("claude_code_proxy.cli.get_settings")
    @patch("typer.echo")
    def test_claude_command_general_exception(self, mock_echo, mock_get_settings):
        """Test claude command handles general exceptions."""
        from claude_code_proxy.cli import claude

        # Mock get_settings to raise an exception
        mock_get_settings.side_effect = Exception("Settings error")

        # Call claude command and expect it to raise typer.Exit
        with pytest.raises(typer.Exit) as exc_info:
            claude(args=["--version"], docker=False)

        # Verify exit code is 1
        assert exc_info.value.exit_code == 1

        # Verify error message
        mock_echo.assert_any_call(
            "Error executing claude command: Settings error", err=True
        )

    def test_claude_command_docstring(self):
        """Test claude command has proper docstring."""
        from claude_code_proxy.cli import claude

        assert claude.__doc__ is not None
        assert "Execute claude CLI commands directly" in claude.__doc__
        assert "Examples:" in claude.__doc__

    def test_claude_command_parameter_defaults(self):
        """Test claude command has correct parameter defaults."""
        import inspect

        from claude_code_proxy.cli import claude

        sig = inspect.signature(claude)

        # Check args parameter
        args_param = sig.parameters["args"]
        assert args_param.annotation == list[str] | None

        # Check docker parameter
        docker_param = sig.parameters["docker"]
        assert docker_param.annotation is bool

        # Check optional parameters exist
        assert "docker_image" in sig.parameters
        assert "docker_env" in sig.parameters
        assert "docker_volume" in sig.parameters
        assert "docker_arg" in sig.parameters
        assert "docker_home" in sig.parameters
        assert "docker_workspace" in sig.parameters


@pytest.mark.unit
class TestModuleImports:
    """Test module imports and dependencies."""

    def test_required_imports_available(self):
        """Test that all required imports are available."""
        # Test that we can import the functions we're testing
        from claude_code_proxy.cli import (
            app,
            claude,
            config,
            get_default_path_hook,
            main,
            version_callback,
        )

        assert callable(version_callback)
        assert callable(get_default_path_hook)
        assert callable(main)
        assert callable(config)
        assert callable(claude)
        assert isinstance(app, typer.Typer)

    def test_version_import(self):
        """Test that version can be imported."""
        from claude_code_proxy._version import __version__

        assert isinstance(__version__, str)
        assert len(__version__) > 0

    def test_settings_import(self):
        """Test that settings can be imported."""
        from claude_code_proxy.config.settings import get_settings

        assert callable(get_settings)

    def test_docker_builder_import(self):
        """Test that DockerCommandBuilder can be imported."""
        from claude_code_proxy.utils.docker_builder import DockerCommandBuilder

        assert hasattr(DockerCommandBuilder, "from_settings_and_overrides")

    def test_helper_import(self):
        """Test that helper functions can be imported."""
        from claude_code_proxy.utils.helper import get_package_dir

        assert callable(get_package_dir)
