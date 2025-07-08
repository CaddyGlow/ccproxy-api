"""Tests for ccproxy/commands/config.py module."""

import json
import secrets
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, Mock, patch

import pytest
import typer
from click import Context
from rich.console import Console
from typer.testing import CliRunner

from ccproxy.cli.commands.config import app
from ccproxy.cli.commands.config.commands import (
    _detect_config_format,
    _write_config_file,
    _write_json_config,
    _write_toml_config,
    _write_yaml_config,
    config_init,
    config_list,
    generate_token,
    get_config_path_from_context,
)
from ccproxy.cli.commands.config.schema_commands import (
    config_schema,
    config_validate,
)


@pytest.mark.unit
class TestGetConfigPathFromContext:
    """Test get_config_path_from_context function."""

    def test_get_config_path_from_context_with_valid_context(self):
        """Test getting config path from valid typer context."""
        with patch(
            "ccproxy.cli.commands.config.commands.get_current_context"
        ) as mock_ctx:
            mock_context = Mock()
            mock_context.obj = {"config_path": "/path/to/config.toml"}
            mock_ctx.return_value = mock_context

            result = get_config_path_from_context()

            assert result == Path("/path/to/config.toml")

    def test_get_config_path_from_context_with_none_config_path(self):
        """Test getting config path when config_path is None."""
        with patch(
            "ccproxy.cli.commands.config.commands.get_current_context"
        ) as mock_ctx:
            mock_context = Mock()
            mock_context.obj = {"config_path": None}
            mock_ctx.return_value = mock_context

            result = get_config_path_from_context()

            assert result is None

    def test_get_config_path_from_context_no_obj(self):
        """Test getting config path when context has no obj."""
        with patch(
            "ccproxy.cli.commands.config.commands.get_current_context"
        ) as mock_ctx:
            mock_context = Mock()
            mock_context.obj = None
            mock_ctx.return_value = mock_context

            result = get_config_path_from_context()

            assert result is None

    def test_get_config_path_from_context_no_config_path_key(self):
        """Test getting config path when obj doesn't have config_path key."""
        with patch(
            "ccproxy.cli.commands.config.commands.get_current_context"
        ) as mock_ctx:
            mock_context = Mock()
            mock_context.obj = {"other_key": "value"}
            mock_ctx.return_value = mock_context

            result = get_config_path_from_context()

            assert result is None

    def test_get_config_path_from_context_no_context(self):
        """Test getting config path when no context is available."""
        with patch(
            "ccproxy.cli.commands.config.commands.get_current_context"
        ) as mock_ctx:
            mock_ctx.return_value = None

            result = get_config_path_from_context()

            assert result is None

    def test_get_config_path_from_context_runtime_error(self):
        """Test getting config path when RuntimeError is raised (no active context)."""
        with patch(
            "ccproxy.cli.commands.config.commands.get_current_context"
        ) as mock_ctx:
            mock_ctx.side_effect = RuntimeError("No active click context")

            result = get_config_path_from_context()

            assert result is None


@pytest.mark.unit
class TestConfigListCommand:
    """Test config_list command."""

    def setup_method(self):
        """Setup test environment."""
        self.runner = CliRunner(env={"NO_COLOR": "1", "TERM": "dumb"})

    def test_config_list_success(self):
        """Test successful config list command execution."""
        mock_settings = Mock()
        mock_settings.host = "127.0.0.1"
        mock_settings.port = 8000
        mock_settings.log_level = "INFO"
        mock_settings.workers = 4
        mock_settings.reload = False
        mock_settings.server_url = "http://127.0.0.1:8000"
        mock_settings.claude_cli_path = "/usr/bin/claude"
        mock_settings.auth_token = "test-token"
        mock_settings.cors_origins = ["*"]
        mock_settings.api_tools_handling = "warning"

        # Mock docker settings
        mock_docker = Mock()
        mock_docker.docker_image = "claude-code-proxy"
        mock_docker.docker_home_directory = None
        mock_docker.docker_workspace_directory = None
        mock_docker.docker_volumes = []
        mock_docker.docker_environment = {}
        mock_docker.docker_additional_args = []
        mock_docker.user_mapping_enabled = True
        mock_docker.user_uid = 1000
        mock_docker.user_gid = 1000
        mock_settings.docker_settings = mock_docker

        with (
            patch(
                "ccproxy.cli.commands.config.commands.get_settings",
                return_value=mock_settings,
            ),
            patch(
                "ccproxy.cli.commands.config.commands.get_config_path_from_context",
                return_value=None,
            ),
        ):
            result = self.runner.invoke(app, ["list"])

            assert result.exit_code == 0
            assert "Server Configuration" in result.stdout
            assert "127.0.0.1" in result.stdout
            assert "8000" in result.stdout

    def test_config_list_with_config_path(self):
        """Test config list command with config path from context."""
        mock_settings = Mock()
        mock_settings.host = "localhost"
        mock_settings.port = 9000
        mock_settings.log_level = "DEBUG"
        mock_settings.workers = 2
        mock_settings.reload = True
        mock_settings.server_url = "http://localhost:9000"
        mock_settings.claude_cli_path = None
        mock_settings.auth_token = None
        mock_settings.cors_origins = ["https://example.com"]
        mock_settings.api_tools_handling = "error"

        mock_docker = Mock()
        mock_docker.docker_image = "custom-image"
        mock_docker.docker_home_directory = "/custom/home"
        mock_docker.docker_workspace_directory = "/custom/workspace"
        mock_docker.docker_volumes = ["vol1:vol2"]
        mock_docker.docker_environment = {"ENV": "test"}
        mock_docker.docker_additional_args = ["--arg"]
        mock_docker.user_mapping_enabled = False
        mock_docker.user_uid = None
        mock_docker.user_gid = None
        mock_settings.docker_settings = mock_docker

        config_path = Path("/custom/config.toml")

        with (
            patch(
                "ccproxy.cli.commands.config.commands.get_settings",
                return_value=mock_settings,
            ) as mock_get_settings,
            patch(
                "ccproxy.cli.commands.config.commands.get_config_path_from_context",
                return_value=config_path,
            ),
        ):
            result = self.runner.invoke(app, ["list"])

            assert result.exit_code == 0
            mock_get_settings.assert_called_once_with(config_path=config_path)

    def test_config_list_exception_handling(self):
        """Test config list command with exception in get_settings."""
        with patch(
            "ccproxy.cli.commands.config.commands.get_settings"
        ) as mock_get_settings:
            mock_get_settings.side_effect = ValueError("Test error")

            result = self.runner.invoke(app, ["list"])

            assert result.exit_code == 1
            assert "Error loading configuration: Test error" in result.stdout


@pytest.mark.unit
class TestConfigInitCommand:
    """Test config_init command."""

    def setup_method(self):
        """Setup test environment."""
        self.runner = CliRunner(env={"NO_COLOR": "1", "TERM": "dumb"})

    def test_config_init_toml_default(self):
        """Test config init with default TOML format."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)

            with patch(
                "ccproxy.utils.xdg.get_ccproxy_config_dir",
                return_value=output_dir,
            ):
                result = self.runner.invoke(app, ["init"])

                assert result.exit_code == 0
                assert "Created example configuration file:" in result.stdout

                config_file = output_dir / "config.toml"
                assert config_file.exists()

                content = config_file.read_text()
                assert 'host = "127.0.0.1"' in content
                assert "port = 8000" in content
                assert "[docker_settings]" in content

    def test_config_init_json_format(self):
        """Test config init with JSON format."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)

            with patch(
                "ccproxy.utils.xdg.get_ccproxy_config_dir",
                return_value=output_dir,
            ):
                result = self.runner.invoke(app, ["init", "--format", "json"])

                assert result.exit_code == 0

                config_file = output_dir / "config.json"
                assert config_file.exists()

                with config_file.open() as f:
                    config_data = json.load(f)
                    assert config_data["host"] == "127.0.0.1"
                    assert config_data["port"] == 8000

    def test_config_init_yaml_format(self):
        """Test config init with YAML format."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)

            with patch(
                "ccproxy.utils.xdg.get_ccproxy_config_dir",
                return_value=output_dir,
            ):
                result = self.runner.invoke(app, ["init", "--format", "yaml"])

                assert result.exit_code == 0

                config_file = output_dir / "config.yaml"
                assert config_file.exists()

    def test_config_init_yaml_format_no_yaml_module(self):
        """Test config init with YAML format when PyYAML is not available."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)

            with (
                patch(
                    "ccproxy.utils.xdg.get_ccproxy_config_dir",
                    return_value=output_dir,
                ),
                patch(
                    "ccproxy.cli.commands.config.commands.yaml",
                    None,
                    create=True,
                ),
            ):
                # Mock import error
                import builtins

                original_import = builtins.__import__

                def mock_import(name, *args, **kwargs):
                    if name == "yaml":
                        raise ImportError("No module named 'yaml'")
                    return original_import(name, *args, **kwargs)

                with patch("builtins.__import__", side_effect=mock_import):
                    result = self.runner.invoke(app, ["init", "--format", "yaml"])

                    assert result.exit_code == 1
                    assert "YAML support is not available" in result.stdout

    def test_config_init_invalid_format(self):
        """Test config init with invalid format."""
        result = self.runner.invoke(app, ["init", "--format", "invalid"])

        assert result.exit_code == 1
        assert "Invalid format 'invalid'" in result.stdout

    def test_config_init_custom_output_dir(self):
        """Test config init with custom output directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "custom"

            result = self.runner.invoke(app, ["init", "--output-dir", str(output_dir)])

            assert result.exit_code == 0
            assert output_dir.exists()

            config_file = output_dir / "config.toml"
            assert config_file.exists()

    def test_config_init_file_exists_no_force(self):
        """Test config init when file exists and --force is not used."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            config_file = output_dir / "config.toml"
            config_file.write_text("existing content")

            with patch(
                "ccproxy.utils.xdg.get_ccproxy_config_dir",
                return_value=output_dir,
            ):
                result = self.runner.invoke(app, ["init"])

                assert result.exit_code == 1
                assert "already exists" in result.stdout

    def test_config_init_file_exists_with_force(self):
        """Test config init when file exists and --force is used."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            config_file = output_dir / "config.toml"
            config_file.write_text("existing content")

            with patch(
                "ccproxy.utils.xdg.get_ccproxy_config_dir",
                return_value=output_dir,
            ):
                result = self.runner.invoke(app, ["init", "--force"])

                assert result.exit_code == 0
                assert "Created example configuration file:" in result.stdout

    def test_config_init_exception_handling(self):
        """Test config init with exception during execution."""
        with patch("ccproxy.utils.xdg.get_ccproxy_config_dir") as mock_get_dir:
            mock_get_dir.side_effect = PermissionError("Permission denied")

            result = self.runner.invoke(app, ["init"])

            assert result.exit_code == 1
            assert "Error creating configuration file" in result.stdout


@pytest.mark.unit
class TestConfigSchemaCommand:
    """Test config_schema command."""

    def setup_method(self):
        """Setup test environment."""
        self.runner = CliRunner(env={"NO_COLOR": "1", "TERM": "dumb"})

    def test_config_schema_default(self):
        """Test config schema command with default options."""
        mock_files = [Path("schema1.json"), Path("schema2.json")]

        with patch(
            "ccproxy.utils.schema.generate_schema_files",
            return_value=mock_files,
        ):
            result = self.runner.invoke(app, ["schema"])

            assert result.exit_code == 0
            assert "Generating JSON Schema files" in result.stdout
            assert "ccproxy-schema.json" in result.stdout

    def test_config_schema_custom_output_dir(self):
        """Test config schema command with custom output directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            mock_files = [output_dir / "schema.json"]

            with patch(
                "ccproxy.cli.commands.config.schema_commands.generate_schema_files",
                return_value=mock_files,
            ) as mock_gen:
                result = self.runner.invoke(
                    app, ["schema", "--output-dir", str(output_dir)]
                )

                assert result.exit_code == 0
                mock_gen.assert_called_once_with(output_dir)

    def test_config_schema_with_taplo(self):
        """Test config schema command with taplo option."""
        mock_files = [Path("schema.json")]
        mock_taplo_config = Path("taplo.toml")

        with (
            patch(
                "ccproxy.cli.commands.config.schema_commands.generate_schema_files",
                return_value=mock_files,
            ),
            patch(
                "ccproxy.cli.commands.config.schema_commands.generate_taplo_config",
                return_value=mock_taplo_config,
            ) as mock_taplo,
        ):
            result = self.runner.invoke(app, ["schema", "--taplo"])

            assert result.exit_code == 0
            assert "Generating taplo configuration" in result.stdout
            assert "Generated: taplo.toml" in result.stdout
            mock_taplo.assert_called_once()

    def test_config_schema_exception_handling(self):
        """Test config schema command with exception."""
        with patch(
            "ccproxy.cli.commands.config.schema_commands.generate_schema_files"
        ) as mock_gen:
            mock_gen.side_effect = ValueError("Schema generation failed")

            result = self.runner.invoke(app, ["schema"])

            assert result.exit_code == 1
            assert "Error generating schema" in result.stdout


@pytest.mark.unit
class TestConfigValidateCommand:
    """Test config_validate command."""

    def setup_method(self):
        """Setup test environment."""
        self.runner = CliRunner(env={"NO_COLOR": "1", "TERM": "dumb"})

    def test_config_validate_success(self):
        """Test config validate command with valid file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write("host = 'test'\nport = 8000")
            f.flush()

            try:
                with patch(
                    "ccproxy.cli.commands.config.schema_commands.validate_config_with_schema",
                    return_value=True,
                ):
                    result = self.runner.invoke(app, ["validate", f.name])

                    assert result.exit_code == 0
                    assert "Configuration file is valid" in result.stdout
            finally:
                Path(f.name).unlink()

    def test_config_validate_invalid_file(self):
        """Test config validate command with invalid file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write("host = 'test'\nport = 8000")
            f.flush()

            try:
                with patch(
                    "ccproxy.cli.commands.config.schema_commands.validate_config_with_schema",
                    return_value=False,
                ):
                    result = self.runner.invoke(app, ["validate", f.name])

                    assert result.exit_code == 1
                    assert "Configuration file validation failed" in result.stdout
            finally:
                Path(f.name).unlink()

    def test_config_validate_nonexistent_file(self):
        """Test config validate command with nonexistent file."""
        result = self.runner.invoke(app, ["validate", "/nonexistent/file.toml"])

        assert result.exit_code == 1
        assert "File /nonexistent/file.toml does not exist" in result.stdout

    def test_config_validate_import_error(self):
        """Test config validate command with import error."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write("host = 'test'")
            f.flush()

            try:
                with patch(
                    "ccproxy.cli.commands.config.schema_commands.validate_config_with_schema"
                ) as mock_validate:
                    mock_validate.side_effect = ImportError(
                        "check-jsonschema not available"
                    )

                    result = self.runner.invoke(app, ["validate", f.name])

                    assert result.exit_code == 1
                    assert "Install check-jsonschema" in result.stdout
            finally:
                Path(f.name).unlink()

    def test_config_validate_validation_exception(self):
        """Test config validate command with validation exception."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write("host = 'test'")
            f.flush()

            try:
                with patch(
                    "ccproxy.cli.commands.config.schema_commands.validate_config_with_schema"
                ) as mock_validate:
                    mock_validate.side_effect = ValueError("Validation error occurred")

                    result = self.runner.invoke(app, ["validate", f.name])

                    assert result.exit_code == 1
                    assert (
                        "Validation error: Validation error occurred" in result.stdout
                    )
            finally:
                Path(f.name).unlink()

    def test_config_validate_general_exception(self):
        """Test config validate command with general exception."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write("host = 'test'")
            f.flush()

            try:
                # Mock the validate_config_with_schema to raise a general exception
                with patch(
                    "ccproxy.cli.commands.config.schema_commands.validate_config_with_schema"
                ) as mock_validate:
                    mock_validate.side_effect = OSError("File system error")

                    result = self.runner.invoke(app, ["validate", f.name])

                    assert result.exit_code == 1
                    assert "Error validating configuration" in result.stdout
            finally:
                Path(f.name).unlink()


@pytest.mark.unit
class TestGenerateTokenCommand:
    """Test generate_token command."""

    def setup_method(self):
        """Setup test environment."""
        self.runner = CliRunner(env={"NO_COLOR": "1", "TERM": "dumb"})

    def test_generate_token_display_only(self):
        """Test generate token command without saving."""
        with patch(
            "ccproxy.cli.commands.config.commands.secrets.token_urlsafe",
            return_value="test-token-123",
        ):
            result = self.runner.invoke(app, ["generate-token"])

            assert result.exit_code == 0
            assert "Generated Authentication Token" in result.stdout
            assert "test-token-123" in result.stdout
            assert "Server Environment Variables" in result.stdout
            assert "Client Environment Variables" in result.stdout
            assert "export AUTH_TOKEN=test-token-123" in result.stdout
            assert "export ANTHROPIC_API_KEY=test-token-123" in result.stdout
            assert "export OPENAI_API_KEY=test-token-123" in result.stdout

    def test_generate_token_save_new_file(self):
        """Test generate token command with save to new file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = Path(temp_dir) / ".ccproxy.toml"

            with (
                patch(
                    "ccproxy.cli.commands.config.commands.secrets.token_urlsafe",
                    return_value="save-token-456",
                ),
                patch("ccproxy.utils.find_toml_config_file", return_value=None),
            ):
                result = self.runner.invoke(
                    app, ["generate-token", "--save", "--config-file", str(config_file)]
                )

                assert result.exit_code == 0
                assert "Token saved to" in result.stdout
                assert config_file.exists()

                content = config_file.read_text()
                assert "save-token-456" in content

    def test_generate_token_save_existing_file_no_force(self):
        """Test generate token command with save to existing file without force."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = Path(temp_dir) / "config.toml"
            config_file.write_text('auth_token = "existing-token"')

            mock_settings = Mock()
            mock_load_config = {"auth_token": "existing-token"}

            with (
                patch(
                    "ccproxy.cli.commands.config.commands.secrets.token_urlsafe",
                    return_value="new-token-789",
                ),
                patch(
                    "ccproxy.config.settings.Settings.load_config_file",
                    return_value=mock_load_config,
                ),
            ):
                # Use input to simulate user saying "n" (no) to confirmation
                result = self.runner.invoke(
                    app,
                    ["generate-token", "--save", "--config-file", str(config_file)],
                    input="n\n",
                )

                assert result.exit_code == 0
                assert "Token generation cancelled" in result.stdout

    def test_generate_token_save_existing_file_with_force(self):
        """Test generate token command with save to existing file with force."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = Path(temp_dir) / "config.toml"
            config_file.write_text('auth_token = "existing-token"')

            mock_load_config = {"auth_token": "existing-token"}

            with (
                patch(
                    "ccproxy.cli.commands.config.commands.secrets.token_urlsafe",
                    return_value="force-token-xyz",
                ),
                patch(
                    "ccproxy.config.settings.Settings.load_config_file",
                    return_value=mock_load_config,
                ),
            ):
                result = self.runner.invoke(
                    app,
                    [
                        "generate-token",
                        "--save",
                        "--config-file",
                        str(config_file),
                        "--force",
                    ],
                )

                assert result.exit_code == 0
                assert "Token saved to" in result.stdout

                content = config_file.read_text()
                assert "force-token-xyz" in content

    def test_generate_token_save_existing_file_confirm_yes(self):
        """Test generate token command with save to existing file and user confirms."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = Path(temp_dir) / "config.toml"
            config_file.write_text('auth_token = "existing-token"')

            mock_load_config = {"auth_token": "existing-token"}

            with (
                patch(
                    "ccproxy.cli.commands.config.commands.secrets.token_urlsafe",
                    return_value="confirm-token-abc",
                ),
                patch(
                    "ccproxy.config.settings.Settings.load_config_file",
                    return_value=mock_load_config,
                ),
            ):
                result = self.runner.invoke(
                    app,
                    ["generate-token", "--save", "--config-file", str(config_file)],
                    input="y\n",
                )

                assert result.exit_code == 0
                assert "Token saved to" in result.stdout

    def test_generate_token_save_auto_detect_config(self):
        """Test generate token command with save using auto-detected config."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = Path(temp_dir) / "existing.toml"

            with (
                patch(
                    "ccproxy.cli.commands.config.commands.secrets.token_urlsafe",
                    return_value="auto-token-def",
                ),
                patch(
                    "ccproxy.utils.find_toml_config_file",
                    return_value=config_file,
                ),
                patch(
                    "ccproxy.config.settings.Settings.load_config_file",
                    return_value={},
                ),
            ):
                result = self.runner.invoke(app, ["generate-token", "--save"])

                assert result.exit_code == 0
                assert "Token saved to" in result.stdout

    def test_generate_token_save_auto_detect_no_config(self):
        """Test generate token command with save when no config is auto-detected."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Change to temp directory to ensure clean environment
            import os

            original_cwd = Path.cwd()
            os.chdir(temp_dir)

            try:
                expected_file = Path(".ccproxy.toml")

                with (
                    patch(
                        "ccproxy.cli.commands.config.commands.secrets.token_urlsafe",
                        return_value="default-token-ghi",
                    ),
                    patch(
                        "ccproxy.utils.find_toml_config_file",
                        return_value=None,
                    ),
                ):
                    result = self.runner.invoke(app, ["generate-token", "--save"])

                    assert result.exit_code == 0
                    assert "Token saved to" in result.stdout
            finally:
                os.chdir(original_cwd)

    def test_generate_token_save_json_format(self):
        """Test generate token command with save to JSON format."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = Path(temp_dir) / "config.json"

            with (
                patch(
                    "ccproxy.cli.commands.config.commands.secrets.token_urlsafe",
                    return_value="json-token-jkl",
                ),
                patch(
                    "ccproxy.config.settings.Settings.load_config_file",
                    return_value={},
                ),
            ):
                result = self.runner.invoke(
                    app, ["generate-token", "--save", "--config-file", str(config_file)]
                )

                assert result.exit_code == 0
                assert config_file.exists()

                with config_file.open() as f:
                    data = json.load(f)
                    assert data["auth_token"] == "json-token-jkl"

    def test_generate_token_save_yaml_format(self):
        """Test generate token command with save to YAML format."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = Path(temp_dir) / "config.yaml"

            with (
                patch(
                    "ccproxy.cli.commands.config.commands.secrets.token_urlsafe",
                    return_value="yaml-token-mno",
                ),
                patch(
                    "ccproxy.config.settings.Settings.load_config_file",
                    return_value={},
                ),
            ):
                result = self.runner.invoke(
                    app, ["generate-token", "--save", "--config-file", str(config_file)]
                )

                assert result.exit_code == 0
                assert config_file.exists()

    def test_generate_token_save_read_config_exception(self):
        """Test generate token command with exception reading existing config."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = Path(temp_dir) / "config.toml"
            config_file.write_text("invalid content")

            with (
                patch(
                    "ccproxy.cli.commands.config.commands.secrets.token_urlsafe",
                    return_value="exception-token",
                ),
                patch("ccproxy.config.settings.Settings.load_config_file") as mock_load,
            ):
                mock_load.side_effect = ValueError("Config read error")

                result = self.runner.invoke(
                    app, ["generate-token", "--save", "--config-file", str(config_file)]
                )

                assert result.exit_code == 0
                assert "Warning: Could not read existing config file" in result.stdout
                assert "Will create new configuration file" in result.stdout

    def test_generate_token_exception_handling(self):
        """Test generate token command with general exception."""
        with patch(
            "ccproxy.cli.commands.config.commands.secrets.token_urlsafe"
        ) as mock_token:
            mock_token.side_effect = OSError("Random generation failed")

            result = self.runner.invoke(app, ["generate-token"])

            assert result.exit_code == 1
            assert "Error generating token" in result.stdout


@pytest.mark.unit
class TestDetectConfigFormat:
    """Test _detect_config_format function."""

    def test_detect_toml_format(self):
        """Test detecting TOML format."""
        assert _detect_config_format(Path("config.toml")) == "toml"

    def test_detect_json_format(self):
        """Test detecting JSON format."""
        assert _detect_config_format(Path("config.json")) == "json"

    def test_detect_yaml_format(self):
        """Test detecting YAML format."""
        assert _detect_config_format(Path("config.yaml")) == "yaml"
        assert _detect_config_format(Path("config.yml")) == "yaml"

    def test_detect_unknown_format_defaults_to_toml(self):
        """Test detecting unknown format defaults to TOML."""
        assert _detect_config_format(Path("config.xml")) == "toml"
        assert _detect_config_format(Path("config")) == "toml"

    def test_detect_format_case_insensitive(self):
        """Test format detection is case insensitive."""
        assert _detect_config_format(Path("config.TOML")) == "toml"
        assert _detect_config_format(Path("config.JSON")) == "json"
        assert _detect_config_format(Path("config.YAML")) == "yaml"


@pytest.mark.unit
class TestWriteConfigFunctions:
    """Test config file writing functions."""

    def test_write_json_config(self):
        """Test writing JSON config file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            config_file = Path(f.name)

        try:
            config_data = {"host": "127.0.0.1", "port": 8000, "debug": True}
            _write_json_config(config_file, config_data)

            assert config_file.exists()

            with config_file.open() as f:
                data = json.load(f)
                assert data == config_data
        finally:
            config_file.unlink()

    def test_write_yaml_config(self):
        """Test writing YAML config file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            config_file = Path(f.name)

        try:
            config_data = {"host": "127.0.0.1", "port": 8000}
            _write_yaml_config(config_file, config_data)

            assert config_file.exists()
            content = config_file.read_text()
            assert "Claude Code Proxy API Configuration" in content
            assert "host: 127.0.0.1" in content
        finally:
            config_file.unlink()

    def test_write_yaml_config_no_yaml_module(self):
        """Test writing YAML config when PyYAML is not available."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            config_file = Path(f.name)

        try:
            import builtins

            original_import = builtins.__import__

            def mock_import(name, *args, **kwargs):
                if name == "yaml":
                    raise ImportError("No module named 'yaml'")
                return original_import(name, *args, **kwargs)

            with patch("builtins.__import__", side_effect=mock_import):
                config_data = {"host": "127.0.0.1"}

                with pytest.raises(ValueError, match="YAML support not available"):
                    _write_yaml_config(config_file, config_data)
        finally:
            if config_file.exists():
                config_file.unlink()

    def test_write_toml_config_basic(self):
        """Test writing basic TOML config file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            config_file = Path(f.name)

        try:
            config_data = {"auth_token": "test-token"}
            _write_toml_config(config_file, config_data)

            assert config_file.exists()
            content = config_file.read_text()
            assert "Claude Code Proxy API Configuration" in content
            assert 'auth_token = "test-token"' in content
        finally:
            config_file.unlink()

    def test_write_toml_config_comprehensive(self):
        """Test writing comprehensive TOML config file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            config_file = Path(f.name)

        try:
            config_data = {
                "host": "localhost",
                "port": 9000,
                "log_level": "DEBUG",
                "workers": 2,
                "reload": True,
                "auth_token": "test-token",
                "cors_origins": ["https://example.com", "https://test.com"],
                "tools_handling": "error",
                "claude_cli_path": "/usr/bin/claude",
                "docker_settings": {
                    "docker_image": "custom-image",
                    "docker_volumes": ["vol1:vol2", "vol3:vol4"],
                    "docker_environment": {"ENV1": "value1", "ENV2": "value2"},
                    "docker_additional_args": ["--arg1", "--arg2"],
                    "user_mapping_enabled": False,
                    "user_uid": 1001,
                    "user_gid": 1002,
                },
                "extra_setting": "extra_value",
            }
            _write_toml_config(config_file, config_data)

            assert config_file.exists()
            content = config_file.read_text()

            # Check server settings
            assert 'host = "localhost"' in content
            assert "port = 9000" in content
            assert 'log_level = "DEBUG"' in content
            assert "workers = 2" in content
            assert "reload = true" in content

            # Check security settings
            assert 'auth_token = "test-token"' in content
            assert (
                'cors_origins = ["https://example.com", "https://test.com"]' in content
            )
            assert 'tools_handling = "error"' in content

            # Check Claude CLI settings
            assert 'claude_cli_path = "/usr/bin/claude"' in content

            # Check Docker settings
            assert "[docker_settings]" in content
            assert 'docker_image = "custom-image"' in content
            assert "user_mapping_enabled = false" in content
            assert "user_uid = 1001" in content
            assert "user_gid = 1002" in content

            # Check additional settings
            assert 'extra_setting = "extra_value"' in content
        finally:
            config_file.unlink()

    def test_write_toml_config_empty_lists_and_dicts(self):
        """Test writing TOML config with empty lists and dicts."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            config_file = Path(f.name)

        try:
            config_data: dict[str, Any] = {
                "docker_settings": {
                    "docker_volumes": [],
                    "docker_environment": {},
                    "docker_additional_args": [],
                }
            }
            _write_toml_config(config_file, config_data)

            assert config_file.exists()
            content = config_file.read_text()
            assert "docker_volumes = []" in content
            assert "docker_environment = {}" in content
            assert "docker_additional_args = []" in content
        finally:
            config_file.unlink()

    def test_write_toml_config_none_values(self):
        """Test writing TOML config with None values."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            config_file = Path(f.name)

        try:
            config_data = {
                "claude_cli_path": None,
                "docker_settings": {
                    "docker_home_directory": None,
                    "user_uid": None,
                },
            }
            _write_toml_config(config_file, config_data)

            assert config_file.exists()
            content = config_file.read_text()
            assert "# claude_cli_path = " in content
            assert "# docker_home_directory = null" in content
            assert "# user_uid = null" in content
        finally:
            config_file.unlink()

    def test_write_toml_config_exception_handling(self):
        """Test TOML config writing with exception."""
        with patch("builtins.open") as mock_open:
            mock_open.side_effect = PermissionError("Permission denied")

            config_file = Path("/invalid/path/config.toml")
            config_data = {"test": "value"}

            with pytest.raises(ValueError, match="Failed to write TOML configuration"):
                _write_toml_config(config_file, config_data)

    def test_write_config_file_toml(self):
        """Test generic write_config_file function with TOML format."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            config_file = Path(f.name)

        try:
            config_data = {"host": "test"}
            _write_config_file(config_file, config_data, "toml")

            assert config_file.exists()
            content = config_file.read_text()
            assert 'host = "test"' in content
        finally:
            config_file.unlink()

    def test_write_config_file_json(self):
        """Test generic write_config_file function with JSON format."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            config_file = Path(f.name)

        try:
            config_data = {"host": "test"}
            _write_config_file(config_file, config_data, "json")

            assert config_file.exists()
            with config_file.open() as f:
                data = json.load(f)
                assert data["host"] == "test"
        finally:
            config_file.unlink()

    def test_write_config_file_yaml(self):
        """Test generic write_config_file function with YAML format."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            config_file = Path(f.name)

        try:
            config_data = {"host": "test"}
            _write_config_file(config_file, config_data, "yaml")

            assert config_file.exists()
        finally:
            config_file.unlink()

    def test_write_config_file_unsupported_format(self):
        """Test generic write_config_file function with unsupported format."""
        config_file = Path("config.xml")
        config_data = {"host": "test"}

        with pytest.raises(ValueError, match="Unsupported config format: xml"):
            _write_config_file(config_file, config_data, "xml")


@pytest.mark.unit
class TestConfigApp:
    """Test the config typer app configuration."""

    def test_app_configuration(self):
        """Test that the config app is properly configured."""
        assert app.info.name == "config"
        assert (
            app.info.help is not None
            and "Configuration management commands" in app.info.help
        )
        assert app.rich_markup_mode == "rich"
        assert app._add_completion is True
        assert app.info.no_args_is_help is True

    def test_app_commands_registered(self):
        """Test that all commands are registered with the app."""
        # Get registered commands from the app
        commands = app.registered_commands
        command_names = [cmd.name for cmd in commands]

        expected_commands = ["list", "init", "schema", "validate", "generate-token"]
        for expected_cmd in expected_commands:
            assert expected_cmd in command_names


@pytest.mark.integration
class TestConfigCommandsIntegration:
    """Integration tests for config commands."""

    def setup_method(self):
        """Setup test environment."""
        self.runner = CliRunner(env={"NO_COLOR": "1", "TERM": "dumb"})

    def test_config_workflow_init_validate_list(self):
        """Test complete workflow: init -> validate -> list."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = Path(temp_dir) / "test-config.toml"

            # Step 1: Initialize config file
            result = self.runner.invoke(app, ["init", "--output-dir", temp_dir])
            assert result.exit_code == 0

            created_file = Path(temp_dir) / "config.toml"
            assert created_file.exists()

            # Step 2: Validate the created config file
            with patch(
                "ccproxy.cli.commands.config.schema_commands.validate_config_with_schema",
                return_value=True,
            ):
                result = self.runner.invoke(app, ["validate", str(created_file)])
                assert result.exit_code == 0
                assert "Configuration file is valid" in result.stdout

            # Step 3: List configuration (mocked to avoid dependency issues)
            mock_settings = Mock()
            mock_settings.host = "127.0.0.1"
            mock_settings.port = 8000
            mock_settings.log_level = "INFO"
            mock_settings.workers = 4
            mock_settings.reload = False
            mock_settings.server_url = "http://127.0.0.1:8000"
            mock_settings.claude_cli_path = None
            mock_settings.auth_token = None
            mock_settings.cors_origins = ["*"]
            mock_settings.api_tools_handling = "warning"

            mock_docker = Mock()
            mock_docker.docker_image = "claude-code-proxy"
            mock_docker.docker_home_directory = None
            mock_docker.docker_workspace_directory = None
            mock_docker.docker_volumes = []
            mock_docker.docker_environment = {}
            mock_docker.docker_additional_args = []
            mock_docker.user_mapping_enabled = True
            mock_docker.user_uid = 1000
            mock_docker.user_gid = 1000
            mock_settings.docker_settings = mock_docker

            with (
                patch(
                    "ccproxy.cli.commands.config.commands.get_settings",
                    return_value=mock_settings,
                ),
                patch(
                    "ccproxy.cli.commands.config.commands.get_config_path_from_context",
                    return_value=None,
                ),
            ):
                result = self.runner.invoke(app, ["list"])
                assert result.exit_code == 0
                assert "Server Configuration" in result.stdout

    def test_config_token_generation_and_save(self):
        """Test token generation and saving workflow."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = Path(temp_dir) / "auth-config.toml"

            with (
                patch(
                    "ccproxy.cli.commands.config.commands.secrets.token_urlsafe",
                    return_value="integration-test-token",
                ),
                patch("ccproxy.utils.find_toml_config_file", return_value=None),
            ):
                # Generate and save token
                result = self.runner.invoke(
                    app, ["generate-token", "--save", "--config-file", str(config_file)]
                )

                assert result.exit_code == 0
                assert "Token saved to" in result.stdout
                assert config_file.exists()

                # Verify token is in the file
                content = config_file.read_text()
                assert "integration-test-token" in content
                assert "Security configuration" in content
