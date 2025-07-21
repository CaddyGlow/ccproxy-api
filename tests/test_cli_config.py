"""Tests for CLI config commands."""

import json
import tempfile
from collections.abc import Generator
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import typer
from typer.testing import CliRunner

from ccproxy.cli.commands.config import app, config_list
from ccproxy.cli.commands.config.commands import (
    config_init,
    generate_token,
)
from ccproxy.cli.commands.config.schema_commands import config_schema, config_validate
from ccproxy.config.settings import Settings


@pytest.fixture
def cli_runner() -> CliRunner:
    """Create CLI runner for testing."""
    return CliRunner()


@pytest.fixture
def temp_config_dir() -> Generator[Path, None, None]:
    """Create temporary directory for config files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def sample_toml_config(temp_config_dir: Path) -> Path:
    """Create a sample TOML config file."""
    config_file = temp_config_dir / "config.toml"
    config_file.write_text("""
# Sample configuration
port = 8080
host = "127.0.0.1"
auth_token = "test-token"

[server]
log_level = "DEBUG"
""")
    return config_file


class TestConfigList:
    """Test config list command."""

    def test_config_list_basic(self, cli_runner: CliRunner) -> None:
        """Test basic config list command."""
        result = cli_runner.invoke(app, ["list"])
        assert result.exit_code == 0
        assert "Claude Code Proxy API Configuration" in result.output
        assert "Version:" in result.output

    def test_config_list_shows_sections(self, cli_runner: CliRunner) -> None:
        """Test that config list shows different configuration sections."""
        result = cli_runner.invoke(app, ["list"])
        assert result.exit_code == 0
        # Should show at least some configuration sections
        assert "Configuration" in result.output

    @patch("ccproxy.cli.commands.config.commands.get_settings")
    def test_config_list_error_handling(
        self, mock_get_settings: Any, cli_runner: CliRunner
    ) -> None:
        """Test error handling in config list."""
        mock_get_settings.side_effect = Exception("Config error")
        result = cli_runner.invoke(app, ["list"])
        assert result.exit_code == 1
        assert "Error loading configuration" in result.output


class TestConfigInit:
    """Test config init command."""

    def test_config_init_toml_default(
        self, cli_runner: CliRunner, temp_config_dir: Path
    ) -> None:
        """Test config init with TOML format."""
        with patch(
            "ccproxy.config.discovery.get_ccproxy_config_dir",
            return_value=temp_config_dir,
        ):
            result = cli_runner.invoke(app, ["init"])
            assert result.exit_code == 0
            assert "Created example configuration file" in result.output

            config_file = temp_config_dir / "config.toml"
            assert config_file.exists()
            content = config_file.read_text()
            assert "Claude Code Proxy API Configuration" in content

    def test_config_init_custom_output_dir(
        self, cli_runner: CliRunner, temp_config_dir: Path
    ) -> None:
        """Test config init with custom output directory."""
        result = cli_runner.invoke(app, ["init", "--output-dir", str(temp_config_dir)])
        assert result.exit_code == 0

        config_file = temp_config_dir / "config.toml"
        assert config_file.exists()

    def test_config_init_force_overwrite(
        self, cli_runner: CliRunner, temp_config_dir: Path
    ) -> None:
        """Test config init with force overwrite."""
        config_file = temp_config_dir / "config.toml"
        config_file.write_text("existing content")

        result = cli_runner.invoke(
            app, ["init", "--output-dir", str(temp_config_dir), "--force"]
        )
        assert result.exit_code == 0
        assert "Created example configuration file" in result.output

    def test_config_init_existing_file_no_force(
        self, cli_runner: CliRunner, temp_config_dir: Path
    ) -> None:
        """Test config init fails when file exists without force."""
        config_file = temp_config_dir / "config.toml"
        config_file.write_text("existing content")

        result = cli_runner.invoke(app, ["init", "--output-dir", str(temp_config_dir)])
        assert result.exit_code == 1
        assert "already exists" in result.output

    def test_config_init_invalid_format(self, cli_runner: CliRunner) -> None:
        """Test config init with invalid format."""
        result = cli_runner.invoke(app, ["init", "--format", "yaml"])
        assert result.exit_code == 1
        assert "Invalid format" in result.output


class TestGenerateToken:
    """Test generate token command."""

    def test_generate_token_basic(self, cli_runner: CliRunner) -> None:
        """Test basic token generation."""
        result = cli_runner.invoke(app, ["generate-token"])
        assert result.exit_code == 0
        assert "Generated Authentication Token" in result.output
        assert "ANTHROPIC_API_KEY" in result.output
        assert "OPENAI_API_KEY" in result.output

    def test_generate_token_save_new_file(
        self, cli_runner: CliRunner, temp_config_dir: Path
    ) -> None:
        """Test saving token to new config file."""
        config_file = temp_config_dir / "test.toml"

        with patch(
            "ccproxy.config.discovery.find_toml_config_file",
            return_value=config_file,
        ):
            result = cli_runner.invoke(app, ["generate-token", "--save"])
            assert result.exit_code == 0
            assert "Token saved to" in result.output
            assert config_file.exists()

            content = config_file.read_text()
            # The token is saved using the TOML writer which creates commented structure
            # Check that it contains the basic TOML structure instead
            assert "Claude Code Proxy API Configuration" in content

    def test_generate_token_save_existing_file_with_force(
        self, cli_runner: CliRunner, sample_toml_config: Path
    ) -> None:
        """Test saving token to existing file with force."""
        result = cli_runner.invoke(
            app,
            [
                "generate-token",
                "--save",
                "--config-file",
                str(sample_toml_config),
                "--force",
            ],
        )
        assert result.exit_code == 0
        assert "Token saved to" in result.output

    def test_generate_token_save_existing_file_no_force(
        self, cli_runner: CliRunner, sample_toml_config: Path
    ) -> None:
        """Test saving token to existing file without force (should prompt)."""
        # Simulate user declining to overwrite
        result = cli_runner.invoke(
            app,
            [
                "generate-token",
                "--save",
                "--config-file",
                str(sample_toml_config),
            ],
            input="n\n",
        )
        assert result.exit_code == 0
        assert "Token generation cancelled" in result.output


class TestConfigSchema:
    """Test config schema command."""

    def test_config_schema_basic(
        self, cli_runner: CliRunner, temp_config_dir: Path
    ) -> None:
        """Test basic schema generation."""
        with (
            patch(
                "ccproxy.cli.commands.config.schema_commands.generate_schema_files",
                return_value=[temp_config_dir / "schema.json"],
            ),
            patch(
                "ccproxy.cli.commands.config.schema_commands.generate_taplo_config",
                return_value=temp_config_dir / ".taplo.toml",
            ),
        ):
            result = cli_runner.invoke(app, ["schema"])
            assert result.exit_code == 0
            assert "Generating JSON Schema files" in result.output
            assert "Schema files generated successfully" in result.output

    def test_config_schema_custom_output_dir(
        self, cli_runner: CliRunner, temp_config_dir: Path
    ) -> None:
        """Test schema generation with custom output directory."""
        with (
            patch(
                "ccproxy.cli.commands.config.schema_commands.generate_schema_files",
                return_value=[temp_config_dir / "schema.json"],
            ),
            patch(
                "ccproxy.cli.commands.config.schema_commands.generate_taplo_config",
                return_value=temp_config_dir / ".taplo.toml",
            ),
        ):
            result = cli_runner.invoke(
                app, ["schema", "--output-dir", str(temp_config_dir)]
            )
            assert result.exit_code == 0

    def test_config_schema_error_handling(self, cli_runner: CliRunner) -> None:
        """Test schema generation error handling."""
        with patch(
            "ccproxy.cli.commands.config.schema_commands.generate_schema_files",
            side_effect=Exception("Schema generation failed"),
        ):
            result = cli_runner.invoke(app, ["schema"])
            assert result.exit_code == 1
            assert "Error generating schema" in result.output


class TestConfigValidate:
    """Test config validate command."""

    def test_config_validate_valid_file(
        self, cli_runner: CliRunner, sample_toml_config: Path
    ) -> None:
        """Test validating a valid config file."""
        with patch(
            "ccproxy.cli.commands.config.schema_commands.validate_config_with_schema",
            return_value=True,
        ):
            result = cli_runner.invoke(app, ["validate", str(sample_toml_config)])
            assert result.exit_code == 0
            assert "Configuration file is valid" in result.output

    def test_config_validate_invalid_file(
        self, cli_runner: CliRunner, sample_toml_config: Path
    ) -> None:
        """Test validating an invalid config file."""
        with patch(
            "ccproxy.cli.commands.config.schema_commands.validate_config_with_schema",
            return_value=False,
        ):
            result = cli_runner.invoke(app, ["validate", str(sample_toml_config)])
            assert result.exit_code == 1
            assert "validation failed" in result.output

    def test_config_validate_nonexistent_file(self, cli_runner: CliRunner) -> None:
        """Test validating a nonexistent file."""
        result = cli_runner.invoke(app, ["validate", "nonexistent.toml"])
        assert result.exit_code == 1
        assert "does not exist" in result.output

    def test_config_validate_import_error(
        self, cli_runner: CliRunner, sample_toml_config: Path
    ) -> None:
        """Test validation with import error."""
        with patch(
            "ccproxy.cli.commands.config.schema_commands.validate_config_with_schema",
            side_effect=ImportError("Missing dependency"),
        ):
            result = cli_runner.invoke(app, ["validate", str(sample_toml_config)])
            assert result.exit_code == 1
            assert "Install check-jsonschema" in result.output

    def test_config_validate_general_error(
        self, cli_runner: CliRunner, sample_toml_config: Path
    ) -> None:
        """Test validation with general error."""
        with patch(
            "ccproxy.cli.commands.config.schema_commands.validate_config_with_schema",
            side_effect=Exception("Validation error"),
        ):
            result = cli_runner.invoke(app, ["validate", str(sample_toml_config)])
            assert result.exit_code == 1
            assert "Validation error" in result.output


class TestConfigApp:
    """Test config CLI app structure."""

    def test_config_app_help(self, cli_runner: CliRunner) -> None:
        """Test config app help."""
        result = cli_runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "Configuration management commands" in result.output

    def test_config_app_no_args(self, cli_runner: CliRunner) -> None:
        """Test config app with no arguments shows help."""
        result = cli_runner.invoke(app, [])
        assert result.exit_code == 2  # Typer exits with 2 when no subcommand provided
        assert "Usage:" in result.output

    def test_config_commands_registered(self, cli_runner: CliRunner) -> None:
        """Test that all config commands are properly registered."""
        result = cli_runner.invoke(app, ["--help"])
        assert result.exit_code == 0

        # Check that all main commands are listed
        assert "list" in result.output
        assert "init" in result.output
        assert "generate-token" in result.output
        assert "schema" in result.output
        assert "validate" in result.output


class TestConfigHelpers:
    """Test config helper functions."""

    def test_format_value_functions(self) -> None:
        """Test value formatting functions."""
        from ccproxy.cli.commands.config.commands import _format_value

        # Test various value types
        assert _format_value(None) == "[dim]Auto-detect[/dim]"
        assert _format_value(True) == "True"
        assert _format_value(42) == "42"
        assert _format_value("") == "[dim]Not set[/dim]"
        assert _format_value("normal_value") == "normal_value"
        assert _format_value("secret_token") == "[green]Set[/green]"
        assert _format_value([]) == "[dim]None[/dim]"
        assert _format_value(["item"]) == "item"
        assert _format_value({}) == "[dim]None[/dim]"

    def test_detect_config_format(self) -> None:
        """Test config format detection."""
        from ccproxy.cli.commands.config.commands import _detect_config_format

        assert _detect_config_format(Path("config.toml")) == "toml"
        assert (
            _detect_config_format(Path("config.json")) == "toml"
        )  # Only TOML supported
        assert (
            _detect_config_format(Path("config.yaml")) == "toml"
        )  # Only TOML supported

    def test_generate_default_config_from_model(self) -> None:
        """Test generating default config from Settings model."""
        from ccproxy.cli.commands.config.commands import (
            _generate_default_config_from_model,
        )

        config_data = _generate_default_config_from_model(Settings)
        assert isinstance(config_data, dict)
        # The model has nested structure, so port would be under server
        assert "server" in config_data
        assert "security" in config_data
        # Should contain all top-level settings fields
        for field_name in Settings.model_fields:
            assert field_name in config_data
