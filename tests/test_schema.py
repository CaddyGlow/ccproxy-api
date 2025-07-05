"""Test JSON Schema generation and validation functionality."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from claude_code_proxy.utils.schema import (
    generate_json_schema,
    generate_schema_files,
    generate_taplo_config,
    save_schema_file,
    validate_toml_with_schema,
)


@pytest.mark.unit
class TestSchemaGeneration:
    """Test JSON Schema generation functionality."""

    def test_generate_json_schema(self):
        """Test generating JSON Schema from Settings model."""
        schema = generate_json_schema()

        # Check basic schema structure
        assert isinstance(schema, dict)
        assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
        assert schema["title"] == "Claude Code Proxy API Configuration"
        assert "properties" in schema

        # Check key properties exist
        properties = schema["properties"]
        assert "host" in properties
        assert "port" in properties
        assert "log_level" in properties
        assert "docker_settings" in properties

        # Check examples are added
        assert "examples" in properties["host"]
        assert "examples" in properties["port"]
        assert "examples" in properties["log_level"]

    def test_save_schema_file(self):
        """Test saving schema to file."""
        schema = {"test": "schema"}

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test-schema.json"
            save_schema_file(schema, output_path)

            assert output_path.exists()

            with output_path.open() as f:
                loaded_schema = json.load(f)

            assert loaded_schema == schema

    def test_generate_schema_files(self):
        """Test generating multiple schema files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)

            generated_files = generate_schema_files(output_dir)

            assert len(generated_files) == 2

            # Check main schema file
            main_schema = output_dir / "ccproxy-schema.json"
            assert main_schema in generated_files
            assert main_schema.exists()

            # Check current directory schema file
            current_schema = output_dir / ".ccproxy-schema.json"
            assert current_schema in generated_files
            assert current_schema.exists()

            # Verify schema content
            with main_schema.open() as f:
                schema = json.load(f)
            assert schema["title"] == "Claude Code Proxy API Configuration"

    def test_generate_taplo_config(self):
        """Test generating taplo configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)

            config_path = generate_taplo_config(output_dir)

            assert config_path == output_dir / ".taplo.toml"
            assert config_path.exists()

            content = config_path.read_text()
            assert "ccproxy-schema.json" in content
            assert "ccproxy.toml" in content
            assert ".ccproxy.toml" in content
            assert "snake_case" in content

    def test_generate_schema_files_default_directory(self):
        """Test generating schema files in default directory."""
        with (
            patch("claude_code_proxy.utils.schema.Path.cwd") as mock_cwd,
            tempfile.TemporaryDirectory() as temp_dir,
        ):
            mock_cwd.return_value = Path(temp_dir)

            generated_files = generate_schema_files()

            # Should use current directory
            assert len(generated_files) == 2
            assert all(f.parent == Path(temp_dir) for f in generated_files)


@pytest.mark.unit
class TestSchemaValidation:
    """Test TOML validation against JSON Schema."""

    def test_validate_valid_toml(self):
        """Test validation of valid TOML file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write("""
            host = "127.0.0.1"
            port = 8080
            log_level = "DEBUG"
            """)
            f.flush()

            try:
                is_valid = validate_toml_with_schema(Path(f.name))
                assert is_valid is True
            except ImportError:
                # jsonschema not installed, skip test
                pytest.skip("jsonschema package not available")
            finally:
                Path(f.name).unlink()

    def test_validate_invalid_toml(self):
        """Test validation of invalid TOML file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write("""
            host = "127.0.0.1"
            port = "invalid-port"  # Should be integer
            """)
            f.flush()

            try:
                is_valid = validate_toml_with_schema(Path(f.name))
                assert is_valid is False
            except ImportError:
                # jsonschema not installed, skip test
                pytest.skip("jsonschema package not available")
            finally:
                Path(f.name).unlink()

    def test_validate_with_explicit_schema(self):
        """Test validation with explicit schema file."""
        # Create a simple schema
        schema = {
            "type": "object",
            "properties": {"host": {"type": "string"}, "port": {"type": "integer"}},
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            # Save schema
            schema_path = Path(temp_dir) / "schema.json"
            with schema_path.open("w") as f:
                json.dump(schema, f)

            # Create TOML file
            toml_path = Path(temp_dir) / "config.toml"
            toml_path.write_text('host = "localhost"\nport = 8080')

            try:
                is_valid = validate_toml_with_schema(toml_path, schema_path)
                assert is_valid is True
            except ImportError:
                # jsonschema not installed, skip test
                pytest.skip("jsonschema package not available")

    def test_validate_nonexistent_file(self):
        """Test validation of non-existent file."""
        nonexistent_path = Path("/nonexistent/file.toml")

        try:
            # Should raise an exception due to file not existing
            with pytest.raises((FileNotFoundError, OSError)):
                validate_toml_with_schema(nonexistent_path)
        except ImportError:
            # jsonschema not installed, skip test
            pytest.skip("jsonschema package not available")
