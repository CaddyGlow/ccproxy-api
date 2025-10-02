"""Tests for environment variable generator utilities."""

import sys
from io import StringIO
from pathlib import Path

import pytest
from pydantic import BaseModel, Field

from ccproxy.config.env_generator import (
    format_value_for_env,
    generate_env_config,
    generate_env_vars_from_model,
    get_field_description,
    is_hidden_in_example,
    write_env_config,
)


# Test Models
class SimpleModel(BaseModel):
    """Simple test model."""

    name: str = Field(default="test", description="Name field")
    count: int = Field(default=42, description="Count field")
    enabled: bool = Field(default=True, description="Enabled flag")


class NestedModel(BaseModel):
    """Model with nested fields."""

    host: str = Field(default="localhost", description="Host address")
    port: int = Field(default=8000, description="Port number")


class ComplexModel(BaseModel):
    """Complex model with nested models."""

    simple: SimpleModel = Field(
        default_factory=SimpleModel, description="Simple config"
    )
    nested: NestedModel = Field(
        default_factory=NestedModel, description="Nested config"
    )
    tags: list[str] = Field(
        default_factory=lambda: ["tag1", "tag2"], description="Tags"
    )


class HiddenFieldModel(BaseModel):
    """Model with hidden fields."""

    visible: str = Field(default="visible", description="Visible field")
    hidden: str = Field(
        default="hidden",
        description="Hidden field",
        json_schema_extra={"config_example_hidden": True},
    )


@pytest.mark.unit
def test_is_hidden_in_example() -> None:
    """Test field hiding detection."""
    field_info = HiddenFieldModel.model_fields["visible"]
    assert not is_hidden_in_example(field_info)

    hidden_field_info = HiddenFieldModel.model_fields["hidden"]
    assert is_hidden_in_example(hidden_field_info)


@pytest.mark.unit
def test_get_field_description() -> None:
    """Test field description extraction."""
    field_info = SimpleModel.model_fields["name"]
    assert get_field_description(field_info) == "Name field"


@pytest.mark.unit
def test_format_value_for_env() -> None:
    """Test environment variable value formatting."""
    assert format_value_for_env(None) == ""
    assert format_value_for_env(True) == "true"
    assert format_value_for_env(False) == "false"
    assert format_value_for_env(42) == "42"
    assert format_value_for_env(3.14) == "3.14"
    assert format_value_for_env("hello") == "hello"
    assert format_value_for_env([1, 2, 3]) == "[1, 2, 3]"
    assert format_value_for_env(["a", "b"]) == '["a", "b"]'
    assert format_value_for_env({"key": "value"}) == '{"key": "value"}'


@pytest.mark.unit
def test_generate_env_vars_from_model() -> None:
    """Test env var generation from model."""
    env_vars = generate_env_vars_from_model(SimpleModel)

    # Should have 3 variables
    assert len(env_vars) == 3

    # Check names and values
    env_dict = {name: value for name, value, _ in env_vars}
    assert env_dict["NAME"] == "test"
    assert env_dict["COUNT"] == 42
    assert env_dict["ENABLED"] is True


@pytest.mark.unit
def test_generate_env_vars_with_prefix() -> None:
    """Test env var generation with prefix."""
    env_vars = generate_env_vars_from_model(SimpleModel, prefix="APP")

    env_dict = {name: value for name, value, _ in env_vars}
    assert "APP__NAME" in env_dict
    assert "APP__COUNT" in env_dict
    assert "APP__ENABLED" in env_dict


@pytest.mark.unit
def test_generate_env_vars_nested() -> None:
    """Test env var generation for nested models."""
    env_vars = generate_env_vars_from_model(ComplexModel)

    env_names = [name for name, _, _ in env_vars]

    # Should have nested paths
    assert "SIMPLE__NAME" in env_names
    assert "SIMPLE__COUNT" in env_names
    assert "NESTED__HOST" in env_names
    assert "NESTED__PORT" in env_names
    assert "TAGS" in env_names


@pytest.mark.unit
def test_generate_env_vars_with_hidden() -> None:
    """Test env var generation includes/excludes hidden fields."""
    # Without hidden fields (default)
    env_vars = generate_env_vars_from_model(HiddenFieldModel, include_hidden=False)
    env_names = [name for name, _, _ in env_vars]
    assert "VISIBLE" in env_names
    assert "HIDDEN" not in env_names

    # With hidden fields
    env_vars = generate_env_vars_from_model(HiddenFieldModel, include_hidden=True)
    env_names = [name for name, _, _ in env_vars]
    assert "VISIBLE" in env_names
    assert "HIDDEN" in env_names


@pytest.mark.unit
def test_generate_env_config_simple() -> None:
    """Test env config generation for simple model."""
    config = generate_env_config(SimpleModel, commented=True)

    assert "# Environment Variable Configuration" in config
    assert "# export NAME=" in config
    assert "# export COUNT=" in config
    assert "# export ENABLED=" in config


@pytest.mark.unit
def test_generate_env_config_uncommented() -> None:
    """Test env config generation without comments."""
    config = generate_env_config(SimpleModel, commented=False)

    assert "export NAME=test" in config
    assert "export COUNT=42" in config
    assert "export ENABLED=true" in config
    assert "# export" not in config


@pytest.mark.unit
def test_generate_env_config_with_prefix() -> None:
    """Test env config generation with prefix."""
    config = generate_env_config(SimpleModel, prefix="APP", commented=False)

    assert "export APP__NAME=test" in config
    assert "export APP__COUNT=42" in config
    assert "export APP__ENABLED=true" in config


@pytest.mark.unit
def test_generate_env_config_no_export() -> None:
    """Test env config generation without export prefix."""
    config = generate_env_config(SimpleModel, commented=False, export_format=False)

    assert "NAME=test" in config
    assert "COUNT=42" in config
    assert "ENABLED=true" in config
    assert "export" not in config


@pytest.mark.unit
def test_generate_env_config_custom_header() -> None:
    """Test env config generation with custom header."""
    custom_header = "Custom Header\nLine 2"
    config = generate_env_config(SimpleModel, header_comment=custom_header)

    assert "# Custom Header" in config
    assert "# Line 2" in config


@pytest.mark.unit
def test_write_env_config_to_stringio() -> None:
    """Test writing env config to StringIO."""
    buffer = StringIO()
    write_env_config(buffer, SimpleModel, commented=False)

    result = buffer.getvalue()
    assert "export NAME=test" in result
    assert "export COUNT=42" in result


@pytest.mark.unit
def test_write_env_config_to_file(tmp_path: Path) -> None:
    """Test writing env config to file."""
    output_file = tmp_path / "test.env"
    write_env_config(output_file, SimpleModel, commented=False)

    assert output_file.exists()
    content = output_file.read_text()
    assert "export NAME=test" in content


@pytest.mark.unit
def test_write_env_config_to_stdout(capsys: pytest.CaptureFixture[str]) -> None:
    """Test writing env config to stdout."""
    write_env_config(sys.stdout, SimpleModel, commented=False)

    captured = capsys.readouterr()
    assert "export NAME=test" in captured.out


@pytest.mark.unit
def test_write_env_config_with_prefix() -> None:
    """Test writing env config with prefix."""
    buffer = StringIO()
    write_env_config(buffer, SimpleModel, prefix="PLUGINS__TEST", commented=False)

    result = buffer.getvalue()
    assert "export PLUGINS__TEST__NAME=test" in result


@pytest.mark.unit
def test_write_env_config_with_string_path(tmp_path: Path) -> None:
    """Test writing env config with string path."""
    output_file = str(tmp_path / "test.env")
    write_env_config(output_file, SimpleModel, commented=False)

    assert Path(output_file).exists()


@pytest.mark.unit
def test_path_field_converted_to_string() -> None:
    """Test that Path fields are converted to strings."""

    class PathModel(BaseModel):
        path: Path = Field(default=Path("/tmp/test"))

    env_vars = generate_env_vars_from_model(PathModel)
    env_dict = {name: value for name, value, _ in env_vars}

    assert isinstance(env_dict["PATH"], str)
    assert env_dict["PATH"] == "/tmp/test"


@pytest.mark.unit
def test_complex_model_nested_paths() -> None:
    """Test env var generation for complex model with nested paths."""
    config = generate_env_config(ComplexModel, commented=False)

    # Check for nested variable names
    assert "SIMPLE__NAME" in config
    assert "SIMPLE__COUNT" in config
    assert "NESTED__HOST" in config
    assert "NESTED__PORT" in config


@pytest.mark.unit
def test_list_and_dict_as_json() -> None:
    """Test that list and dict fields are formatted as JSON."""

    class JsonModel(BaseModel):
        items: list[str] = Field(default_factory=lambda: ["a", "b", "c"])
        metadata: dict[str, str] = Field(default_factory=lambda: {"key": "value"})

    env_vars = generate_env_vars_from_model(JsonModel)
    env_dict = {name: value for name, value, _ in env_vars}

    # Values should be JSON strings
    assert env_dict["ITEMS"] == ["a", "b", "c"]
    assert env_dict["METADATA"] == {"key": "value"}

    # When formatted for env, should be JSON
    config = generate_env_config(JsonModel, commented=False)
    assert '["a", "b", "c"]' in config
    assert '{"key": "value"}' in config


@pytest.mark.unit
def test_value_with_spaces_quoted() -> None:
    """Test that values with spaces are properly quoted."""

    class SpaceModel(BaseModel):
        message: str = Field(default="hello world")

    config = generate_env_config(SpaceModel, commented=False)

    # Should be quoted because it contains spaces
    assert 'export MESSAGE="hello world"' in config


@pytest.mark.unit
def test_empty_value_handling() -> None:
    """Test handling of None/empty values."""

    class OptionalModel(BaseModel):
        optional: str | None = Field(default=None)

    config = generate_env_config(OptionalModel, commented=False)

    # None should result in empty string
    assert 'export OPTIONAL=""' in config


@pytest.mark.unit
def test_dotenv_format() -> None:
    """Test .env file format (no export, uncommented)."""
    config = generate_env_config(SimpleModel, commented=False, export_format=False)

    # Should not have export prefix on variable lines
    assert "NAME=test" in config
    assert "COUNT=42" in config
    assert "export NAME" not in config
    assert "export COUNT" not in config

    # Variables themselves should not be commented
    lines = config.split("\n")
    var_lines = [
        line for line in lines if "=" in line and not line.strip().startswith("#")
    ]
    assert len(var_lines) == 3  # Should have 3 uncommented variable lines


@pytest.mark.unit
def test_plugin_prefix_format() -> None:
    """Test plugin-style prefix generation."""
    config = generate_env_config(
        SimpleModel, prefix="PLUGINS__MAX_TOKENS", commented=False
    )

    assert "export PLUGINS__MAX_TOKENS__NAME=test" in config
    assert "export PLUGINS__MAX_TOKENS__COUNT=42" in config
    assert "export PLUGINS__MAX_TOKENS__ENABLED=true" in config
