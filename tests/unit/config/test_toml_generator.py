"""Tests for TOML configuration generator utilities."""

import sys
from io import StringIO
from pathlib import Path

import pytest
from pydantic import BaseModel, Field

from ccproxy.config.toml_generator import (
    format_value_for_toml,
    generate_config_from_model,
    generate_nested_config,
    generate_toml_config,
    generate_toml_section,
    get_field_description,
    is_hidden_in_example,
    write_toml_config,
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
    model = HiddenFieldModel()
    field_info = HiddenFieldModel.model_fields["visible"]
    assert not is_hidden_in_example(field_info)

    hidden_field_info = HiddenFieldModel.model_fields["hidden"]
    assert is_hidden_in_example(hidden_field_info)


@pytest.mark.unit
def test_get_field_description() -> None:
    """Test field description extraction."""
    field_info = SimpleModel.model_fields["name"]
    assert get_field_description(field_info) == "Name field"

    # Test field without description
    class NoDescModel(BaseModel):
        field: str = "test"

    field_info = NoDescModel.model_fields["field"]
    assert get_field_description(field_info) == "Configuration setting"


@pytest.mark.unit
def test_format_value_for_toml() -> None:
    """Test TOML value formatting."""
    assert format_value_for_toml(None) == "null"
    assert format_value_for_toml(True) == "true"
    assert format_value_for_toml(False) == "false"
    assert format_value_for_toml(42) == "42"
    assert format_value_for_toml(3.14) == "3.14"
    assert format_value_for_toml("hello") == '"hello"'
    assert format_value_for_toml([1, 2, 3]) == "[1, 2, 3]"
    assert format_value_for_toml(["a", "b"]) == '["a", "b"]'
    assert format_value_for_toml([]) == "[]"
    assert format_value_for_toml({}) == "{}"


@pytest.mark.unit
def test_format_value_escapes_quotes() -> None:
    """Test that quotes are properly escaped in strings."""
    assert format_value_for_toml('hello "world"') == '"hello \\"world\\""'
    assert format_value_for_toml(['say "hi"']) == '["say \\"hi\\""]'


@pytest.mark.unit
def test_generate_config_from_model() -> None:
    """Test config dictionary generation from model."""
    config = generate_config_from_model(SimpleModel)

    assert config["name"] == "test"
    assert config["count"] == 42
    assert config["enabled"] is True


@pytest.mark.unit
def test_generate_config_from_model_with_hidden() -> None:
    """Test config generation includes/excludes hidden fields."""
    # Without hidden fields (default)
    config = generate_config_from_model(HiddenFieldModel, include_hidden=False)
    assert "visible" in config
    assert "hidden" not in config

    # With hidden fields
    config = generate_config_from_model(HiddenFieldModel, include_hidden=True)
    assert "visible" in config
    assert "hidden" in config


@pytest.mark.unit
def test_generate_nested_config() -> None:
    """Test nested model config generation."""
    model = NestedModel()
    config = generate_nested_config(model)

    assert config["host"] == "localhost"
    assert config["port"] == 8000


@pytest.mark.unit
def test_generate_toml_section() -> None:
    """Test TOML section generation."""
    data = {"host": "localhost", "port": 8000}
    result = generate_toml_section(data)

    assert "host = " in result
    assert "port = " in result


@pytest.mark.unit
def test_generate_toml_section_with_prefix() -> None:
    """Test TOML section generation with comment prefix."""
    data = {"host": "localhost"}
    result = generate_toml_section(data, prefix="# ")

    assert "# host = " in result


@pytest.mark.unit
def test_generate_toml_config_simple() -> None:
    """Test TOML config generation for simple model."""
    config_data = generate_config_from_model(SimpleModel)
    toml = generate_toml_config(config_data, SimpleModel, commented=True)

    assert "# Configuration File" in toml
    assert "# name = " in toml
    assert "# count = " in toml
    assert "# enabled = " in toml


@pytest.mark.unit
def test_generate_toml_config_uncommented() -> None:
    """Test TOML config generation without comments."""
    config_data = generate_config_from_model(SimpleModel)
    toml = generate_toml_config(config_data, SimpleModel, commented=False)

    assert 'name = "test"' in toml
    assert "count = 42" in toml
    assert "enabled = true" in toml
    assert "# name = " not in toml


@pytest.mark.unit
def test_generate_toml_config_with_root_field() -> None:
    """Test TOML config generation with root field."""
    config_data = generate_config_from_model(SimpleModel)
    toml = generate_toml_config(
        config_data, SimpleModel, commented=False, root_field="plugins.test"
    )

    assert "[plugins.test]" in toml
    assert 'name = "test"' in toml


@pytest.mark.unit
def test_generate_toml_config_with_root_field_commented() -> None:
    """Test TOML config generation with root field and comments."""
    config_data = generate_config_from_model(SimpleModel)
    toml = generate_toml_config(
        config_data, SimpleModel, commented=True, root_field="plugins.test"
    )

    assert "# [plugins.test]" in toml
    assert "# name = " in toml


@pytest.mark.unit
def test_generate_toml_config_custom_header() -> None:
    """Test TOML config generation with custom header."""
    config_data = generate_config_from_model(SimpleModel)
    custom_header = "Custom Header\nLine 2"
    toml = generate_toml_config(config_data, SimpleModel, header_comment=custom_header)

    assert "# Custom Header" in toml
    assert "# Line 2" in toml


@pytest.mark.unit
def test_write_toml_config_to_stringio() -> None:
    """Test writing TOML config to StringIO."""
    buffer = StringIO()
    write_toml_config(buffer, SimpleModel, commented=False)

    result = buffer.getvalue()
    assert 'name = "test"' in result
    assert "count = 42" in result


@pytest.mark.unit
def test_write_toml_config_to_file(tmp_path: Path) -> None:
    """Test writing TOML config to file."""
    output_file = tmp_path / "test.toml"
    write_toml_config(output_file, SimpleModel, commented=False)

    assert output_file.exists()
    content = output_file.read_text()
    assert 'name = "test"' in content


@pytest.mark.unit
def test_write_toml_config_to_stdout(capsys: pytest.CaptureFixture[str]) -> None:
    """Test writing TOML config to stdout."""
    write_toml_config(sys.stdout, SimpleModel, commented=False)

    captured = capsys.readouterr()
    assert 'name = "test"' in captured.out


@pytest.mark.unit
def test_write_toml_config_with_root_field() -> None:
    """Test writing TOML config with root field."""
    buffer = StringIO()
    write_toml_config(buffer, SimpleModel, commented=False, root_field="plugins.simple")

    result = buffer.getvalue()
    assert "[plugins.simple]" in result
    assert 'name = "test"' in result


@pytest.mark.unit
def test_write_toml_config_with_custom_data() -> None:
    """Test writing TOML config with custom config data."""
    buffer = StringIO()
    custom_data = {"name": "custom", "count": 100, "enabled": False}

    write_toml_config(buffer, SimpleModel, config_data=custom_data, commented=False)

    result = buffer.getvalue()
    assert 'name = "custom"' in result
    assert "count = 100" in result
    assert "enabled = false" in result


@pytest.mark.unit
def test_complex_model_with_nested_sections() -> None:
    """Test TOML generation for complex model with nested sections."""
    config_data = generate_config_from_model(ComplexModel)
    toml = generate_toml_config(config_data, ComplexModel, commented=False)

    # Check for nested sections
    assert "[simple]" in toml
    assert "[nested]" in toml
    assert "tags = " in toml


@pytest.mark.unit
def test_path_field_converted_to_string() -> None:
    """Test that Path fields are converted to strings."""

    class PathModel(BaseModel):
        path: Path = Field(default=Path("/tmp/test"))

    config = generate_config_from_model(PathModel)
    assert isinstance(config["path"], str)
    assert config["path"] == "/tmp/test"


@pytest.mark.unit
def test_write_toml_config_with_string_path(tmp_path: Path) -> None:
    """Test writing TOML config with string path."""
    output_file = str(tmp_path / "test.toml")
    write_toml_config(output_file, SimpleModel, commented=False)

    assert Path(output_file).exists()


@pytest.mark.unit
def test_generate_toml_config_nested_with_root_field() -> None:
    """Test nested model with root field generates correct section paths."""
    config_data = generate_config_from_model(ComplexModel)
    toml = generate_toml_config(
        config_data, ComplexModel, commented=False, root_field="app"
    )

    assert "[app]" in toml
    assert "[app.simple]" in toml
    assert "[app.nested]" in toml


@pytest.mark.unit
def test_dict_field_formatting() -> None:
    """Test dictionary field formatting in TOML."""

    class DictModel(BaseModel):
        metadata: dict[str, str] = Field(
            default_factory=lambda: {"key1": "value1", "key2": "value2"}
        )

    config = generate_config_from_model(DictModel)
    toml = generate_toml_config(config, DictModel, commented=False)

    # Dict should be formatted as section
    assert "[metadata]" in toml
    assert "key1" in toml
    assert "value1" in toml
