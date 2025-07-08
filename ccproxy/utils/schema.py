"""JSON Schema generation utilities for TOML configuration validation."""

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from ccproxy.config.settings import Settings


def generate_json_schema() -> dict[str, Any]:
    """Generate JSON Schema from the Settings Pydantic model.

    Returns:
        dict: JSON Schema for TOML configuration validation
    """
    # Import here to avoid circular import
    from ccproxy.config.settings import Settings

    schema = Settings.model_json_schema()

    # Enhance schema with TOML-specific metadata
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["title"] = "Claude Code Proxy API Configuration"
    schema["description"] = "Configuration schema for ccproxy TOML files"

    # Add examples and better descriptions
    if "properties" in schema:
        # Enhance server properties
        if "host" in schema["properties"]:
            schema["properties"]["host"]["examples"] = [
                "0.0.0.0",
                "127.0.0.1",
                "localhost",
            ]

        if "port" in schema["properties"]:
            schema["properties"]["port"]["examples"] = [8000, 8080, 3000]

        if "log_level" in schema["properties"]:
            schema["properties"]["log_level"]["examples"] = [
                "DEBUG",
                "INFO",
                "WARNING",
                "ERROR",
            ]

        if "cors_origins" in schema["properties"]:
            schema["properties"]["cors_origins"]["examples"] = [
                ["*"],
                ["http://localhost:3000", "https://example.com"],
            ]

        if "tools_handling" in schema["properties"]:
            schema["properties"]["tools_handling"]["description"] = (
                "How to handle tools definitions in requests. "
                "'error' raises an error, 'warning' logs a warning, 'ignore' silently ignores."
            )

    return schema


def save_schema_file(schema: dict[str, Any], output_path: Path) -> None:
    """Save JSON Schema to a file.

    Args:
        schema: JSON Schema dictionary
        output_path: Path where to save the schema file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w") as f:
        json.dump(schema, f, indent=2)


def generate_schema_files(output_dir: Path | None = None) -> list[Path]:
    """Generate JSON Schema files for configuration validation.

    Args:
        output_dir: Directory to save schema files. Defaults to project root.

    Returns:
        list[Path]: List of generated schema file paths
    """
    if output_dir is None:
        output_dir = Path.cwd()

    schema = generate_json_schema()
    generated_files = []

    # Main schema file for ccproxy configuration
    schema_file = output_dir / "ccproxy-schema.json"
    save_schema_file(schema, schema_file)
    generated_files.append(schema_file)

    # Schema file for .ccproxy.toml (current directory config)
    current_schema_file = output_dir / ".ccproxy-schema.json"
    save_schema_file(schema, current_schema_file)
    generated_files.append(current_schema_file)

    return generated_files


def generate_taplo_config(output_dir: Path | None = None) -> Path:
    """Generate taplo configuration for VS Code TOML support.

    Args:
        output_dir: Directory to save taplo config. Defaults to project root.

    Returns:
        Path: Path to generated taplo config file
    """
    if output_dir is None:
        output_dir = Path.cwd()

    config_file = output_dir / ".taplo.toml"

    # Use the correct taplo configuration format
    toml_content = """# Taplo configuration for TOML formatting and validation

# Include ccproxy TOML files
include = ["ccproxy.toml", ".ccproxy.toml", "**/ccproxy/*.toml"]

# Schema configuration for ccproxy files
[schema]
path = "ccproxy-schema.json"
enabled = true

# Formatting options
[formatting]
align_entries = false
array_trailing_comma = true
array_auto_expand = true
array_auto_collapse = true
compact_arrays = false
compact_inline_tables = false
compact_entries = false
indent_string = "  "
indent_entries = true
indent_tables = true
trailing_newline = true
reorder_keys = false
allowed_blank_lines = 1
column_width = 88

# Rule for key formatting
[[rule]]
include = ["**/*.toml"]
keys = ["*"]

[rule.formatting]
reorder_keys = false
"""

    config_file.write_text(toml_content)
    return config_file


def validate_toml_with_schema(toml_path: Path, schema_path: Path | None = None) -> bool:
    """Validate a TOML file against the generated JSON Schema.

    Args:
        toml_path: Path to TOML file to validate
        schema_path: Path to JSON Schema file. Auto-detects if None.

    Returns:
        bool: True if valid, False otherwise

    Raises:
        ValidationError: If TOML content is invalid according to schema
    """
    import subprocess
    import tomllib

    # Load TOML content and convert to JSON for validation
    with toml_path.open("rb") as f:
        toml_data = tomllib.load(f)

    # Create temporary JSON file from TOML data
    import tempfile

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as temp_json:
        json.dump(toml_data, temp_json)
        temp_json_path = temp_json.name

    try:
        # Load or generate schema
        if schema_path is None:
            schema = generate_json_schema()
            # Create temporary schema file
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False
            ) as temp_schema:
                json.dump(schema, temp_schema)
                schema_file = temp_schema.name
        else:
            schema_file = str(schema_path)

        try:
            # Use check-jsonschema CLI to validate
            result = subprocess.run(
                ["check-jsonschema", "--schemafile", schema_file, temp_json_path],
                capture_output=True,
                text=True,
                check=False,
            )
            return result.returncode == 0
        finally:
            # Clean up temporary schema file if we created one
            if schema_path is None:
                Path(schema_file).unlink(missing_ok=True)
    finally:
        # Clean up temporary JSON file
        Path(temp_json_path).unlink(missing_ok=True)


def validate_config_with_schema(
    config_path: Path, schema_path: Path | None = None
) -> bool:
    """Validate a configuration file against the generated JSON Schema.

    Args:
        config_path: Path to config file to validate (TOML, JSON, or YAML)
        schema_path: Path to JSON Schema file. Auto-detects if None.

    Returns:
        bool: True if valid, False otherwise

    Raises:
        ValidationError: If config content is invalid according to schema
        ValueError: If config format is unsupported
    """
    import subprocess
    import tempfile
    import tomllib

    suffix = config_path.suffix.lower()

    # Load config data based on file type
    if suffix in [".toml"]:
        with config_path.open("rb") as f:
            config_data = tomllib.load(f)
    elif suffix in [".json"]:
        with config_path.open("r", encoding="utf-8") as f:
            config_data = json.load(f)
    elif suffix in [".yaml", ".yml"]:
        try:
            import yaml  # type: ignore[import-untyped]
        except ImportError as e:
            raise ValueError(
                "YAML support is not available. Install with: pip install pyyaml"
            ) from e
        with config_path.open("r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f)
    else:
        raise ValueError(
            f"Unsupported config file format: {suffix}. "
            "Supported formats: .toml, .json, .yaml, .yml"
        )

    # Create temporary JSON file from config data
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as temp_json:
        json.dump(config_data, temp_json)
        temp_json_path = temp_json.name

    try:
        # Load or generate schema
        if schema_path is None:
            schema = generate_json_schema()
            # Create temporary schema file
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False
            ) as temp_schema:
                json.dump(schema, temp_schema)
                schema_file = temp_schema.name
        else:
            schema_file = str(schema_path)

        try:
            # Use check-jsonschema CLI to validate
            result = subprocess.run(
                ["check-jsonschema", "--schemafile", schema_file, temp_json_path],
                capture_output=True,
                text=True,
                check=False,
            )
            return result.returncode == 0
        finally:
            # Clean up temporary schema file if we created one
            if schema_path is None:
                Path(schema_file).unlink(missing_ok=True)
    finally:
        # Clean up temporary JSON file
        Path(temp_json_path).unlink(missing_ok=True)
