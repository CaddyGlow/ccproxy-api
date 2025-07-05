"""JSON Schema generation utilities for TOML configuration validation."""

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from claude_code_proxy.config.settings import Settings


def generate_json_schema() -> dict[str, Any]:
    """Generate JSON Schema from the Settings Pydantic model.

    Returns:
        dict: JSON Schema for TOML configuration validation
    """
    # Import here to avoid circular import
    from claude_code_proxy.config.settings import Settings

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
    """Generate JSON Schema files for TOML configuration.

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
    try:
        import jsonschema  # type: ignore[import-untyped]
    except ImportError:
        raise ImportError(
            "jsonschema package required for validation. "
            "Install with: pip install jsonschema"
        ) from None

    import tomllib

    # Load TOML content
    with toml_path.open("rb") as f:
        toml_data = tomllib.load(f)

    # Load or generate schema
    if schema_path is None:
        schema = generate_json_schema()
    else:
        with schema_path.open() as f:
            schema = json.load(f)

    # Validate
    try:
        jsonschema.validate(toml_data, schema)
        return True
    except jsonschema.ValidationError:
        return False
