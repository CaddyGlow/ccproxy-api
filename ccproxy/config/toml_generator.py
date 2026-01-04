"""Utility functions for generating TOML configuration from Pydantic models."""

import sys
from pathlib import Path
from typing import Any, TextIO, TypeVar

from pydantic import BaseModel
from pydantic.fields import FieldInfo


T = TypeVar("T", bound=BaseModel)


def is_hidden_in_example(field_info: FieldInfo) -> bool:
    """Determine if a field should be omitted from generated example configs."""
    if bool(field_info.exclude):
        return True

    extra = getattr(field_info, "json_schema_extra", None) or {}
    return bool(extra.get("config_example_hidden"))


def get_field_description(field_info: FieldInfo) -> str:
    """Get a human-readable description from a Pydantic field."""
    if field_info.description:
        return field_info.description
    return "Configuration setting"


def generate_config_from_model(
    model_class: type[T], include_hidden: bool = False
) -> dict[str, Any]:
    """Generate a default configuration dictionary from a Pydantic model class.

    Args:
        model_class: The Pydantic model class to generate config from
        include_hidden: Whether to include fields marked as hidden in examples

    Returns:
        Dictionary containing the configuration data
    """
    default_instance = model_class()
    config_data: dict[str, Any] = {}

    for field_name, field_info in model_class.model_fields.items():
        if not include_hidden and is_hidden_in_example(field_info):
            continue

        field_value = getattr(default_instance, field_name)

        if isinstance(field_value, BaseModel):
            nested_config = generate_nested_config(field_value, include_hidden)
            if nested_config:
                config_data[field_name] = nested_config
        else:
            if isinstance(field_value, Path):
                config_data[field_name] = str(field_value)
            else:
                config_data[field_name] = field_value

    return config_data


def generate_nested_config(
    model: BaseModel, include_hidden: bool = False
) -> dict[str, Any]:
    """Generate configuration for nested Pydantic models.

    Args:
        model: The Pydantic model instance to generate config from
        include_hidden: Whether to include fields marked as hidden in examples

    Returns:
        Dictionary containing the nested configuration data
    """
    config_data: dict[str, Any] = {}

    # Access model_fields from the class, not the instance
    for field_name, field_info in model.__class__.model_fields.items():
        if not include_hidden and is_hidden_in_example(field_info):
            continue

        field_value = getattr(model, field_name)

        if isinstance(field_value, BaseModel):
            nested_config = generate_nested_config(field_value, include_hidden)
            if nested_config:
                config_data[field_name] = nested_config
        else:
            if isinstance(field_value, Path):
                config_data[field_name] = str(field_value)
            else:
                config_data[field_name] = field_value

    return config_data


def format_value_for_toml(value: Any) -> str:
    """Format a configuration value for TOML output.

    Args:
        value: The value to format

    Returns:
        String representation suitable for TOML
    """
    if value is None:
        return "null"
    elif isinstance(value, bool):
        return "true" if value else "false"
    elif isinstance(value, str):
        # Escape quotes in strings
        escaped = value.replace('"', '\\"')
        return f'"{escaped}"'
    elif isinstance(value, int | float):
        return str(value)
    elif isinstance(value, list):
        if not value:
            return "[]"
        formatted_items = []
        for item in value:
            if isinstance(item, str):
                escaped = item.replace('"', '\\"')
                formatted_items.append(f'"{escaped}"')
            else:
                formatted_items.append(str(item))
        return f"[{', '.join(formatted_items)}]"
    elif isinstance(value, dict):
        if not value:
            return "{}"
        formatted_items = []
        for k, v in value.items():
            if isinstance(v, str):
                escaped = v.replace('"', '\\"')
                formatted_items.append(f'{k} = "{escaped}"')
            else:
                formatted_items.append(f"{k} = {v}")
        return f"{{{', '.join(formatted_items)}}}"
    else:
        return str(value)


def generate_toml_section(
    data: dict[str, Any], prefix: str = "", level: int = 0
) -> str:
    """Generate a TOML section string with proper indentation and commenting.

    Args:
        data: Dictionary of configuration data
        prefix: Comment prefix (e.g., "# " for commented sections)
        level: Nesting level for proper formatting

    Returns:
        TOML section as a string
    """
    lines = []
    for key, value in data.items():
        if isinstance(value, dict):
            lines.append(f"{prefix}[{key}]")
            lines.append(generate_toml_section(value, prefix, level + 1))
        else:
            formatted_value = format_value_for_toml(value)
            lines.append(f"{prefix}{key} = {formatted_value}")
    return "\n".join(lines)


def generate_toml_config(
    config_data: dict[str, Any],
    model_class: type[BaseModel],
    header_comment: str | None = None,
    commented: bool = True,
    root_field: str | None = None,
) -> str:
    """Generate TOML configuration string from config data and model class.

    Args:
        config_data: Configuration dictionary to convert to TOML
        model_class: The Pydantic model class (used for field descriptions)
        header_comment: Optional custom header comment
        commented: Whether to comment out all settings (default True)
        root_field: Optional root section path (e.g., "plugins.max_tokens")
                   to nest all fields under

    Returns:
        TOML configuration as a string
    """
    lines = []

    # Write header
    if header_comment:
        for line in header_comment.split("\n"):
            lines.append(f"# {line}" if line else "#")
    else:
        lines.append("# Configuration File")
        lines.append("# This file configures the settings for the application")
        if commented:
            lines.append("# Most settings are commented out with their default values")
            lines.append("# Uncomment and modify as needed")
    lines.append("")

    prefix = "# " if commented else ""

    # If root_field is specified, write it as a section header
    if root_field:
        lines.append(f"{prefix}[{root_field}]")

    # Write fields with descriptions
    for field_name, field_info in model_class.model_fields.items():
        if is_hidden_in_example(field_info):
            continue

        field_value = config_data.get(field_name)
        if field_value is None:
            continue

        description = get_field_description(field_info)

        # Write description as comment
        lines.append(f"# {description}")

        if isinstance(field_value, dict):
            # For nested dicts under root_field, use subsection notation
            section_name = f"{root_field}.{field_name}" if root_field else field_name
            lines.append(f"{prefix}[{section_name}]")
            lines.append(generate_toml_section(field_value, prefix=prefix, level=0))
        else:
            formatted_value = format_value_for_toml(field_value)
            lines.append(f"{prefix}{field_name} = {formatted_value}")

        lines.append("")

    return "\n".join(lines)


def write_toml_config(
    output: TextIO | Path | str,
    model_class: type[BaseModel],
    config_data: dict[str, Any] | None = None,
    header_comment: str | None = None,
    commented: bool = True,
    include_hidden: bool = False,
    root_field: str | None = None,
) -> None:
    """Write TOML configuration directly to a stream or file.

    Args:
        output: Output destination - can be a TextIO stream (file, StringIO, stdout),
                a Path object, or a string path to a file
        model_class: The Pydantic model class to generate config from
        config_data: Optional config dictionary. If None, generates from model defaults
        header_comment: Optional custom header comment
        commented: Whether to comment out all settings (default True)
        include_hidden: Whether to include fields marked as hidden in examples
        root_field: Optional root section path (e.g., "plugins.max_tokens")

    Examples:
        # Write to stdout
        write_toml_config(sys.stdout, Settings)

        # Write to file
        write_toml_config("config.toml", Settings)
        write_toml_config(Path("config.toml"), Settings)

        # Write to StringIO
        buffer = StringIO()
        write_toml_config(buffer, Settings)
        content = buffer.getvalue()

        # Write to file object
        with open("config.toml", "w") as f:
            write_toml_config(f, Settings)

        # Write with root field
        write_toml_config(sys.stdout, MaxTokensConfig, root_field="plugins.max_tokens")
    """
    # Generate config data if not provided
    if config_data is None:
        config_data = generate_config_from_model(model_class, include_hidden)

    # Generate TOML string
    toml_string = generate_toml_config(
        config_data=config_data,
        model_class=model_class,
        header_comment=header_comment,
        commented=commented,
        root_field=root_field,
    )

    # Determine output type and write
    if isinstance(output, str | Path):
        # Write to file path
        Path(output).write_text(toml_string, encoding="utf-8")
    else:
        # Write to stream (TextIO, stdout, StringIO, etc.)
        output.write(toml_string)
        if output not in (sys.stdout, sys.stderr):
            # Ensure trailing newline for files (not needed for stdout/stderr)
            if not toml_string.endswith("\n"):
                output.write("\n")
