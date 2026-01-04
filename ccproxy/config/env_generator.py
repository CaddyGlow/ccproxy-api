"""Utility functions for generating environment variable configuration from Pydantic models."""

import json
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


def format_value_for_env(value: Any) -> str:
    """Format a configuration value for environment variable output.

    Args:
        value: The value to format

    Returns:
        String representation suitable for environment variables
    """
    if value is None:
        return ""
    elif isinstance(value, bool):
        return "true" if value else "false"
    elif isinstance(value, str):
        return value
    elif isinstance(value, int | float):
        return str(value)
    elif isinstance(value, list | dict):
        # For complex types, try JSON representation
        try:
            return json.dumps(value)
        except (TypeError, ValueError):
            # If JSON serialization fails, use string representation
            return str(value)
    else:
        return str(value)


def generate_env_vars_from_model(
    model_class: type[T],
    prefix: str = "",
    include_hidden: bool = False,
) -> list[tuple[str, Any, str]]:
    """Generate environment variable names and values from a Pydantic model.

    Args:
        model_class: The Pydantic model class to generate env vars from
        prefix: Prefix for env var names (e.g., "SERVER" or "PLUGINS__MAX_TOKENS")
        include_hidden: Whether to include fields marked as hidden in examples

    Returns:
        List of tuples: (env_var_name, value, description)
    """
    default_instance = model_class()
    env_vars: list[tuple[str, Any, str]] = []

    for field_name, field_info in model_class.model_fields.items():
        if not include_hidden and is_hidden_in_example(field_info):
            continue

        field_value = getattr(default_instance, field_name)
        env_var_name = (
            f"{prefix}__{field_name.upper()}" if prefix else field_name.upper()
        )
        description = get_field_description(field_info)

        if isinstance(field_value, BaseModel):
            # Recursively handle nested models
            nested_vars = generate_env_vars_from_model(
                field_value.__class__, env_var_name, include_hidden
            )
            env_vars.extend(nested_vars)
        else:
            # Convert Path to string
            if isinstance(field_value, Path):
                field_value = str(field_value)

            env_vars.append((env_var_name, field_value, description))

    return env_vars


def generate_env_config(
    model_class: type[BaseModel],
    prefix: str = "",
    include_hidden: bool = False,
    commented: bool = True,
    header_comment: str | None = None,
    export_format: bool = True,
) -> str:
    """Generate environment variable configuration string.

    Args:
        model_class: The Pydantic model class to generate config from
        prefix: Prefix for env var names (e.g., "SERVER" or "PLUGINS__MAX_TOKENS")
        include_hidden: Whether to include fields marked as hidden in examples
        commented: Whether to comment out all settings (default True)
        header_comment: Optional custom header comment
        export_format: Whether to use 'export VAR=value' format (True) or 'VAR=value' (False)

    Returns:
        Environment variable configuration as a string
    """
    lines = []

    # Write header
    if header_comment:
        for line in header_comment.split("\n"):
            lines.append(f"# {line}" if line else "#")
    else:
        lines.append("# Environment Variable Configuration")
        lines.append("# This file contains environment variables for the application")
        if commented:
            lines.append("# Uncomment and set values as needed")
    lines.append("")

    # Generate environment variables
    env_vars = generate_env_vars_from_model(model_class, prefix, include_hidden)

    comment_prefix = "# " if commented else ""
    export_prefix = "export " if export_format else ""

    for env_var_name, value, description in env_vars:
        # Write description as comment
        lines.append(f"# {description}")

        # Format the value
        formatted_value = format_value_for_env(value)

        # Write the environment variable
        if formatted_value:
            # Quote values that contain spaces or special characters
            if isinstance(value, str) and (
                " " in value or any(c in value for c in ["$", '"', "'", "\\"])
            ):
                lines.append(
                    f'{comment_prefix}{export_prefix}{env_var_name}="{formatted_value}"'
                )
            else:
                lines.append(
                    f"{comment_prefix}{export_prefix}{env_var_name}={formatted_value}"
                )
        else:
            # Empty value
            lines.append(f'{comment_prefix}{export_prefix}{env_var_name}=""')

        lines.append("")

    return "\n".join(lines)


def write_env_config(
    output: TextIO | Path | str,
    model_class: type[BaseModel],
    prefix: str = "",
    include_hidden: bool = False,
    commented: bool = True,
    header_comment: str | None = None,
    export_format: bool = True,
) -> None:
    """Write environment variable configuration directly to a stream or file.

    Args:
        output: Output destination - can be a TextIO stream (file, StringIO, stdout),
                a Path object, or a string path to a file
        model_class: The Pydantic model class to generate config from
        prefix: Prefix for env var names (e.g., "SERVER" or "PLUGINS__MAX_TOKENS")
        include_hidden: Whether to include fields marked as hidden in examples
        commented: Whether to comment out all settings (default True)
        header_comment: Optional custom header comment
        export_format: Whether to use 'export VAR=value' format (default True)

    Examples:
        # Write to stdout
        write_env_config(sys.stdout, Settings)

        # Write to file
        write_env_config("env.sh", Settings)
        write_env_config(Path("env.sh"), Settings)

        # Write to StringIO
        buffer = StringIO()
        write_env_config(buffer, Settings)
        content = buffer.getvalue()

        # Write with prefix for plugin
        write_env_config(sys.stdout, MaxTokensConfig, prefix="PLUGINS__MAX_TOKENS")

        # Write without export (for .env files)
        write_env_config("env.sh", Settings, export_format=False)
    """
    # Generate env config string
    env_string = generate_env_config(
        model_class=model_class,
        prefix=prefix,
        include_hidden=include_hidden,
        commented=commented,
        header_comment=header_comment,
        export_format=export_format,
    )

    # Determine output type and write
    if isinstance(output, str | Path):
        # Write to file path
        Path(output).write_text(env_string, encoding="utf-8")
    else:
        # Write to stream (TextIO, stdout, StringIO, etc.)
        output.write(env_string)
        if output not in (sys.stdout, sys.stderr):
            # Ensure trailing newline for files (not needed for stdout/stderr)
            if not env_string.endswith("\n"):
                output.write("\n")
