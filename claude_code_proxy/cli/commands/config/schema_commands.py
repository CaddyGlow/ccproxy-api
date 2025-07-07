"""Schema-related config commands for Claude Code Proxy API."""

from pathlib import Path

import typer

from claude_code_proxy.utils.cli import get_rich_toolkit
from claude_code_proxy.utils.schema import (
    generate_schema_files,
    generate_taplo_config,
    validate_config_with_schema,
)


def config_schema(
    output_dir: Path | None = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Output directory for schema files (default: current directory)",
    ),
    taplo: bool = typer.Option(
        False,
        "--taplo",
        help="Generate taplo configuration for TOML editor support",
    ),
) -> None:
    """Generate JSON Schema files for configuration validation.

    This command generates JSON Schema files that can be used by editors
    for configuration file validation, autocomplete, and syntax highlighting.
    Supports TOML, JSON, and YAML configuration files.

    Examples:
        ccproxy config schema                    # Generate schema files in current directory
        ccproxy config schema --output-dir ./schemas  # Generate in specific directory
        ccproxy config schema --taplo           # Also generate taplo config
    """
    toolkit = get_rich_toolkit()

    try:
        # Generate schema files
        if output_dir is None:
            output_dir = Path.cwd()

        toolkit.print(
            "Generating JSON Schema files for TOML configuration...", tag="info"
        )

        generated_files = generate_schema_files(output_dir)

        for file_path in generated_files:
            toolkit.print(f"Generated: {file_path}", tag="success")

        if taplo:
            toolkit.print("Generating taplo configuration...", tag="info")
            taplo_config = generate_taplo_config(output_dir)
            toolkit.print(f"Generated: {taplo_config}", tag="success")

        toolkit.print_line()
        toolkit.print("Schema files generated successfully!", tag="success")
        toolkit.print_line()
        toolkit.print("To use in VS Code:", tag="info")
        toolkit.print("1. Install the 'Even Better TOML' extension", tag="info")
        toolkit.print(
            "2. The schema will be automatically applied to ccproxy TOML files",
            tag="info",
        )
        toolkit.print_line()
        toolkit.print("To use with taplo CLI:", tag="info")
        if taplo:
            toolkit.print("  taplo check your-config.toml", tag="command")
        else:
            toolkit.print(
                "  ccproxy config schema --taplo  # Generate taplo config first",
                tag="command",
            )
            toolkit.print("  taplo check your-config.toml", tag="command")

    except Exception as e:
        toolkit.print(f"Error generating schema: {e}", tag="error")
        raise typer.Exit(1) from e


def config_validate(
    config_file: Path = typer.Argument(
        ...,
        help="Configuration file to validate (TOML, JSON, or YAML)",
    ),
) -> None:
    """Validate a configuration file against the schema.

    This command validates a configuration file (TOML, JSON, or YAML) against
    the JSON Schema to ensure it follows the correct structure and data types.

    Examples:
        ccproxy config validate config.toml  # Validate a TOML config
        ccproxy config validate config.yaml  # Validate a YAML config
        ccproxy config validate config.json  # Validate a JSON config
    """
    toolkit = get_rich_toolkit()

    try:
        # Validate the config file
        if not config_file.exists():
            toolkit.print(f"Error: File {config_file} does not exist.", tag="error")
            raise typer.Exit(1)

        toolkit.print(f"Validating {config_file}...", tag="info")

        try:
            is_valid = validate_config_with_schema(config_file)
            if is_valid:
                toolkit.print(
                    "Configuration file is valid according to schema.", tag="success"
                )
            else:
                toolkit.print("Configuration file validation failed.", tag="error")
                raise typer.Exit(1)
        except ImportError as e:
            toolkit.print(f"Error: {e}", tag="error")
            toolkit.print(
                "Install check-jsonschema: pip install check-jsonschema", tag="error"
            )
            raise typer.Exit(1) from e
        except Exception as e:
            toolkit.print(f"Validation error: {e}", tag="error")
            raise typer.Exit(1) from e

    except Exception as e:
        toolkit.print(f"Error validating configuration: {e}", tag="error")
        raise typer.Exit(1) from e
