#!/usr/bin/env python3
"""Generate configuration (TOML or ENV) from a Pydantic model file.

Usage:
    python scripts/generate_config_from_model.py --format {toml|env} <model_file> <model_class> [options]

Examples:
    # Generate TOML configuration
    python scripts/generate_config_from_model.py --format toml ccproxy/config/settings.py Settings
    python scripts/generate_config_from_model.py -f toml --plugin max_tokens
    python scripts/generate_config_from_model.py -f toml --plugin max_tokens --uncommented

    # Generate ENV configuration
    python scripts/generate_config_from_model.py --format env ccproxy/config/settings.py Settings
    python scripts/generate_config_from_model.py -f env --plugin max_tokens
    python scripts/generate_config_from_model.py -f env --plugin copilot --no-export

    # Save to file
    python scripts/generate_config_from_model.py -f toml --plugin max_tokens -o config.toml
    python scripts/generate_config_from_model.py -f env --plugin max_tokens -o env.sh

    # Plugin mode with custom config class
    python scripts/generate_config_from_model.py -f toml --plugin claude_api --config-class ClaudeAPISettings
    python scripts/generate_config_from_model.py -f env --plugin claude_api --config-class ClaudeAPISettings
"""

import argparse
import importlib
import sys
from pathlib import Path

from pydantic import BaseModel


def import_model_from_file(file_path: str, model_name: str) -> type[BaseModel]:
    """Dynamically import a Pydantic model class from a Python file.

    Args:
        file_path: Path to the Python file containing the model
        model_name: Name of the model class to import

    Returns:
        The Pydantic model class

    Raises:
        FileNotFoundError: If the file doesn't exist
        AttributeError: If the model class doesn't exist in the file
        TypeError: If the imported class is not a Pydantic model
    """
    path = Path(file_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Add project root to Python path for relative imports
    project_root = Path.cwd()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    # Try to import as a Python module using dot notation
    try:
        relative_path = path.relative_to(project_root)
        module_path = str(relative_path.with_suffix("")).replace("/", ".")
        module = importlib.import_module(module_path)
    except (ValueError, ImportError) as e:
        raise ImportError(
            f"Could not import module from {file_path}. "
            f"Make sure it's within the project structure and has no import errors. "
            f"Error: {e}"
        ) from e

    # Get the model class
    if not hasattr(module, model_name):
        available = [name for name in dir(module) if not name.startswith("_")]
        raise AttributeError(
            f"Model class '{model_name}' not found in {file_path}. "
            f"Available classes: {available}"
        )

    model_class = getattr(module, model_name)

    # Verify it's a Pydantic model
    if not (isinstance(model_class, type) and issubclass(model_class, BaseModel)):
        raise TypeError(
            f"'{model_name}' is not a Pydantic BaseModel subclass. "
            f"Type: {type(model_class)}"
        )

    return model_class


def find_plugin_config(
    plugin_name: str, config_class_override: str | None = None
) -> tuple[str, str]:
    """Find the config file and class for a plugin.

    Args:
        plugin_name: The plugin name (e.g., "max_tokens", "copilot")
        config_class_override: Optional explicit config class name

    Returns:
        Tuple of (config_file_path, config_class_name)

    Raises:
        FileNotFoundError: If plugin config file not found
    """
    # Plugin config path
    plugin_dir = Path.cwd() / "ccproxy" / "plugins" / plugin_name
    config_file = plugin_dir / "config.py"

    if not config_file.exists():
        raise FileNotFoundError(
            f"Plugin config file not found: {config_file}\n"
            f"Expected plugin directory: {plugin_dir}"
        )

    # Use override if provided
    if config_class_override:
        class_name = config_class_override
    else:
        # Try common naming conventions
        # Convert plugin_name to PascalCase
        class_name_parts = plugin_name.split("_")
        pascal_name = "".join(part.capitalize() for part in class_name_parts)

        # Common pattern: PluginNameConfig (e.g., MaxTokensConfig)
        class_name = pascal_name + "Config"

    return str(config_file), class_name


def get_plugin_prefix(plugin_name: str, format_type: str) -> str:
    """Get the appropriate prefix for a plugin based on format type.

    Args:
        plugin_name: The plugin name (e.g., "max_tokens")
        format_type: The format type ("toml" or "env")

    Returns:
        Prefix string appropriate for the format
    """
    if format_type == "toml":
        # TOML uses dot notation: plugins.max_tokens
        return f"plugins.{plugin_name}"
    elif format_type == "env":
        # ENV uses double underscore: PLUGINS__MAX_TOKENS
        return f"PLUGINS__{plugin_name.upper()}"
    else:
        raise ValueError(f"Unknown format type: {format_type}")


def main() -> None:
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Generate configuration (TOML or ENV) from a Pydantic model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Format selection (required)
    parser.add_argument(
        "--format",
        "-f",
        choices=["toml", "env"],
        required=True,
        help="Output format: 'toml' for TOML config, 'env' for environment variables",
    )

    # Model specification
    parser.add_argument(
        "model_file",
        nargs="?",
        help="Path to Python file containing the Pydantic model (e.g., ccproxy/config/core.py)",
    )
    parser.add_argument(
        "model_class",
        nargs="?",
        help="Name of the Pydantic model class (e.g., Settings)",
    )

    # Common options
    parser.add_argument(
        "-o",
        "--output",
        help="Output file path (default: stdout)",
        type=Path,
    )
    parser.add_argument(
        "--uncommented",
        action="store_true",
        help="Generate uncommented config (default: commented)",
    )
    parser.add_argument(
        "--include-hidden",
        action="store_true",
        help="Include fields marked as hidden in examples",
    )
    parser.add_argument(
        "--header",
        help="Custom header comment for the config file",
    )

    # Plugin mode
    parser.add_argument(
        "--plugin",
        "-p",
        help="Plugin name (auto-detects config file and sets prefix). Example: 'max_tokens', 'copilot'",
    )
    parser.add_argument(
        "--config-class",
        help="Override config class name when using --plugin mode (e.g., 'ClaudeAPISettings')",
    )

    # Format-specific options
    toml_group = parser.add_argument_group("TOML-specific options")
    toml_group.add_argument(
        "--root-field",
        help="[TOML] Root section path to nest fields under (e.g., 'plugins.max_tokens')",
    )

    env_group = parser.add_argument_group("ENV-specific options")
    env_group.add_argument(
        "--prefix",
        help="[ENV] Prefix for env var names (e.g., 'SERVER' or 'PLUGINS__MAX_TOKENS')",
    )
    env_group.add_argument(
        "--no-export",
        action="store_true",
        help="[ENV] Don't use 'export' prefix (for .env files)",
    )

    args = parser.parse_args()

    try:
        # Import the appropriate generator
        if args.format == "toml":
            from ccproxy.config.toml_generator import write_toml_config as write_config
        else:  # env
            from ccproxy.config.env_generator import write_env_config as write_config

        # Determine model file, class, and prefix
        if args.plugin:
            # Plugin mode: auto-detect config file and class
            if args.model_file or args.model_class:
                print(
                    "Warning: --plugin mode ignores model_file and model_class arguments",
                    file=sys.stderr,
                )

            model_file, model_class_name = find_plugin_config(
                args.plugin, args.config_class
            )

            # Get format-specific prefix
            auto_prefix = get_plugin_prefix(args.plugin, args.format)

            # Use provided prefix/root-field if specified, otherwise use auto-detected
            if args.format == "toml":
                prefix = args.root_field if args.root_field else auto_prefix
                prefix_label = "Root field"
            else:  # env
                prefix = args.prefix if args.prefix else auto_prefix
                prefix_label = "Env prefix"

            print(
                f"Plugin mode: {args.plugin}\n"
                f"  Config file: {model_file}\n"
                f"  Config class: {model_class_name}\n"
                f"  {prefix_label}: {prefix}",
                file=sys.stderr,
            )
        else:
            # Manual mode: use provided file and class
            if not args.model_file or not args.model_class:
                parser.error(
                    "model_file and model_class are required when not using --plugin mode"
                )

            model_file = args.model_file
            model_class_name = args.model_class

            # Get prefix from format-specific arguments
            if args.format == "toml":
                prefix = args.root_field if args.root_field else ""
            else:  # env
                prefix = args.prefix if args.prefix else ""

        # Import the model class
        model_class = import_model_from_file(model_file, model_class_name)

        # Write to output (file or stdout)
        output = args.output if args.output else sys.stdout

        # Build kwargs based on format
        common_kwargs = {
            "output": output,
            "model_class": model_class,
            "header_comment": args.header,
            "commented": not args.uncommented,
            "include_hidden": args.include_hidden,
        }

        if args.format == "toml":
            write_config(**common_kwargs, root_field=prefix if prefix else None)
        else:  # env
            write_config(
                **common_kwargs, prefix=prefix, export_format=not args.no_export
            )

        # Print confirmation to stderr if writing to file
        if args.output:
            format_name = (
                "TOML configuration"
                if args.format == "toml"
                else "Environment variable configuration"
            )
            print(f"{format_name} written to: {args.output}", file=sys.stderr)

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except (AttributeError, TypeError, ImportError) as e:
        print(f"Error loading model: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
