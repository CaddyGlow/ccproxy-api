"""Config command for Claude Code Proxy API."""

import json
from pathlib import Path
from typing import Optional

import typer
from click import get_current_context
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from claude_code_proxy._version import __version__
from claude_code_proxy.config.settings import get_settings
from claude_code_proxy.utils.schema import (
    generate_schema_files,
    generate_taplo_config,
    validate_config_with_schema,
)


def get_config_path_from_context() -> Path | None:
    """Get config path from typer context if available."""
    try:
        ctx = get_current_context()
        if ctx and ctx.obj and "config_path" in ctx.obj:
            config_path = ctx.obj["config_path"]
            return config_path if config_path is None else Path(config_path)
    except RuntimeError:
        # No active click context (e.g., in tests)
        pass
    return None


app = typer.Typer(
    name="config",
    help="Configuration management commands",
    rich_markup_mode="rich",
    add_completion=True,
    no_args_is_help=True,
)


@app.command(name="list")
def config_list() -> None:
    """Show current configuration."""
    try:
        settings = get_settings(config_path=get_config_path_from_context())
        console = Console()

        # Main server configuration table
        server_table = Table(
            title="Server Configuration", show_header=True, header_style="bold magenta"
        )
        server_table.add_column("Setting", style="cyan", width=20)
        server_table.add_column("Value", style="green")
        server_table.add_column("Description", style="dim")

        server_table.add_row("host", settings.host, "Server host address")
        server_table.add_row("port", str(settings.port), "Server port number")
        server_table.add_row("log_level", settings.log_level, "Logging verbosity level")
        server_table.add_row(
            "workers", str(settings.workers), "Number of worker processes"
        )
        server_table.add_row(
            "reload", str(settings.reload), "Auto-reload for development"
        )
        server_table.add_row(
            "server_url", settings.server_url, "Complete server URL (computed)"
        )

        # Claude CLI configuration table
        claude_table = Table(
            title="Claude CLI Configuration",
            show_header=True,
            header_style="bold magenta",
        )
        claude_table.add_column("Setting", style="cyan", width=20)
        claude_table.add_column("Value", style="green")
        claude_table.add_column("Description", style="dim")

        claude_path_display = settings.claude_cli_path or "[dim]Auto-detect[/dim]"
        claude_table.add_row(
            "claude_cli_path", claude_path_display, "Path to Claude CLI executable"
        )

        # Security configuration table
        security_table = Table(
            title="Security Configuration",
            show_header=True,
            header_style="bold magenta",
        )
        security_table.add_column("Setting", style="cyan", width=20)
        security_table.add_column("Value", style="green")
        security_table.add_column("Description", style="dim")

        auth_token_display = (
            "[dim]Not set[/dim]" if not settings.auth_token else "[green]Set[/green]"
        )
        cors_origins_display = (
            ", ".join(settings.cors_origins)
            if settings.cors_origins
            else "[dim]None[/dim]"
        )
        security_table.add_row(
            "tools_handling", settings.tools_handling, "How to handle tools in requests"
        )
        security_table.add_row(
            "auth_token", auth_token_display, "Bearer token for authentication"
        )
        security_table.add_row(
            "cors_origins", cors_origins_display, "Allowed CORS origins"
        )

        # Docker configuration table
        docker_table = Table(
            title="Docker Configuration", show_header=True, header_style="bold magenta"
        )
        docker_table.add_column("Setting", style="cyan", width=20)
        docker_table.add_column("Value", style="green")
        docker_table.add_column("Description", style="dim")

        docker_table.add_row(
            "docker_image",
            settings.docker_settings.docker_image,
            "Docker image for Claude commands",
        )
        docker_table.add_row(
            "docker_home_directory",
            settings.docker_settings.docker_home_directory or "[dim]Auto-detect[/dim]",
            "Host directory for container home",
        )
        docker_table.add_row(
            "docker_workspace_directory",
            settings.docker_settings.docker_workspace_directory
            or "[dim]Auto-detect[/dim]",
            "Host directory for workspace",
        )

        # Docker volumes
        if settings.docker_settings.docker_volumes:
            volumes_text = "\n".join(settings.docker_settings.docker_volumes)
            docker_table.add_row("docker_volumes", volumes_text, "Docker volume mounts")
        else:
            docker_table.add_row(
                "docker_volumes", "[dim]None[/dim]", "Docker volume mounts"
            )

        # Docker environment variables
        if settings.docker_settings.docker_environment:
            env_text = "\n".join(
                [
                    f"{k}={v}"
                    for k, v in settings.docker_settings.docker_environment.items()
                ]
            )
            docker_table.add_row(
                "docker_environment", env_text, "Docker environment variables"
            )
        else:
            docker_table.add_row(
                "docker_environment", "[dim]None[/dim]", "Docker environment variables"
            )

        # Additional docker args
        if settings.docker_settings.docker_additional_args:
            args_text = " ".join(settings.docker_settings.docker_additional_args)
            docker_table.add_row(
                "docker_additional_args", args_text, "Extra Docker run arguments"
            )
        else:
            docker_table.add_row(
                "docker_additional_args",
                "[dim]None[/dim]",
                "Extra Docker run arguments",
            )

        # User mapping settings
        docker_table.add_row(
            "user_mapping_enabled",
            str(settings.docker_settings.user_mapping_enabled),
            "Enable/disable UID/GID mapping",
        )

        uid_display = (
            str(settings.docker_settings.user_uid)
            if settings.docker_settings.user_uid is not None
            else "[dim]Auto-detect[/dim]"
        )
        docker_table.add_row("user_uid", uid_display, "User ID for container")

        gid_display = (
            str(settings.docker_settings.user_gid)
            if settings.docker_settings.user_gid is not None
            else "[dim]Auto-detect[/dim]"
        )
        docker_table.add_row("user_gid", gid_display, "Group ID for container")

        # Display all tables
        console.print(
            Panel.fit(
                f"[bold]Claude Code Proxy API Configuration[/bold]\n[dim]Version: {__version__}[/dim]",
                border_style="blue",
            )
        )
        console.print()
        console.print(server_table)
        console.print()
        console.print(claude_table)
        console.print()
        console.print(security_table)
        console.print()
        console.print(docker_table)

        # Show configuration file sources
        console.print()
        info_text = Text()
        info_text.append("Configuration loaded from: ", style="bold")
        info_text.append(
            "environment variables, .env file, and TOML configuration files",
            style="dim",
        )
        console.print(
            Panel(info_text, title="Configuration Sources", border_style="green")
        )

    except Exception as e:
        console = Console()
        console.print(f"[bold red]Error loading configuration:[/bold red] {e}")
        raise typer.Exit(1) from e


@app.command(name="init")
def config_init(
    format: str = typer.Option(
        "toml",
        "--format",
        "-f",
        help="Configuration file format (toml, json, or yaml)",
    ),
    output_dir: Path | None = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Output directory for example config files (default: XDG_CONFIG_HOME/ccproxy)",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Overwrite existing configuration files",
    ),
) -> None:
    """Generate example configuration files.

    This command creates example configuration files with all available options
    and documentation comments.

    Examples:
        ccproxy config init                      # Create TOML config in default location
        ccproxy config init --format json        # Create JSON config
        ccproxy config init --format yaml        # Create YAML config
        ccproxy config init --output-dir ./config  # Create in specific directory
    """
    # Validate format
    valid_formats = ["toml", "json", "yaml"]
    if format not in valid_formats:
        typer.echo(
            f"Error: Invalid format '{format}'. Must be one of: {', '.join(valid_formats)}",
            err=True,
        )
        raise typer.Exit(1)

    try:
        from claude_code_proxy.utils.xdg import get_ccproxy_config_dir

        # Determine output directory
        if output_dir is None:
            output_dir = get_ccproxy_config_dir()

        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate example configuration
        example_config = {
            "host": "127.0.0.1",
            "port": 8000,
            "log_level": "INFO",
            "workers": 4,
            "reload": False,
            "cors_origins": ["*"],
            "auth_token": None,
            "tools_handling": "warning",
            "claude_cli_path": None,
            "docker_settings": {
                "docker_image": "claude-code-proxy",
                "docker_volumes": [],
                "docker_environment": {},
                "docker_additional_args": [],
                "docker_home_directory": None,
                "docker_workspace_directory": None,
                "user_mapping_enabled": True,
                "user_uid": None,
                "user_gid": None,
            },
            "pool_settings": {
                "enabled": True,
                "min_size": 2,
                "max_size": 10,
                "idle_timeout": 300,
                "warmup_on_startup": True,
                "health_check_interval": 60,
                "acquire_timeout": 5.0,
            },
        }

        # Determine output file name
        if format == "toml":
            output_file = output_dir / "config.toml"
            if output_file.exists() and not force:
                typer.echo(
                    f"Error: {output_file} already exists. Use --force to overwrite.",
                    err=True,
                )
                raise typer.Exit(1)

            # Write TOML with comments
            with output_file.open("w", encoding="utf-8") as f:
                f.write("# Claude Code Proxy API Configuration\n")
                f.write("# This file configures the ccproxy server settings\n\n")

                f.write("# Server configuration\n")
                f.write('host = "127.0.0.1"  # Server host address\n')
                f.write("port = 8000  # Server port number\n")
                f.write(
                    'log_level = "INFO"  # Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)\n'
                )
                f.write("workers = 4  # Number of worker processes\n")
                f.write("reload = false  # Enable auto-reload for development\n\n")

                f.write("# Security configuration\n")
                f.write('cors_origins = ["*"]  # CORS allowed origins\n')
                f.write(
                    '# auth_token = "your-secret-token"  # Bearer token for API authentication (optional)\n'
                )
                f.write(
                    'tools_handling = "warning"  # How to handle tools in requests (error, warning, ignore)\n\n'
                )

                f.write("# Claude CLI configuration\n")
                f.write(
                    '# claude_cli_path = "/path/to/claude"  # Path to Claude CLI executable (auto-detect if not set)\n\n'
                )

                f.write("# Docker configuration\n")
                f.write("[docker_settings]\n")
                f.write(
                    'docker_image = "claude-code-proxy"  # Docker image for Claude commands\n'
                )
                f.write(
                    "docker_volumes = []  # Volume mounts in 'host:container[:options]' format\n"
                )
                f.write(
                    "docker_environment = {}  # Environment variables for Docker container\n"
                )
                f.write(
                    "docker_additional_args = []  # Additional Docker run arguments\n"
                )
                f.write(
                    '# docker_home_directory = "/path/to/home"  # Host directory for container home\n'
                )
                f.write(
                    '# docker_workspace_directory = "/path/to/workspace"  # Host directory for workspace\n'
                )
                f.write("user_mapping_enabled = true  # Enable UID/GID mapping\n")
                f.write(
                    "# user_uid = 1000  # User ID for container (auto-detect if not set)\n"
                )
                f.write(
                    "# user_gid = 1000  # Group ID for container (auto-detect if not set)\n\n"
                )

                f.write("# Connection pool configuration\n")
                f.write("[pool_settings]\n")
                f.write("enabled = true  # Enable connection pooling\n")
                f.write("min_size = 2  # Minimum pool size\n")
                f.write("max_size = 10  # Maximum pool size\n")
                f.write("idle_timeout = 300  # Seconds before idle connections close\n")
                f.write("warmup_on_startup = true  # Pre-create minimum instances\n")
                f.write("health_check_interval = 60  # Seconds between health checks\n")
                f.write("acquire_timeout = 5.0  # Max seconds to wait for instance\n")

        elif format == "json":
            output_file = output_dir / "config.json"
            if output_file.exists() and not force:
                typer.echo(
                    f"Error: {output_file} already exists. Use --force to overwrite.",
                    err=True,
                )
                raise typer.Exit(1)

            # Write JSON with pretty formatting
            with output_file.open("w", encoding="utf-8") as f:
                json.dump(example_config, f, indent=2)
                f.write("\n")

        elif format == "yaml":
            try:
                import yaml  # type: ignore[import-untyped]
            except ImportError as e:
                typer.echo(
                    "Error: YAML support is not available. Install with: pip install pyyaml",
                    err=True,
                )
                raise typer.Exit(1) from e

            output_file = output_dir / "config.yaml"
            if output_file.exists() and not force:
                typer.echo(
                    f"Error: {output_file} already exists. Use --force to overwrite.",
                    err=True,
                )
                raise typer.Exit(1)

            # Write YAML with comments
            with output_file.open("w", encoding="utf-8") as f:
                f.write("# Claude Code Proxy API Configuration\n")
                f.write("# This file configures the ccproxy server settings\n\n")
                yaml.dump(example_config, f, default_flow_style=False, sort_keys=False)

        typer.echo(f"Created example configuration file: {output_file}")
        typer.echo("")
        typer.echo("To use this configuration:")
        typer.echo(f"  ccproxy --config {output_file} api")
        typer.echo("")
        typer.echo("Or set the CONFIG_FILE environment variable:")
        typer.echo(f"  export CONFIG_FILE={output_file}")
        typer.echo("  ccproxy api")

    except Exception as e:
        typer.echo(f"Error creating configuration file: {e}", err=True)
        raise typer.Exit(1) from e


@app.command(name="schema")
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
    try:
        # Generate schema files
        if output_dir is None:
            output_dir = Path.cwd()

        typer.echo("Generating JSON Schema files for TOML configuration...")

        generated_files = generate_schema_files(output_dir)

        for file_path in generated_files:
            typer.echo(f"Generated: {file_path}")

        if taplo:
            typer.echo("Generating taplo configuration...")
            taplo_config = generate_taplo_config(output_dir)
            typer.echo(f"Generated: {taplo_config}")

        typer.echo("")
        typer.echo("Schema files generated successfully!")
        typer.echo("")
        typer.echo("To use in VS Code:")
        typer.echo("1. Install the 'Even Better TOML' extension")
        typer.echo("2. The schema will be automatically applied to ccproxy TOML files")
        typer.echo("")
        typer.echo("To use with taplo CLI:")
        if taplo:
            typer.echo("  taplo check your-config.toml")
        else:
            typer.echo("  ccproxy config schema --taplo  # Generate taplo config first")
            typer.echo("  taplo check your-config.toml")

    except Exception as e:
        typer.echo(f"Error generating schema: {e}", err=True)
        raise typer.Exit(1) from e


@app.command(name="validate")
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
    try:
        # Validate the config file
        if not config_file.exists():
            typer.echo(f"Error: File {config_file} does not exist.", err=True)
            raise typer.Exit(1)

        typer.echo(f"Validating {config_file}...")

        try:
            is_valid = validate_config_with_schema(config_file)
            if is_valid:
                typer.echo("✓ Configuration file is valid according to schema.")
            else:
                typer.echo("✗ Configuration file validation failed.", err=True)
                raise typer.Exit(1)
        except ImportError as e:
            typer.echo(f"Error: {e}", err=True)
            typer.echo(
                "Install check-jsonschema: pip install check-jsonschema", err=True
            )
            raise typer.Exit(1) from e
        except Exception as e:
            typer.echo(f"Validation error: {e}", err=True)
            raise typer.Exit(1) from e

    except Exception as e:
        typer.echo(f"Error validating configuration: {e}", err=True)
        raise typer.Exit(1) from e
