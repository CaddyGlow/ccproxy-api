"""Config command for Claude Code Proxy API."""

import json
import secrets
import tempfile
import tomllib
from pathlib import Path
from typing import Any, Optional

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
            "tools_handling",
            settings.api_tools_handling,
            "How to handle tools in requests",
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
            "workers": 1,
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

                # Pool settings removed - connection pooling functionality has been removed

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


@app.command(name="generate-token")
def generate_token(
    save: bool = typer.Option(
        False,
        "--save",
        "--write",
        help="Save the token to configuration file",
    ),
    config_file: Path | None = typer.Option(
        None,
        "--config-file",
        "-c",
        help="Configuration file to update (default: auto-detect or create .ccproxy.toml)",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Overwrite existing auth_token without confirmation",
    ),
) -> None:
    """Generate a secure random token for API authentication.

    This command generates a secure authentication token that can be used with
    both Anthropic and OpenAI compatible APIs.

    Use --save to write the token to a configuration file. The command supports
    TOML, JSON, and YAML formats and will auto-detect the format from the file extension.

    Examples:
        ccproxy config generate-token                    # Generate and display token
        ccproxy config generate-token --save             # Generate and save to config
        ccproxy config generate-token --save --config-file custom.toml  # Save to TOML config
        ccproxy config generate-token --save --config-file config.json  # Save to JSON config
        ccproxy config generate-token --save --config-file config.yaml  # Save to YAML config
        ccproxy config generate-token --save --force     # Overwrite existing token
    """
    try:
        # Generate a secure token
        token = secrets.token_urlsafe(32)

        console = Console()

        # Display the generated token
        console.print()
        console.print(
            Panel.fit(
                f"[bold green]Generated Authentication Token[/bold green]\n[dim]Token: [/dim][bold]{token}[/bold]",
                border_style="green",
            )
        )
        console.print()

        # Show environment variable commands - server first, then clients
        console.print("[bold]Server Environment Variables:[/bold]")
        console.print(f"[cyan]export AUTH_TOKEN={token}[/cyan]")
        console.print()

        console.print("[bold]Client Environment Variables:[/bold]")
        console.print()

        console.print("[dim]For Anthropic Python SDK clients:[/dim]")
        console.print(f"[cyan]export ANTHROPIC_API_KEY={token}[/cyan]")
        console.print("[cyan]export ANTHROPIC_BASE_URL=http://localhost:8000[/cyan]")
        console.print()

        console.print("[dim]For OpenAI Python SDK clients:[/dim]")
        console.print(f"[cyan]export OPENAI_API_KEY={token}[/cyan]")
        console.print(
            "[cyan]export OPENAI_BASE_URL=http://localhost:8000/openai[/cyan]"
        )
        console.print()

        console.print("[bold]For .env file:[/bold]")
        console.print(f"[cyan]AUTH_TOKEN={token}[/cyan]")
        console.print()

        console.print("[bold]Usage with curl (using environment variables):[/bold]")
        console.print("[dim]Anthropic API:[/dim]")
        console.print('[cyan]curl -H "x-api-key: $ANTHROPIC_API_KEY" \\\\[/cyan]')
        console.print('[cyan]     -H "Content-Type: application/json" \\\\[/cyan]')
        console.print('[cyan]     "$ANTHROPIC_BASE_URL/v1/messages"[/cyan]')
        console.print()
        console.print("[dim]OpenAI API:[/dim]")
        console.print(
            '[cyan]curl -H "Authorization: Bearer $OPENAI_API_KEY" \\\\[/cyan]'
        )
        console.print('[cyan]     -H "Content-Type: application/json" \\\\[/cyan]')
        console.print('[cyan]     "$OPENAI_BASE_URL/v1/chat/completions"[/cyan]')
        console.print()

        # Mention the save functionality if not using it
        if not save:
            console.print(
                "[dim]Tip: Use --save to write this token to a configuration file[/dim]"
            )
            console.print()

        # Save to config file if requested
        if save:
            # Determine config file path
            if config_file is None:
                # Try to find existing config file or create default
                from claude_code_proxy.utils import find_toml_config_file

                config_file = find_toml_config_file()

                if config_file is None:
                    # Create default config file in current directory
                    config_file = Path(".ccproxy.toml")

            console.print(
                f"[bold]Saving token to configuration file:[/bold] {config_file}"
            )

            # Detect file format from extension
            file_format = _detect_config_format(config_file)
            console.print(f"[dim]Detected format: {file_format.upper()}[/dim]")

            # Read existing config or create new one using existing Settings functionality
            config_data = {}
            existing_token = None

            if config_file.exists():
                try:
                    from claude_code_proxy.config.settings import Settings

                    config_data = Settings.load_config_file(config_file)
                    existing_token = config_data.get("auth_token")
                    console.print("[dim]Found existing configuration file[/dim]")
                except Exception as e:
                    console.print(
                        f"[yellow]Warning: Could not read existing config file: {e}[/yellow]"
                    )
                    console.print("[dim]Will create new configuration file[/dim]")
            else:
                console.print("[dim]Will create new configuration file[/dim]")

            # Check for existing token and ask for confirmation if needed
            if existing_token and not force:
                console.print()
                console.print(
                    "[yellow]Warning: Configuration file already contains an auth_token[/yellow]"
                )
                console.print(f"[dim]Current token: {existing_token[:16]}...[/dim]")
                console.print(f"[dim]New token: {token[:16]}...[/dim]")
                console.print()

                if not typer.confirm("Do you want to overwrite the existing token?"):
                    console.print("[dim]Token generation cancelled[/dim]")
                    return

            # Update auth_token in config
            config_data["auth_token"] = token

            # Write updated config in the appropriate format
            _write_config_file(config_file, config_data, file_format)

            console.print(f"[green]✓[/green] Token saved to {config_file}")
            console.print()
            console.print("[bold]To use this configuration:[/bold]")
            console.print(f"[cyan]ccproxy --config {config_file} api[/cyan]")
            console.print()
            console.print("[dim]Or set CONFIG_FILE environment variable:[/dim]")
            console.print(f"[cyan]export CONFIG_FILE={config_file}[/cyan]")
            console.print("[cyan]ccproxy api[/cyan]")

    except Exception as e:
        typer.echo(f"Error generating token: {e}", err=True)
        raise typer.Exit(1) from e


def _detect_config_format(config_file: Path) -> str:
    """Detect configuration file format from extension."""
    suffix = config_file.suffix.lower()
    if suffix in [".toml"]:
        return "toml"
    elif suffix in [".json"]:
        return "json"
    elif suffix in [".yaml", ".yml"]:
        return "yaml"
    else:
        # Default to TOML if unknown extension
        return "toml"


def _write_json_config(config_file: Path, config_data: dict[str, Any]) -> None:
    """Write configuration data to a JSON file with proper formatting."""
    with config_file.open("w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=2, sort_keys=True)
        f.write("\n")


def _write_yaml_config(config_file: Path, config_data: dict[str, Any]) -> None:
    """Write configuration data to a YAML file with proper formatting."""
    try:
        import yaml

        with config_file.open("w", encoding="utf-8") as f:
            f.write("# Claude Code Proxy API Configuration\n")
            f.write("# Generated by ccproxy config generate-token\n\n")
            yaml.dump(
                config_data, f, default_flow_style=False, sort_keys=True, indent=2
            )
    except ImportError as e:
        raise ValueError(
            "YAML support not available. Install with: pip install pyyaml"
        ) from e


def _write_config_file(
    config_file: Path, config_data: dict[str, Any], file_format: str
) -> None:
    """Write configuration data to file in the specified format."""
    if file_format == "toml":
        _write_toml_config(config_file, config_data)
    elif file_format == "json":
        _write_json_config(config_file, config_data)
    elif file_format == "yaml":
        _write_yaml_config(config_file, config_data)
    else:
        raise ValueError(f"Unsupported config format: {file_format}")


def _write_toml_config(config_file: Path, config_data: dict[str, Any]) -> None:
    """Write configuration data to a TOML file with proper formatting."""
    try:
        # Create a nicely formatted TOML file
        with config_file.open("w", encoding="utf-8") as f:
            f.write("# Claude Code Proxy API Configuration\n")
            f.write("# Generated by ccproxy config generate-token\n\n")

            # Write server settings
            if any(
                key in config_data
                for key in ["host", "port", "log_level", "workers", "reload"]
            ):
                f.write("# Server configuration\n")
                if "host" in config_data:
                    f.write(f'host = "{config_data["host"]}"\n')
                if "port" in config_data:
                    f.write(f"port = {config_data['port']}\n")
                if "log_level" in config_data:
                    f.write(f'log_level = "{config_data["log_level"]}"\n')
                if "workers" in config_data:
                    f.write(f"workers = {config_data['workers']}\n")
                if "reload" in config_data:
                    f.write(f"reload = {str(config_data['reload']).lower()}\n")
                f.write("\n")

            # Write security settings
            if any(
                key in config_data
                for key in ["auth_token", "cors_origins", "tools_handling"]
            ):
                f.write("# Security configuration\n")
                if "auth_token" in config_data:
                    f.write(f'auth_token = "{config_data["auth_token"]}"\n')
                if "cors_origins" in config_data:
                    origins = config_data["cors_origins"]
                    if isinstance(origins, list):
                        origins_str = '", "'.join(origins)
                        f.write(f'cors_origins = ["{origins_str}"]\n')
                    else:
                        f.write(f'cors_origins = ["{origins}"]\n')
                if "tools_handling" in config_data:
                    f.write(f'tools_handling = "{config_data["tools_handling"]}"\n')
                f.write("\n")

            # Write Claude CLI configuration
            if "claude_cli_path" in config_data:
                f.write("# Claude CLI configuration\n")
                if config_data["claude_cli_path"]:
                    f.write(f'claude_cli_path = "{config_data["claude_cli_path"]}"\n')
                else:
                    f.write(
                        '# claude_cli_path = "/path/to/claude"  # Auto-detect if not set\n'
                    )
                f.write("\n")

            # Write Docker settings
            if "docker_settings" in config_data:
                docker_settings = config_data["docker_settings"]
                f.write("# Docker configuration\n")
                f.write("[docker_settings]\n")

                for key, value in docker_settings.items():
                    if isinstance(value, str):
                        f.write(f'{key} = "{value}"\n')
                    elif isinstance(value, bool):
                        f.write(f"{key} = {str(value).lower()}\n")
                    elif isinstance(value, int | float):
                        f.write(f"{key} = {value}\n")
                    elif isinstance(value, list):
                        if value:  # Only write non-empty lists
                            if all(isinstance(item, str) for item in value):
                                items_str = '", "'.join(value)
                                f.write(f'{key} = ["{items_str}"]\n')
                            else:
                                f.write(f"{key} = {value}\n")
                        else:
                            f.write(f"{key} = []\n")
                    elif isinstance(value, dict):
                        if value:  # Only write non-empty dicts
                            f.write(f"{key} = {json.dumps(value)}\n")
                        else:
                            f.write(f"{key} = {{}}\n")
                    elif value is None:
                        f.write(f"# {key} = null  # Not configured\n")
                f.write("\n")

            # Write any remaining top-level settings
            written_keys = {
                "host",
                "port",
                "log_level",
                "workers",
                "reload",
                "auth_token",
                "cors_origins",
                "tools_handling",
                "claude_cli_path",
                "docker_settings",
            }
            remaining_keys = set(config_data.keys()) - written_keys

            if remaining_keys:
                f.write("# Additional settings\n")
                for key in sorted(remaining_keys):
                    value = config_data[key]
                    if isinstance(value, str):
                        f.write(f'{key} = "{value}"\n')
                    elif isinstance(value, bool):
                        f.write(f"{key} = {str(value).lower()}\n")
                    elif isinstance(value, int | float):
                        f.write(f"{key} = {value}\n")
                    elif isinstance(value, list | dict):
                        f.write(f"{key} = {json.dumps(value)}\n")
                    elif value is None:
                        f.write(f"# {key} = null\n")

    except Exception as e:
        raise ValueError(f"Failed to write TOML configuration: {e}") from e
