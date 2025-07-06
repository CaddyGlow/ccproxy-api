"""Main entry point for Claude Proxy API Server."""

import json
import logging
import os
import secrets
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional

import fastapi_cli.discover
import typer
import uvicorn
from click import get_current_context
from fastapi_cli.cli import _run
from fastapi_cli.cli import app as fastapi_app
from fastapi_cli.exceptions import FastAPICLIException
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from claude_code_proxy._version import __version__
from claude_code_proxy.config.settings import get_settings
from claude_code_proxy.utils.docker_builder import DockerCommandBuilder
from claude_code_proxy.utils.helper import get_package_dir
from claude_code_proxy.utils.schema import (
    generate_schema_files,
    generate_taplo_config,
    validate_config_with_schema,
)


def version_callback(value: bool) -> None:
    """Print version and exit."""
    if value:
        typer.echo(f"claude-code-proxy-api {__version__}")
        raise typer.Exit()


app = typer.Typer(
    rich_markup_mode="rich",
    add_completion=False,
    no_args_is_help=True,
    pretty_exceptions_enable=False,
)
logger = logging.getLogger(__name__)


# Add global --version option
@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        "-V",
        callback=version_callback,
        is_eager=True,
        help="Show version and exit.",
    ),
    config: Path | None = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to configuration file (TOML, JSON, or YAML)",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
) -> None:
    """Claude Code Proxy API Server - Anthropic and OpenAI compatible interface for Claude."""
    # Store config path in context for use by commands
    try:
        ctx = get_current_context()
        ctx.ensure_object(dict)
        ctx.obj["config_path"] = config
    except RuntimeError:
        # No active click context (e.g., in tests)
        pass


# Remove the fastapi callback to avoid the warning
# fastapi_app.callback()(lambda: None)
fastapi_app.callback()(None)  # type: ignore[type-var]
# Register fastapi app with typer
app.add_typer(fastapi_app)


def get_default_path_hook() -> Path:
    app_entry_path = get_package_dir() / "claude_code_proxy" / "main.py"
    if app_entry_path.is_file():
        return app_entry_path

    raise FastAPICLIException(
        "Could not find a default file to run, please provide an explicit path"
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


fastapi_cli.discover.get_default_path = get_default_path_hook


@app.command()
def config() -> None:
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


@app.command()
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


@app.command()
def generate_token() -> None:
    """Generate a secure random token for API authentication."""
    token = secrets.token_urlsafe(32)
    typer.echo("Generated authentication token:")
    typer.echo(f"AUTH_TOKEN={token}")
    typer.echo("")
    typer.echo("Add this to your environment variables:")
    typer.echo(f"export AUTH_TOKEN={token}")
    typer.echo("")
    typer.echo("Or add to your .env file:")
    typer.echo(f"AUTH_TOKEN={token}")


@app.command()
def schema(
    output_dir: Path | None = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Output directory for schema files (default: current directory)",
    ),
    validate: Path | None = typer.Option(
        None,
        "--validate",
        "-v",
        help="Validate a configuration file (TOML/JSON/YAML) against the schema",
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
        ccproxy schema                    # Generate schema files in current directory
        ccproxy schema --output-dir ./schemas  # Generate in specific directory
        ccproxy schema --taplo           # Also generate taplo config
        ccproxy schema --validate config.toml  # Validate a config file
        ccproxy schema --validate config.yaml  # Validate a YAML config
        ccproxy schema --validate config.json  # Validate a JSON config
    """
    try:
        if validate:
            # Validate a TOML file
            if not validate.exists():
                typer.echo(f"Error: File {validate} does not exist.", err=True)
                raise typer.Exit(1)

            typer.echo(f"Validating {validate}...")

            try:
                is_valid = validate_config_with_schema(validate)
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
        else:
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
            typer.echo(
                "2. The schema will be automatically applied to ccproxy TOML files"
            )
            typer.echo("")
            typer.echo("To use with taplo CLI:")
            if taplo:
                typer.echo("  taplo check your-config.toml")
            else:
                typer.echo("  ccproxy schema --taplo  # Generate taplo config first")
                typer.echo("  taplo check your-config.toml")

    except Exception as e:
        typer.echo(f"Error generating schema: {e}", err=True)
        raise typer.Exit(1) from e


@app.command()
def api(
    docker: bool = typer.Option(
        False,
        "--docker",
        "-d",
        help="Run API server using Docker instead of local execution",
    ),
    port: int = typer.Option(
        8000,
        "--port",
        "-p",
        help="Port to run the server on",
    ),
    host: str = typer.Option(
        "127.0.0.1",
        "--host",
        "-h",
        help="Host to bind the server to",
    ),
    reload: bool = typer.Option(
        False,
        "--reload",
        help="Enable auto-reload for development",
    ),
    docker_image: str | None = typer.Option(
        None,
        "--docker-image",
        help="Docker image to use (overrides config)",
    ),
    docker_env: list[str] = typer.Option(
        [],
        "--docker-env",
        help="Environment variables to pass to Docker (KEY=VALUE format, can be used multiple times)",
    ),
    docker_volume: list[str] = typer.Option(
        [],
        "--docker-volume",
        help="Volume mounts to add (host:container[:options] format, can be used multiple times)",
    ),
    docker_arg: list[str] = typer.Option(
        [],
        "--docker-arg",
        help="Additional Docker run arguments (can be used multiple times)",
    ),
    docker_home: str | None = typer.Option(
        None,
        "--docker-home",
        help="Home directory inside Docker container (overrides config)",
    ),
    docker_workspace: str | None = typer.Option(
        None,
        "--docker-workspace",
        help="Workspace directory inside Docker container (overrides config)",
    ),
    user_mapping_enabled: bool | None = typer.Option(
        None,
        "--user-mapping/--no-user-mapping",
        help="Enable/disable UID/GID mapping (overrides config)",
    ),
    user_uid: int | None = typer.Option(
        None,
        "--user-uid",
        help="User ID to run container as (overrides config)",
        min=0,
    ),
    user_gid: int | None = typer.Option(
        None,
        "--user-gid",
        help="Group ID to run container as (overrides config)",
        min=0,
    ),
) -> None:
    """
    Start the Claude Code Proxy API server.

    This command starts the API server either locally or in Docker.
    The server provides both Anthropic and OpenAI-compatible endpoints.

    Examples:
        ccproxy run
        ccproxy run --port 8080 --reload
        ccproxy run --docker
        ccproxy run --docker --docker-image custom:latest --port 8080
    """
    try:
        if docker:
            # Load settings to get Docker configuration
            settings = get_settings(config_path=get_config_path_from_context())
            port = port if port is None else settings.port
            # Prepare server command using fastapi
            server_args = [
                "run",
                "--host",
                "0.0.0.0",  # Docker needs to bind to 0.0.0.0
                "--port",
                str(port),
            ]

            if reload:
                server_args.append("--reload")

            # Build and execute Docker command with settings and CLI overrides
            typer.echo("Starting Claude Code Proxy API server with Docker...")
            typer.echo(f"Server will be available at: http://{host}:{port}")

            # Show the command before executing
            docker_cmd = DockerCommandBuilder.from_settings_and_overrides(
                settings.docker_settings,
                docker_image=docker_image,
                docker_env=docker_env + [f"PORT={port}"],
                docker_volume=docker_volume,
                docker_arg=docker_arg + ["-p", f"{port}:{port}"],
                docker_home=docker_home,
                docker_workspace=docker_workspace,
                user_mapping_enabled=user_mapping_enabled,
                user_uid=user_uid,
                user_gid=user_gid,
                cmd_args=server_args,
            )
            typer.echo(f"Executing: {' '.join(docker_cmd)}")
            typer.echo("")

            # Execute using the new Docker builder method
            DockerCommandBuilder.execute_from_settings(
                settings.docker_settings,
                docker_image=docker_image,
                docker_env=docker_env + [f"PORT={port}"],
                docker_volume=docker_volume,
                docker_arg=docker_arg + ["-p", f"{port}:{port}"],
                docker_home=docker_home,
                docker_workspace=docker_workspace,
                user_mapping_enabled=user_mapping_enabled,
                user_uid=user_uid,
                user_gid=user_gid,
                cmd_args=server_args,
            )
        else:
            # Run server locally using fastapi-cli's _run function
            typer.echo("Starting Claude Code Proxy API server locally...")
            typer.echo(f"Server will be available at: http://{host}:{port}")
            typer.echo("")

            # Use fastapi-cli's internal _run function
            _run(
                command="production",
                path=get_default_path_hook(),
                host=host,
                port=port,
                reload=reload,
            )

    except Exception as e:
        typer.echo(f"Error starting server: {e}", err=True)
        raise typer.Exit(1) from e


@app.command()
def claude(
    args: list[str] | None = typer.Argument(
        default=None,
        help="Arguments to pass to claude CLI (e.g. --version, doctor, config)",
    ),
    docker: bool = typer.Option(
        False,
        "--docker",
        "-d",
        help="Run claude command from docker image instead of local CLI",
    ),
    docker_image: str | None = typer.Option(
        None,
        "--docker-image",
        help="Docker image to use (overrides config)",
    ),
    docker_env: list[str] = typer.Option(
        [],
        "--docker-env",
        help="Environment variables to pass to Docker (KEY=VALUE format, can be used multiple times)",
    ),
    docker_volume: list[str] = typer.Option(
        [],
        "--docker-volume",
        help="Volume mounts to add (host:container[:options] format, can be used multiple times)",
    ),
    docker_arg: list[str] = typer.Option(
        [],
        "--docker-arg",
        help="Additional Docker run arguments (can be used multiple times)",
    ),
    docker_home: str | None = typer.Option(
        None,
        "--docker-home",
        help="Home directory inside Docker container (overrides config)",
    ),
    docker_workspace: str | None = typer.Option(
        None,
        "--docker-workspace",
        help="Workspace directory inside Docker container (overrides config)",
    ),
    user_mapping_enabled: bool | None = typer.Option(
        None,
        "--user-mapping/--no-user-mapping",
        help="Enable/disable UID/GID mapping (overrides config)",
    ),
    user_uid: int | None = typer.Option(
        None,
        "--user-uid",
        help="User ID to run container as (overrides config)",
        min=0,
    ),
    user_gid: int | None = typer.Option(
        None,
        "--user-gid",
        help="Group ID to run container as (overrides config)",
        min=0,
    ),
) -> None:
    """
    Execute claude CLI commands directly.

    This is a simple pass-through to the claude CLI executable
    found by the settings system or run from docker image.

    Examples:
        ccproxy claude -- --version
        ccproxy claude -- doctor
        ccproxy claude -- config
        ccproxy claude --docker -- --version
        ccproxy claude --docker --docker-image custom:latest -- --version
        ccproxy claude --docker --docker-env API_KEY=sk-... --docker-volume ./data:/data -- chat
    """
    # Handle None args case
    if args is None:
        args = []

    try:
        if docker:
            # Load settings to get Docker configuration
            settings = get_settings(config_path=get_config_path_from_context())

            # Show the command before executing
            docker_cmd = DockerCommandBuilder.from_settings_and_overrides(
                settings.docker_settings,
                docker_image=docker_image,
                docker_env=docker_env,
                docker_volume=docker_volume,
                docker_arg=docker_arg,
                docker_home=docker_home,
                docker_workspace=docker_workspace,
                user_mapping_enabled=user_mapping_enabled,
                user_uid=user_uid,
                user_gid=user_gid,
                command=["claude"],
                cmd_args=args,
            )

            typer.echo(f"Executing: {' '.join(docker_cmd)}")
            typer.echo("")

            # Execute using the new Docker builder method
            DockerCommandBuilder.execute_from_settings(
                settings.docker_settings,
                docker_image=docker_image,
                docker_env=docker_env,
                docker_volume=docker_volume,
                docker_arg=docker_arg,
                docker_home=docker_home,
                docker_workspace=docker_workspace,
                user_mapping_enabled=user_mapping_enabled,
                user_uid=user_uid,
                user_gid=user_gid,
                command=["claude"],
                cmd_args=args,
            )
        else:
            # Load settings to find claude path
            settings = get_settings(config_path=get_config_path_from_context())

            # Get claude path
            claude_path = settings.claude_cli_path
            if not claude_path:
                typer.echo("Error: Claude CLI not found.", err=True)
                typer.echo(
                    "Please install Claude CLI or configure claude_cli_path.", err=True
                )
                raise typer.Exit(1)

            # Resolve to absolute path
            if not Path(claude_path).is_absolute():
                claude_path = str(Path(claude_path).resolve())

            typer.echo(f"Executing: {claude_path} {' '.join(args)}")
            typer.echo("")

            # Execute command directly
            try:
                # Use os.execvp to replace current process with claude
                # This hands over full control to claude, including signal handling
                os.execvp(claude_path, [claude_path] + args)
            except OSError as e:
                typer.echo(f"Failed to execute command: {e}", err=True)
                raise typer.Exit(1) from e

    except Exception as e:
        typer.echo(f"Error executing claude command: {e}", err=True)
        raise typer.Exit(1) from e


if __name__ == "__main__":
    app()
