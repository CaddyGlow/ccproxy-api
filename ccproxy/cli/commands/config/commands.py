"""Main config commands for CCProxy API."""

import json
import secrets
from pathlib import Path
from typing import Any

import structlog
import typer
from click import get_current_context
from pydantic import BaseModel
from pydantic.fields import FieldInfo

from ccproxy.cli.helpers import get_rich_toolkit
from ccproxy.config.settings import Settings
from ccproxy.core._version import __version__
from ccproxy.services.container import ServiceContainer


logger = structlog.get_logger(__name__)


def _get_service_container() -> ServiceContainer:
    """Create a service container for the config commands."""
    settings = Settings.from_config(config_path=get_config_path_from_context())
    return ServiceContainer(settings)


def _create_config_table(title: str, rows: list[tuple[str, str, str]]) -> Any:
    """Create a configuration table with standard styling."""
    from rich.table import Table

    table = Table(title=title, show_header=True, header_style="bold magenta")
    table.add_column("Setting", style="cyan", width=20)
    table.add_column("Value", style="green")
    table.add_column("Description", style="dim")

    for setting, value, description in rows:
        table.add_row(setting, value, description)

    return table


def _format_value(value: Any) -> str:
    """Format a configuration value for display."""
    if value is None:
        return "[dim]Auto-detect[/dim]"
    elif isinstance(value, bool | int | float):
        return str(value)
    elif isinstance(value, str):
        if not value:
            return "[dim]Not set[/dim]"
        if any(
            keyword in value.lower()
            for keyword in ["token", "key", "secret", "password"]
        ):
            return "[green]Set[/green]"
        return value
    elif isinstance(value, list):
        if not value:
            return "[dim]None[/dim]"
        if len(value) == 1:
            return str(value[0])
        return "\n".join(str(item) for item in value)
    elif isinstance(value, dict):
        if not value:
            return "[dim]None[/dim]"
        return "\n".join(f"{k}={v}" for k, v in value.items())
    else:
        return str(value)


def _get_field_description(field_info: FieldInfo) -> str:
    """Get a human-readable description from a Pydantic field."""
    if field_info.description:
        return field_info.description
    return "Configuration setting"


def _generate_config_rows_from_model(
    model: BaseModel, prefix: str = ""
) -> list[tuple[str, str, str]]:
    """Generate configuration rows from a Pydantic model dynamically."""
    rows = []

    for field_name, _field_info in model.model_fields.items():
        field_value = getattr(model, field_name)
        display_name = f"{prefix}{field_name}" if prefix else field_name

        if isinstance(field_value, BaseModel):
            model_name = field_value.__class__.__name__
            rows.append(
                (
                    display_name,
                    f"[dim]{model_name} configuration[/dim]",
                    _get_field_description(_field_info),
                )
            )

            sub_rows = _generate_config_rows_from_model(field_value, f"{display_name}_")
            rows.extend(sub_rows)
        else:
            formatted_value = _format_value(field_value)
            description = _get_field_description(_field_info)
            rows.append((display_name, formatted_value, description))

    return rows


def _group_config_rows(
    rows: list[tuple[str, str, str]],
) -> dict[str, list[tuple[str, str, str]]]:
    """Group configuration rows by their top-level section."""
    groups: dict[str, list[tuple[str, str, str]]] = {}

    for setting, value, description in rows:
        if setting.startswith("server"):
            group_name = "Server Configuration"
        elif setting.startswith("security"):
            group_name = "Security Configuration"
        elif setting.startswith("cors"):
            group_name = "CORS Configuration"
        elif setting.startswith("claude"):
            group_name = "Claude CLI Configuration"
        elif setting.startswith("auth"):
            group_name = "Authentication Configuration"
        elif setting.startswith("docker"):
            group_name = "Docker Configuration"
        elif setting.startswith("observability"):
            group_name = "Observability Configuration"
        elif setting.startswith("scheduler"):
            group_name = "Scheduler Configuration"
        elif setting.startswith("pricing"):
            group_name = "Pricing Configuration"
        else:
            group_name = "General Configuration"

        if group_name not in groups:
            groups[group_name] = []

        clean_setting = setting.split("_", 1)[1] if "_" in setting else setting
        groups[group_name].append((clean_setting, value, description))

    return groups


def get_config_path_from_context() -> Path | None:
    """Get config path from typer context if available."""
    try:
        ctx = get_current_context()
        if ctx and ctx.obj and "config_path" in ctx.obj:
            config_path = ctx.obj["config_path"]
            return config_path if config_path is None else Path(config_path)
    except RuntimeError:
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
    toolkit = get_rich_toolkit()

    try:
        container = _get_service_container()
        settings = container.get_service(Settings)

        from rich.console import Console
        from rich.panel import Panel
        from rich.text import Text

        console = Console()

        all_rows = _generate_config_rows_from_model(settings)

        all_rows.append(
            ("server_url", settings.server_url, "Complete server URL (computed)")
        )

        grouped_rows = _group_config_rows(all_rows)

        console.print(
            Panel.fit(
                f"[bold]CCProxy API Configuration[/bold]\n[dim]Version: {__version__}[/dim]",
                border_style="blue",
            )
        )
        console.print()

        for section_name, section_rows in grouped_rows.items():
            if section_rows:
                table = _create_config_table(section_name, section_rows)
                console.print(table)
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

    except (OSError, PermissionError) as e:
        logger.error("config_list_file_access_error", error=str(e), exc_info=e)
        toolkit.print(f"Error accessing configuration files: {e}", tag="error")
        raise typer.Exit(1) from e
    except (json.JSONDecodeError, ValueError) as e:
        logger.error("config_list_parsing_error", error=str(e), exc_info=e)
        toolkit.print(f"Configuration parsing error: {e}", tag="error")
        raise typer.Exit(1) from e
    except ImportError as e:
        logger.error("config_list_import_error", error=str(e), exc_info=e)
        toolkit.print(f"Module import error: {e}", tag="error")
        raise typer.Exit(1) from e
    except Exception as e:
        logger.error("config_list_unexpected_error", error=str(e), exc_info=e)
        toolkit.print(f"Error loading configuration: {e}", tag="error")
        raise typer.Exit(1) from e


@app.command(name="init")
def config_init(
    format: str = typer.Option(
        "toml",
        "--format",
        "-f",
        help="Configuration file format (only toml is supported)",
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
    """Generate example configuration files."""
    if format != "toml":
        toolkit = get_rich_toolkit()
        toolkit.print(
            f"Error: Invalid format '{format}'. Only 'toml' format is supported.",
            tag="error",
        )
        raise typer.Exit(1)

    toolkit = get_rich_toolkit()

    try:
        from ccproxy.config.utils import get_ccproxy_config_dir

        if output_dir is None:
            output_dir = get_ccproxy_config_dir()

        output_dir.mkdir(parents=True, exist_ok=True)

        example_config = _generate_default_config_from_model(Settings)

        if format == "toml":
            output_file = output_dir / "config.toml"
            if output_file.exists() and not force:
                toolkit.print(
                    f"Error: {output_file} already exists. Use --force to overwrite.",
                    tag="error",
                )
                raise typer.Exit(1)

            _write_toml_config_with_comments(output_file, example_config, Settings)

        toolkit.print(
            f"Created example configuration file: {output_file}", tag="success"
        )
        toolkit.print_line()
        toolkit.print("To use this configuration:", tag="info")
        toolkit.print(f"  ccproxy --config {output_file} api", tag="command")
        toolkit.print_line()
        toolkit.print("Or set the CONFIG_FILE environment variable:", tag="info")
        toolkit.print(f"  export CONFIG_FILE={output_file}", tag="command")
        toolkit.print("  ccproxy api", tag="command")

    except (OSError, PermissionError) as e:
        logger.error("config_init_file_access_error", error=str(e), exc_info=e)
        toolkit.print(
            f"Error creating configuration file (permission/IO error): {e}", tag="error"
        )
        raise typer.Exit(1) from e
    except ImportError as e:
        logger.error("config_init_import_error", error=str(e), exc_info=e)
        toolkit.print(f"Module import error: {e}", tag="error")
        raise typer.Exit(1) from e
    except ValueError as e:
        logger.error("config_init_value_error", error=str(e), exc_info=e)
        toolkit.print(f"Configuration value error: {e}", tag="error")
        raise typer.Exit(1) from e
    except Exception as e:
        logger.error("config_init_unexpected_error", error=str(e), exc_info=e)
        toolkit.print(f"Error creating configuration file: {e}", tag="error")
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
    """Generate a secure random token for API authentication."""
    toolkit = get_rich_toolkit()

    try:
        token = secrets.token_urlsafe(32)

        from rich.console import Console
        from rich.panel import Panel

        console = Console()

        console.print()
        console.print(
            Panel.fit(
                f"[bold green]Generated Authentication Token[/bold green]\n[dim]Token: [/dim][bold]{token}[/bold]",
                border_style="green",
            )
        )
        console.print()

        console.print("[bold]Server Environment Variables:[/bold]")
        console.print(f"[cyan]export SECURITY__AUTH_TOKEN={token}[/cyan]")
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
        console.print(f"[cyan]SECURITY__AUTH_TOKEN={token}[/cyan]")
        console.print()

        console.print("[bold]Usage with curl (using environment variables):[/bold]")
        console.print("[dim]Anthropic API:[/dim]")
        console.print(r'[cyan]curl -H "x-api-key: $ANTHROPIC_API_KEY" \ [/cyan]')
        console.print(r'[cyan]     -H "Content-Type: application/json" \ [/cyan]')
        console.print('[cyan]     "$ANTHROPIC_BASE_URL/v1/messages"[/cyan]')
        console.print()
        console.print("[dim]OpenAI API:[/dim]")
        console.print(
            r'[cyan]curl -H "Authorization: Bearer $OPENAI_API_KEY" \ [/cyan]'
        )
        console.print(r'[cyan]     -H "Content-Type: application/json" \ [/cyan]')
        console.print('[cyan]     "$OPENAI_BASE_URL/v1/chat/completions"[/cyan]')
        console.print()

        if not save:
            console.print(
                "[dim]Tip: Use --save to write this token to a configuration file[/dim]"
            )
            console.print()

        if save:
            if config_file is None:
                from ccproxy.config.utils import find_toml_config_file

                config_file = find_toml_config_file()

                if config_file is None:
                    config_file = Path(".ccproxy.toml")

            console.print(
                f"[bold]Saving token to configuration file:[/bold] {config_file}"
            )

            file_format = _detect_config_format(config_file)
            console.print(f"[dim]Detected format: {file_format.upper()}[/dim]")

            config_data = {}
            existing_token = None

            if config_file.exists():
                try:
                    from ccproxy.config.settings import Settings

                    config_data = Settings.load_config_file(config_file)
                    existing_token = config_data.get("auth_token")
                    console.print("[dim]Found existing configuration file[/dim]")
                except (OSError, PermissionError) as e:
                    logger.warning(
                        "generate_token_config_file_access_error",
                        error=str(e),
                        exc_info=e,
                    )
                    console.print(
                        f"[yellow]Warning: Could not access existing config file: {e}[/yellow]"
                    )
                    console.print("[dim]Will create new configuration file[/dim]")
                except (json.JSONDecodeError, ValueError) as e:
                    logger.warning(
                        "generate_token_config_file_parse_error",
                        error=str(e),
                        exc_info=e,
                    )
                    console.print(
                        f"[yellow]Warning: Could not parse existing config file: {e}[/yellow]"
                    )
                    console.print("[dim]Will create new configuration file[/dim]")
                except Exception as e:
                    logger.warning(
                        "generate_token_config_file_read_error",
                        error=str(e),
                        exc_info=e,
                    )
                    console.print(
                        f"[yellow]Warning: Could not read existing config file: {e}[/yellow]"
                    )
                    console.print("[dim]Will create new configuration file[/dim]")
            else:
                console.print("[dim]Will create new configuration file[/dim]")

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

            config_data["auth_token"] = token

            _write_config_file(config_file, config_data, file_format)

            console.print(f"[green]✓[/green] Token saved to {config_file}")
            console.print()
            console.print("[bold]To use this configuration:[/bold]")
            console.print(f"[cyan]ccproxy --config {config_file} api[/cyan]")
            console.print()
            console.print("[dim]Or set CONFIG_FILE environment variable:[/dim]")
            console.print(f"[cyan]export CONFIG_FILE={config_file}[/cyan]")
            console.print("[cyan]ccproxy api[/cyan]")

    except (OSError, PermissionError) as e:
        logger.error("generate_token_file_write_error", error=str(e), exc_info=e)
        toolkit.print(f"Error writing configuration file: {e}", tag="error")
        raise typer.Exit(1) from e
    except ValueError as e:
        logger.error("generate_token_value_error", error=str(e), exc_info=e)
        toolkit.print(f"Token generation configuration error: {e}", tag="error")
        raise typer.Exit(1) from e
    except ImportError as e:
        logger.error("generate_token_import_error", error=str(e), exc_info=e)
        toolkit.print(f"Module import error: {e}", tag="error")
        raise typer.Exit(1) from e
    except Exception as e:
        logger.error("generate_token_unexpected_error", error=str(e), exc_info=e)
        toolkit.print(f"Error generating token: {e}", tag="error")
        raise typer.Exit(1) from e


def _detect_config_format(config_file: Path) -> str:
    """Detect configuration file format from extension."""
    suffix = config_file.suffix.lower()
    if suffix in [".toml"]:
        return "toml"
    else:
        return "toml"


def _generate_default_config_from_model(
    settings_class: type[Settings],
) -> dict[str, Any]:
    """Generate a default configuration dictionary from the Settings model."""
    default_settings = settings_class()

    config_data: dict[str, Any] = {}

    for field_name, _field_info in settings_class.model_fields.items():
        field_value = getattr(default_settings, field_name)

        if isinstance(field_value, BaseModel):
            config_data[field_name] = _generate_nested_config_from_model(field_value)
        else:
            if isinstance(field_value, Path):
                config_data[field_name] = str(field_value)
            else:
                config_data[field_name] = field_value

    return config_data


def _generate_nested_config_from_model(model: BaseModel) -> dict[str, Any]:
    """Generate configuration for nested models."""
    config_data: dict[str, Any] = {}

    for field_name, _field_info in model.model_fields.items():
        field_value = getattr(model, field_name)

        if isinstance(field_value, BaseModel):
            config_data[field_name] = _generate_nested_config_from_model(field_value)
        else:
            if isinstance(field_value, Path):
                config_data[field_name] = str(field_value)
            else:
                config_data[field_name] = field_value

    return config_data


def _write_toml_config_with_comments(
    config_file: Path, config_data: dict[str, Any], settings_class: type[Settings]
) -> None:
    """Write configuration data to a TOML file with comments and proper formatting."""
    with config_file.open("w", encoding="utf-8") as f:
        f.write("# CCProxy API Configuration\n")
        f.write("# This file configures the ccproxy server settings\n")
        f.write("# Most settings are commented out with their default values\n")
        f.write("# Uncomment and modify as needed\n\n")

        for field_name, _field_info in settings_class.model_fields.items():
            field_value = config_data.get(field_name)
            description = _get_field_description(_field_info)

            f.write(f"# {description}\n")

            if isinstance(field_value, dict):
                f.write(f"# [{field_name}]\n")
                _write_toml_section(f, field_value, prefix="# ", level=0)
            else:
                formatted_value = _format_config_value_for_toml(field_value)
                f.write(f"# {field_name} = {formatted_value}\n")

            f.write("\n")


def _write_toml_section(
    f: Any, data: dict[str, Any], prefix: str = "", level: int = 0
) -> None:
    """Write a TOML section with proper indentation and commenting."""
    for key, value in data.items():
        if isinstance(value, dict):
            f.write(f"{prefix}[{key}]\n")
            _write_toml_section(f, value, prefix, level + 1)
        else:
            formatted_value = _format_config_value_for_toml(value)
            f.write(f"{prefix}{key} = {formatted_value}\n")


def _format_config_value_for_toml(value: Any) -> str:
    """Format a configuration value for TOML output."""
    if value is None:
        return "null"
    elif isinstance(value, bool):
        return "true" if value else "false"
    elif isinstance(value, str):
        return f'"{value}"'  # Correctly escape quotes within strings
    elif isinstance(value, int | float):
        return str(value)
    elif isinstance(value, list):
        if not value:
            return "[]"
        formatted_items = []
        for item in value:
            if isinstance(item, str):
                formatted_items.append(
                    f'"{item}"'
                )  # Correctly escape quotes within list strings
            else:
                formatted_items.append(str(item))
        return f"[{', '.join(formatted_items)}]]"
    elif isinstance(value, dict):
        if not value:
            return "{{}}"
        formatted_items = []
        for k, v in value.items():
            if isinstance(v, str):
                formatted_items.append(
                    f'{k} = "{v}"'
                )  # Correctly escape quotes within dict strings
            else:
                formatted_items.append(f"{k} = {v}")
        return f"{{{', '.join(formatted_items)}}}"
    else:
        return str(value)


def _write_config_file(
    config_file: Path, config_data: dict[str, Any], file_format: str
) -> None:
    """Write configuration data to file in the specified format."""
    if file_format == "toml":
        _write_toml_config_with_comments(config_file, config_data, Settings)
    else:
        raise ValueError(
            f"Unsupported config format: {file_format}. Only TOML is supported."
        )
