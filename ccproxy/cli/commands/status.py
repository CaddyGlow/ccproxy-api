"""Status command for displaying system information."""

import typer
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ccproxy.cli.helpers import get_rich_toolkit
from ccproxy.config.settings import Settings
from ccproxy.core._version import __version__
from ccproxy.core.logging import get_logger
from ccproxy.core.status_report import (
    ConfigSnapshot,
    PluginSnapshot,
    SystemSnapshot,
    collect_config_snapshot,
    collect_plugin_snapshot,
    collect_system_snapshot,
)


app = typer.Typer(name="status", help="Show system status and information")
logger = get_logger(__name__)


@app.callback(invoke_without_command=True)
def status_main(
    ctx: typer.Context,
    plugins: bool = typer.Option(
        True, "--plugins/--no-plugins", help="Show plugin status"
    ),
    config: bool = typer.Option(
        True, "--config/--no-config", help="Show configuration info"
    ),
) -> None:
    """Show CCProxy system status and configuration."""
    if ctx.invoked_subcommand is not None:
        return

    toolkit = get_rich_toolkit()
    console = Console()

    # Header
    toolkit.print("[bold cyan]CCProxy Status[/bold cyan]", centered=True)
    toolkit.print(f"Version: {__version__}", tag="info")
    toolkit.print_line()

    try:
        settings = Settings.from_config()

        system_snapshot = collect_system_snapshot(settings)
        _show_system_info(console, system_snapshot)

        if config:
            config_snapshot = collect_config_snapshot()
            _show_config_info(console, config_snapshot)

        if plugins:
            plugin_snapshot = collect_plugin_snapshot(settings)
            _show_plugin_status(console, plugin_snapshot)

    except Exception as e:
        logger.error("status_command_error", error=str(e), exc_info=e)
        toolkit.print(f"[red]✗[/red] Error gathering status: {e}", tag="error")


def _show_system_info(console: Console, snapshot: SystemSnapshot) -> None:
    """Show basic system information."""
    table = Table(
        show_header=False,
        box=box.ROUNDED,
        title="System Information",
        title_style="bold white",
        border_style="blue",
    )
    table.add_column("Property", style="cyan", no_wrap=True)
    table.add_column("Value", style="white")

    # Server configuration
    table.add_row("Server Host", snapshot.host)
    table.add_row("Server Port", str(snapshot.port))
    table.add_row("Log Level", snapshot.log_level)

    # Authentication
    auth_token_status = (
        "✓ Configured" if snapshot.auth_token_configured else "✗ Not set"
    )
    table.add_row("API Token", auth_token_status)

    # Plugin directories
    plugin_dirs = snapshot.plugin_directories
    plugins_enabled = "✓ Enabled" if snapshot.plugins_enabled else "✗ Disabled"
    table.add_row("Plugins", plugins_enabled)

    # Show each plugin directory with existence check
    for i, plugin_dir in enumerate(plugin_dirs):
        dir_exists = "✓ Exists" if plugin_dir.exists else "✗ Missing"
        dir_label = (
            f"Plugin Directory {i + 1}" if len(plugin_dirs) > 1 else "Plugin Directory"
        )
        table.add_row(dir_label, f"{plugin_dir.path} ({dir_exists})")

    console.print(table)
    console.print()


def _show_config_info(console: Console, snapshot: ConfigSnapshot) -> None:
    """Show configuration source information."""
    if snapshot.sources:
        info_text = Text()
        info_text.append("Configuration Sources:\n", style="bold")
        for source in snapshot.sources:
            prefix = "✓" if source.exists else "✗"
            info_text.append(f"{prefix} {source.path}\n")

        console.print(Panel(info_text, title="Configuration", border_style="green"))
        console.print()


def _show_plugin_status(console: Console, snapshot: PluginSnapshot) -> None:
    """Show plugin discovery and status information."""

    # Short-circuit if plugins are entirely disabled
    if not snapshot.plugin_system_enabled:
        table = Table(
            show_header=True,
            header_style="bold cyan",
            box=box.ROUNDED,
            title="Plugin Status",
            title_style="bold white",
            border_style="yellow",
        )
        table.add_column("Plugin", style="cyan", no_wrap=True)
        table.add_column("Status", style="white", no_wrap=True)
        table.add_column("Version", style="yellow")
        table.add_column("Description", style="dim white")

        table.add_row(
            "Plugin System",
            "[red]✗ Disabled[/red]",
            "",
            "Plugin system disabled in configuration",
        )

        console.print(table)
        console.print(
            Panel(
                "Plugin system is disabled. No plugins will be loaded or executed.",
                title="Plugin Summary",
                border_style="yellow",
            )
        )
        if snapshot.configuration_notes:
            _print_plugin_configuration_notes(console, snapshot.configuration_notes)
        return

    table = Table(
        show_header=True,
        header_style="bold cyan",
        box=box.ROUNDED,
        title="Plugin Status",
        title_style="bold white",
        border_style="green",
    )
    table.add_column("Plugin", style="cyan", no_wrap=True)
    table.add_column("Status", style="white", no_wrap=True)
    table.add_column("Version", style="yellow")
    table.add_column("Description", style="dim white")

    for info in snapshot.enabled_plugins:
        status = "[green]✓ Enabled[/green]"
        version = info.version or "unknown"
        description = info.description or ""

        if info.state == "error":
            status = "[red]✗ Error[/red]"
            error_message = info.error or "Unknown error"
            preview = error_message[:50]
            if len(error_message) > 50:
                preview += "..."
            description = f"Error: {preview}"

        table.add_row(info.name, status, version, description)

    for name in snapshot.disabled_plugins:
        table.add_row(
            name,
            "[yellow]⦸ Disabled[/yellow]",
            "unknown",
            "Plugin disabled in configuration",
        )

    if not snapshot.enabled_plugins and not snapshot.disabled_plugins:
        table.add_row("No plugins found", "[yellow]⚠ Warning[/yellow]", "", "")

    console.print(table)

    summary_text = Text()
    summary_text.append(f"Total plugins: {snapshot.total_count} ", style="bold")
    summary_text.append(f"(Enabled: {snapshot.enabled_count}, ", style="green")
    summary_text.append(f"Disabled: {snapshot.disabled_count})", style="yellow")

    console.print(Panel(summary_text, title="Plugin Summary", border_style="blue"))

    if snapshot.configuration_notes:
        _print_plugin_configuration_notes(console, snapshot.configuration_notes)


def _print_plugin_configuration_notes(console: Console, notes: tuple[str, ...]) -> None:
    config_text = Text()
    config_text.append("Configuration: ", style="bold")
    config_text.append(" • ".join(notes))
    console.print(Panel(config_text, title="Plugin Configuration", border_style="blue"))
    console.print()


@app.command()
def plugins() -> None:
    """Show detailed plugin information only."""
    toolkit = get_rich_toolkit()
    console = Console()

    toolkit.print("[bold cyan]Plugin Status[/bold cyan]", centered=True)
    toolkit.print_line()

    try:
        settings = Settings.from_config()
        plugin_snapshot = collect_plugin_snapshot(settings)
        _show_plugin_status(console, plugin_snapshot)
    except Exception as e:
        logger.error("plugin_status_command_error", error=str(e), exc_info=e)
        toolkit.print(f"[red]✗[/red] Error: {e}", tag="error")


@app.command()
def config() -> None:
    """Show configuration information only."""
    toolkit = get_rich_toolkit()
    console = Console()

    toolkit.print("[bold cyan]Configuration Status[/bold cyan]", centered=True)
    toolkit.print_line()

    try:
        settings = Settings.from_config()
        system_snapshot = collect_system_snapshot(settings)
        config_snapshot = collect_config_snapshot()
        _show_system_info(console, system_snapshot)
        _show_config_info(console, config_snapshot)
    except Exception as e:
        logger.error("config_status_command_error", error=str(e), exc_info=e)
        toolkit.print(f"[red]✗[/red] Error: {e}", tag="error")
