"""CLI commands for interacting with plugins."""

import typer
from rich.console import Console
from rich.table import Table

from ccproxy.config.settings import get_settings
from ccproxy.plugins import discover_and_load_plugins


app = typer.Typer(name="plugins", help="Manage and inspect plugins.")


@app.command()
def settings() -> None:
    """List all available plugin settings."""
    console = Console()

    # Get settings and discover plugins
    settings_obj = get_settings()
    factories = discover_and_load_plugins(settings_obj)

    if not factories:
        console.print("No plugins found.")
        return

    for _name, factory in factories.items():
        manifest = factory.get_manifest()
        table = Table(
            title=f"Plugin: [bold]{manifest.name}[/bold] v{manifest.version}",
            show_header=True,
            header_style="bold magenta",
        )
        table.add_column("Setting", style="dim")
        table.add_column("Type")
        table.add_column("Default")

        # TODO: Add config class support to v2 plugins
        # For now, skip config display
        console.print(f"Plugin: [bold]{manifest.name}[/bold] v{manifest.version}")
        console.print("  Configuration display not yet implemented for v2 plugins.")
        console.print()


@app.command()
def dependencies() -> None:
    """Display how plugin dependencies are managed."""

    console = Console()
    console.print(
        "Plugin dependencies are managed at the package level (pyproject.toml/extras)."
    )
