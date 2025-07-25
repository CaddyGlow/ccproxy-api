"""Main entry point for CCProxy API Server."""

from pathlib import Path
from typing import Annotated

import typer
from structlog import get_logger

from ccproxy._version import __version__
from ccproxy.cli.helpers import (
    get_rich_toolkit,
)

from .commands.auth import app as auth_app
from .commands.config import app as config_app
from .commands.permission_handler import app as permission_handler_app
from .commands.serve import api


def version_callback(value: bool) -> None:
    """Print version and exit."""
    if value:
        toolkit = get_rich_toolkit()
        toolkit.print(f"ccproxy {__version__}", tag="version")
        raise typer.Exit()


app = typer.Typer(
    rich_markup_mode="rich",
    add_completion=True,
    no_args_is_help=False,
    pretty_exceptions_enable=False,
    invoke_without_command=True,
)

# Logger will be configured by configuration manager
logger = get_logger(__name__)


# Add global options
@app.callback()
def app_main(
    ctx: typer.Context,
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            "-V",
            callback=version_callback,
            is_eager=True,
            help="Show version and exit.",
        ),
    ] = False,
    config: Annotated[
        Path | None,
        typer.Option(
            "--config",
            "-c",
            help="Path to configuration file (TOML, JSON, or YAML)",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ] = None,
) -> None:
    """CCProxy API Server - Anthropic and OpenAI compatible interface for Claude."""
    # Store config path for commands to use
    ctx.ensure_object(dict)
    ctx.obj["config_path"] = config

    # If no command is invoked, run the serve command by default
    if ctx.invoked_subcommand is None:
        # Import here to avoid circular imports
        from .commands.serve import api

        # Invoke the serve command
        ctx.invoke(api)


# Register config command
app.add_typer(config_app)

# Register auth command
app.add_typer(auth_app)

# Register permission handler command
app.add_typer(permission_handler_app)


# Register imported commands
app.command(name="serve")(api)
# Claude command removed - functionality moved to serve command


def main() -> None:
    """Entry point for the CLI application."""
    app()


if __name__ == "__main__":
    import sys

    sys.exit(app())
