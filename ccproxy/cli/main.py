"""Main entry point for CCProxy API Server.

Adds per-invocation debug logging of CLI argv and relevant environment
variables (masked) so every command emits its context consistently.
"""

import os
import sys
from pathlib import Path
from typing import Annotated

import typer

from ccproxy.cli.helpers import (
    get_rich_toolkit,
)
from ccproxy.core._version import __version__
from ccproxy.core.logging import bootstrap_cli_logging, get_logger, set_command_context
from ccproxy.core.plugins.cli_discovery import discover_plugin_cli_extensions
from ccproxy.core.plugins.declaration import CliArgumentSpec, CliCommandSpec

# from plugins.permissions.handlers.cli import app as permission_handler_app
from .commands.auth import app as auth_app
from .commands.config import app as config_app
from .commands.plugins import app as plugins_app
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


def register_plugin_cli_extensions(app: typer.Typer) -> None:
    """Register plugin CLI commands and arguments during app creation."""
    try:
        # Load settings to apply plugin filtering
        try:
            from ccproxy.config.settings import Settings

            settings = Settings.from_config()
        except Exception as e:
            # Graceful degradation - use no filtering if settings fail to load
            logger.debug("settings_load_failed_for_cli_discovery", error=str(e))
            settings = None

        plugin_manifests = discover_plugin_cli_extensions(settings)

        logger.debug(
            "plugin_cli_discovery_complete",
            plugin_count=len(plugin_manifests),
            plugins=[name for name, _ in plugin_manifests],
        )

        for plugin_name, manifest in plugin_manifests:
            # Register new commands
            for cmd_spec in manifest.cli_commands:
                _register_plugin_command(app, plugin_name, cmd_spec)

            # Extend existing commands with new arguments
            for arg_spec in manifest.cli_arguments:
                _extend_command_with_argument(app, plugin_name, arg_spec)

    except Exception as e:
        # Graceful degradation - CLI still works without plugin extensions
        logger.debug("plugin_cli_extension_registration_failed", error=str(e))


def _register_plugin_command(
    app: typer.Typer, plugin_name: str, cmd_spec: CliCommandSpec
) -> None:
    """Register a single plugin command."""
    try:
        if cmd_spec.parent_command is None:
            # Top-level command
            app.command(
                name=cmd_spec.command_name,
                help=cmd_spec.help_text or f"Command from {plugin_name} plugin",
            )(cmd_spec.command_function)
            logger.debug(
                "plugin_command_registered",
                plugin=plugin_name,
                command=cmd_spec.command_name,
                type="top_level",
            )
        else:
            # Subcommand - add to existing command groups
            parent_app = _get_command_app(cmd_spec.parent_command)
            if parent_app:
                parent_app.command(
                    name=cmd_spec.command_name,
                    help=cmd_spec.help_text or f"Command from {plugin_name} plugin",
                )(cmd_spec.command_function)
                logger.debug(
                    "plugin_command_registered",
                    plugin=plugin_name,
                    command=cmd_spec.command_name,
                    parent=cmd_spec.parent_command,
                    type="subcommand",
                )
            else:
                logger.warning(
                    "plugin_command_parent_not_found",
                    plugin=plugin_name,
                    command=cmd_spec.command_name,
                    parent=cmd_spec.parent_command,
                )
    except Exception as e:
        logger.warning(
            "plugin_command_registration_failed",
            plugin=plugin_name,
            command=cmd_spec.command_name,
            error=str(e),
        )


def _extend_command_with_argument(
    app: typer.Typer, plugin_name: str, arg_spec: CliArgumentSpec
) -> None:
    """Extend an existing command with a new argument."""
    # This is more complex and may require command wrapping or dynamic parameter injection
    # For now, log the extension attempt
    logger.debug(
        "plugin_argument_extension_requested",
        plugin=plugin_name,
        target_command=arg_spec.target_command,
        argument=arg_spec.argument_name,
    )
    # TODO: Implement argument injection into existing commands


def _get_command_app(command_name: str) -> typer.Typer | None:
    """Get the typer app for a parent command."""
    command_apps = {
        "auth": auth_app,
        "config": config_app,
        "plugins": plugins_app,
    }
    return command_apps.get(command_name)


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
# app.add_typer(permission_handler_app)

# Register plugins command
app.add_typer(plugins_app)

register_plugin_cli_extensions(app)

# Register imported commands
app.command(name="serve")(api)
# Claude command removed - functionality moved to serve command


def main() -> None:
    """Entry point for the CLI application."""
    # Bind a command-wide correlation ID so all logs have `cmd_id`
    set_command_context()
    # Early logging bootstrap from env/argv; safe to reconfigure later
    bootstrap_cli_logging()
    # Log invocation context (argv + env) for all commands
    _log_cli_invocation_context()
    app()


if __name__ == "__main__":
    sys.exit(app())


def _mask_env_value(key: str, value: str) -> str:
    """Mask sensitive values based on common substrings in the key."""
    lowered = key.lower()
    sensitive_markers = [
        "token",
        "secret",
        "password",
        "passwd",
        "key",
        "api_key",
        "bearer",
        "auth",
        "credential",
    ]
    if any(m in lowered for m in sensitive_markers):
        if not value:
            return value
        # keep only last 4 chars for minimal debugging
        tail = value[-4:] if len(value) > 4 else "".join("*" for _ in value)
        return f"***MASKED***{tail}"
    return value


def _collect_relevant_env() -> dict[str, str]:
    """Collect env vars relevant to settings/plugins and mask sensitive ones.

    We include nested-style variables (containing "__") and key CCProxy groups.
    """
    prefixes = (
        "LOGGING__",
        "PLUGINS__",
        "SERVER__",
        "STORAGE__",
        "AUTH__",
        "CCPROXY__",
        "CCPROXY_",
    )
    env = {}
    for k, v in os.environ.items():
        # Ignore variables that start with double underscore
        if k.startswith("__"):
            continue
        if "__" in k or k.startswith(prefixes):
            env[k] = _mask_env_value(k, v)
    # Sort for stable output
    return dict(sorted(env.items(), key=lambda kv: kv[0]))


def _log_cli_invocation_context() -> None:
    """Log argv and selected env at debug level for all commands."""
    try:
        env = _collect_relevant_env()
        logger.debug(
            "cli_invocation",
            argv=sys.argv,
            env=env,
            category="cli",
        )
    except Exception:
        # Never let logging context fail the CLI
        pass
