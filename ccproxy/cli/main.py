"""Main entry point for CCProxy API Server.

Adds per-invocation debug logging of CLI argv and relevant environment
variables (masked) so every command emits its context consistently.
"""

import os
import sys
from pathlib import Path
from typing import Annotated, Any, cast

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
from .commands.status import app as status_app


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

_plugins_registered = False


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

        # Register new commands first
        for plugin_name, manifest in plugin_manifests:
            for cmd_spec in manifest.cli_commands:
                _register_plugin_command(app, plugin_name, cmd_spec)

        # Batch extend existing commands with new arguments
        arg_batches: dict[str, list[tuple[str, CliArgumentSpec]]] = {}
        for plugin_name, manifest in plugin_manifests:
            for arg_spec in manifest.cli_arguments:
                arg_batches.setdefault(arg_spec.target_command, []).append(
                    (plugin_name, arg_spec)
                )

        for target, pairs in arg_batches.items():
            _extend_command_with_arguments(app, target, pairs)

    except Exception as e:
        # Graceful degradation - CLI still works without plugin extensions
        logger.debug("plugin_cli_extension_registration_failed", error=str(e))


def ensure_plugin_cli_extensions_registered(app: typer.Typer) -> None:
    """Register plugin CLI extensions once, after logging is configured."""
    global _plugins_registered
    if _plugins_registered:
        return

    register_plugin_cli_extensions(app)
    _plugins_registered = True


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


def _extend_command_with_arguments(
    app: typer.Typer, target_name: str, pairs: list[tuple[str, CliArgumentSpec]]
) -> None:
    """Extend an existing command with multiple new arguments at once."""
    try:
        # Utility: resolve a command callback by dotted path, supporting subcommands
        def _resolve_callback(root_app: typer.Typer, dotted: str) -> Any:
            parts = dotted.split()
            # Allow both space-separated or colon-separated ("auth login")
            if len(parts) == 1 and (":" in dotted or "/" in dotted):
                # normalize common separators
                dotted_norm = dotted.replace(":", " ").replace("/", " ")
                parts = dotted_norm.split()

            current_app: typer.Typer | None = root_app
            callback = None

            # Typer API does not expose a stable public registry; we rely on
            # attributes commonly present on Typer instances.
            for idx, part in enumerate(parts):
                if current_app is None:
                    return None
                # Try commands at this level
                cmds = getattr(current_app, "registered_commands", [])
                sub_app: typer.Typer | None = None
                found = False
                for cmd in cmds:
                    name = getattr(cmd, "name", None)
                    if name == part:
                        found = True
                        if idx == len(parts) - 1:
                            callback = getattr(cmd, "callback", None)
                        else:
                            # navigate into sub-typer if present
                            sub_app = getattr(cmd, "typer_instance", None)
                        break
                current_app = sub_app if sub_app is not None else current_app
                if not found and idx == len(parts) - 1:
                    # maybe the target is a top-level name even if earlier parts failed
                    pass
            return callback

        # Find existing command callback on the root app or nested apps
        original_callback = _resolve_callback(app, target_name)

        if original_callback is None:
            logger.debug(
                "plugin_argument_extension_target_not_found",
                target_command=target_name,
            )
            return
        # Build all options and construct wrapper
        built: list[tuple[CliArgumentSpec, Any]] = []
        for _, arg_spec in pairs:
            typer_kwargs = dict(arg_spec.typer_kwargs or {})
            option_names = typer_kwargs.pop("option", None)
            if option_names is None:
                option_names = [f"--{arg_spec.argument_name.replace('_', '-')}"]
            if isinstance(option_names, str):
                option_names = [option_names]
            option = typer.Option(
                *option_names,
                help=arg_spec.help_text or "",
                **typer_kwargs,
            )
            built.append((arg_spec, option))

        # Preserve original options: build wrapper with original signature + injected params
        import inspect

        sig = inspect.signature(original_callback)
        param_defs: list[str] = []
        call_args: list[str] = []
        annotations: dict[str, object] = {}

        # Always include ctx first for our bookkeeping
        param_defs.append("ctx: typer.Context")
        annotations["ctx"] = typer.Context

        # Mirror original parameters with their defaults in order
        for name, param in sig.parameters.items():
            if getattr(original_callback, "__annotations__", {}).get(name) is not None:
                annotations[name] = original_callback.__annotations__[name]
            if param.default is inspect._empty:
                param_defs.append(name)
            else:
                param_defs.append(f"{name}={repr(param.default)}")
            call_args.append(name)

        # Append injected params
        for arg_spec, _ in built:
            default_expr = "..." if arg_spec.required else "None"
            param_defs.append(f"{arg_spec.argument_name}: object = {default_expr}")

        params_sig = ", ".join(param_defs)

        body_lines = [
            "ctx.ensure_object(dict)",
            "plugin_map = ctx.obj.get('plugin_cli_args') or {}",
            "plugin_map = plugin_map if isinstance(plugin_map, dict) else {}",
        ]
        for arg_spec, _ in built:
            name = arg_spec.argument_name
            body_lines.append(f"if {name} is not None: plugin_map['{name}'] = {name}")
        body_lines.append("ctx.obj['plugin_cli_args'] = plugin_map")
        body_lines.append(f"return original_callback({', '.join(call_args)})")

        body_src = "\n    ".join(body_lines)
        func_src = f"def _wrapped({params_sig}):\n    {body_src}\n"
        local_ns: dict[str, object] = {
            "typer": typer,
            "original_callback": original_callback,
        }
        exec(func_src, local_ns, local_ns)
        _wrapped = local_ns["_wrapped"]

        # Copy original annotations and append injected option annotations
        for name, ann in getattr(original_callback, "__annotations__", {}).items():
            annotations.setdefault(name, ann)
        for arg_spec, option in built:
            annotations[arg_spec.argument_name] = Annotated[
                arg_spec.argument_type | None, option
            ]
        _wrapped.__annotations__ = annotations

        _wrapped.__name__ = getattr(  # type: ignore[attr-defined]
            original_callback, "__name__", f"_{target_name}_wrapped"
        )
        _wrapped.__doc__ = getattr(original_callback, "__doc__", None)

        app.command(name=target_name)(cast(Any, _wrapped))

        for plugin_name, arg_spec in pairs:
            logger.debug(
                "plugin_argument_extension_registered",
                plugin=plugin_name,
                target_command=target_name,
                argument=arg_spec.argument_name,
            )
    except Exception as e:
        logger.debug(
            "plugin_argument_extension_failed",
            target_command=target_name,
            error=str(e),
        )


def _get_command_app(command_name: str) -> typer.Typer | None:
    """Get the typer app for a parent command."""
    command_apps = {
        "auth": auth_app,
        "config": config_app,
        "plugins": plugins_app,
        "status": status_app,
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
    # Store config path and initialize plugin arg bucket
    ctx.ensure_object(dict)
    ctx.obj["config_path"] = config
    if "plugin_cli_args" not in ctx.obj or not isinstance(
        ctx.obj.get("plugin_cli_args"), dict
    ):
        ctx.obj["plugin_cli_args"] = {}

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

# Register status command
app.add_typer(status_app)

# Register imported commands first
app.command(name="serve")(api)

# Claude command removed - functionality moved to serve command


def main() -> None:
    """Entry point for the CLI application."""
    # Bind a command-wide correlation ID so all logs have `cmd_id`
    set_command_context()
    # Early logging bootstrap from env/argv; safe to reconfigure later
    bootstrap_cli_logging()
    # Register plugin-supplied CLI commands after logging honors env overrides
    ensure_plugin_cli_extensions_registered(app)
    # Log invocation context (argv + env) for all commands
    _log_cli_invocation_context()
    app()


if __name__ == "__main__":
    main()


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
