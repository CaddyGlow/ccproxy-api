"""CLI helper utilities for CCProxy API."""

from pathlib import Path
from typing import Any

import typer
from rich_toolkit import RichToolkit, RichToolkitTheme
from rich_toolkit.styles import TaggedStyle


def get_rich_toolkit() -> RichToolkit:
    theme = RichToolkitTheme(
        style=TaggedStyle(tag_width=11),
        theme={
            # Core tags
            "tag.title": "white on #009485",
            "tag": "white on #007166",
            "placeholder": "grey85",
            "text": "white",
            "selected": "#007166",
            "result": "grey85",
            "progress": "on #007166",
            # Status tags
            "error": "bold red",
            "success": "bold green",
            "warning": "bold yellow",
            "info": "blue",
            # CLI specific tags
            "version": "cyan",
            "docker": "blue",
            "local": "green",
            "claude": "magenta",
            "config": "cyan",
            "volume": "yellow",
            "env": "bright_blue",
            "debug": "dim white",
            "command": "bright_cyan",
            # Logging
            "log.info": "black on blue",
            "log.error": "white on red",
            "log.warning": "black on yellow",
            "log.debug": "dim white",
        },
    )

    return RichToolkit(theme=theme)


def bold(text: str) -> str:
    return f"[bold]{text}[/bold]"


def dim(text: str) -> str:
    return f"[dim]{text}[/dim]"


def italic(text: str) -> str:
    return f"[italic]{text}[/italic]"


def warning(text: str) -> str:
    return f"[yellow]{text}[/yellow]"


def error(text: str) -> str:
    return f"[red]{text}[/red]"


def code(text: str) -> str:
    return f"[cyan]{text}[/cyan]"


def success(text: str) -> str:
    return f"[green]{text}[/green]"


def link(text: str, link: str) -> str:
    return f"[link={link}]{text}[/link]"


def is_running_in_docker() -> bool:
    return Path("/.dockerenv").exists()


def get_plugin_cli_args(ctx: typer.Context | None = None) -> dict[str, Any]:
    """Return plugin-injected CLI args from Typer context.

    Ensures a dict is returned. If no context is provided, attempts to fetch the
    current Click/Typer context. This is safe to call from command bodies.
    """
    try:
        if ctx is None:
            # Lazy import to avoid hard dependency when not in CLI execution
            from click import get_current_context as _get_ctx

            ctx = _get_ctx(silent=True)  # type: ignore[assignment]
        if ctx and getattr(ctx, "obj", None) and isinstance(ctx.obj, dict):
            data = ctx.obj.get("plugin_cli_args")
            if isinstance(data, dict):
                # Only return non-None values to simplify merging
                return {k: v for k, v in data.items() if v is not None}
    except Exception:
        pass
    return {}
