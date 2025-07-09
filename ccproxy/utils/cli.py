import logging
from pathlib import Path
from typing import Any

from rich_toolkit import RichToolkit, RichToolkitTheme
from rich_toolkit.styles import TaggedStyle
from uvicorn.logging import DefaultFormatter

from ccproxy.utils.helper import patched_typing


class CustomFormatter(DefaultFormatter):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.toolkit = get_rich_toolkit()

    def formatMessage(self, record: logging.LogRecord) -> str:  # noqa: N802
        return self.toolkit.print_as_string(record.getMessage(), tag=record.levelname)


def get_uvicorn_log_config() -> dict[str, Any]:
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "()": CustomFormatter,
                "fmt": "%(levelprefix)s %(message)s",
                "use_colors": None,
            },
            "access": {
                "()": CustomFormatter,
                "fmt": "%(levelprefix)s %(client_addr)s - '%(request_line)s' %(status_code)s",
            },
        },
        "handlers": {
            "default": {
                "formatter": "default",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stderr",
            },
            "access": {
                "formatter": "access",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
            },
        },
        "loggers": {
            "uvicorn": {"handlers": ["default"], "level": "INFO"},
            "uvicorn.error": {"level": "INFO"},
            "uvicorn.access": {
                "handlers": ["access"],
                "level": "INFO",
                "propagate": False,
            },
        },
    }


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


def merge_claude_code_options(base_options: Any, **overrides: Any) -> Any:
    """
    Create a new ClaudeCodeOptions instance by merging base options with overrides.

    Args:
        base_options: Base ClaudeCodeOptions instance to copy from
        **overrides: Dictionary of option overrides

    Returns:
        New ClaudeCodeOptions instance with merged options
    """
    with patched_typing():
        from claude_code_sdk import ClaudeCodeOptions

    # Create a new options instance with the base values
    options = ClaudeCodeOptions()

    # Copy all attributes from base_options
    if base_options:
        for attr in [
            "model",
            "max_thinking_tokens",
            "max_turns",
            "cwd",
            "system_prompt",
            "append_system_prompt",
            "permission_mode",
            "permission_prompt_tool_name",
            "continue_conversation",
            "resume",
            "allowed_tools",
            "disallowed_tools",
            "mcp_servers",
            "mcp_tools",
            # Anthropic API fields
            "temperature",
            "top_p",
            "top_k",
            "stop_sequences",
            "tools",
            "metadata",
            "service_tier",
        ]:
            if hasattr(base_options, attr):
                base_value = getattr(base_options, attr)
                if base_value is not None:
                    setattr(options, attr, base_value)

    # Apply overrides
    for key, value in overrides.items():
        if value is not None and hasattr(options, key):
            # Handle special type conversions for specific fields
            if key == "cwd" and not isinstance(value, str):
                value = str(value)
            setattr(options, key, value)

    return options


def is_running_in_docker() -> bool:
    return Path("/.dockerenv").exists()
