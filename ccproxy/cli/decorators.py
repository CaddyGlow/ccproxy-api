"""CLI command decorators for plugin dependency management."""

from collections.abc import Callable
from typing import Any, ParamSpec, TypeVar


P = ParamSpec("P")
R = TypeVar("R")


def needs_auth_provider() -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator to mark CLI commands that need an auth provider.

    This decorator marks the command as requiring the auth provider specified
    in the command arguments. The actual plugin loading is handled by the
    command implementation using load_cli_plugins().

    Usage:
        @app.command()
        @needs_auth_provider()
        async def auth_status(provider: str):
            # Command implementation
            pass
    """

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        # Add metadata to the function
        func._needs_auth_provider = True  # type: ignore
        return func

    return decorator


def allows_plugins(
    plugin_names: list[str],
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator to specify additional plugins a CLI command can use.

    This decorator specifies additional CLI-safe plugins that the command
    wants to use beyond the default set. These plugins must still be marked
    as cli_safe = True to be loaded.

    Args:
        plugin_names: List of plugin names to allow (e.g., ["request_tracer", "metrics"])

    Usage:
        @app.command()
        @allows_plugins(["request_tracer", "metrics"])
        async def my_command():
            # Command implementation
            pass
    """

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        # Add metadata to the function
        func._allows_plugins = plugin_names  # type: ignore
        return func

    return decorator


def get_command_auth_provider(func: Callable[..., Any]) -> bool:
    """Check if a command needs an auth provider.

    Args:
        func: Function to check

    Returns:
        True if the command is decorated with @needs_auth_provider()
    """
    return getattr(func, "_needs_auth_provider", False)


def get_command_allowed_plugins(func: Callable[..., Any]) -> list[str]:
    """Get the allowed plugins for a command.

    Args:
        func: Function to check

    Returns:
        List of allowed plugin names (empty list if none specified)
    """
    return getattr(func, "_allows_plugins", [])
