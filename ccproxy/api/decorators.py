"""Route decorators for CCProxy API."""

from collections.abc import Callable
from functools import wraps
from typing import Any


def format_chain(
    chain: list[str],
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator to set the format chain for a route.

    Args:
        chain: Format chain list (e.g., ["openai"], ["openai", "anthropic"])

    Usage:
        @router.post("/v1/chat/completions")
        @format_chain(["openai"])
        async def chat_completions(request: Request, ...):
            ...
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        # Store the format chain as a function attribute
        func.__format_chain__ = chain  # type: ignore[attr-defined]

        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            return await func(*args, **kwargs)

        # Ensure the wrapper also has the format chain attribute
        wrapper.__format_chain__ = chain  # type: ignore[attr-defined]
        return wrapper

    return decorator


def base_format(format_name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator to set the base format for a route (shorthand for single format).

    Args:
        format_name: Base format name (e.g., "openai", "anthropic", "response_api")

    Usage:
        @router.post("/v1/messages")
        @base_format("anthropic")
        async def messages(request: Request, ...):
            ...
    """
    return format_chain([format_name])
