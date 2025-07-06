from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any


# Fix for typing.TypedDict not supported
# pydantic.errors.PydanticUserError:
# Please use `typing_extensions.TypedDict` instead of `typing.TypedDict` on Python < 3.12.
# For further information visit https://errors.pydantic.dev/2.11/u/typed-dict-version
@contextmanager
def patched_typing() -> Iterator[None]:
    import typing

    import typing_extensions

    original = typing.TypedDict
    typing.TypedDict = typing_extensions.TypedDict
    try:
        yield
    finally:
        typing.TypedDict = original


def get_package_dir() -> Path:
    try:
        import importlib.util

        # Get the path to the claude_code_proxy package and resolve it
        spec = importlib.util.find_spec("claude_code_proxy")
        if spec and spec.origin:
            package_dir = Path(spec.origin).parent.parent.resolve()
        else:
            package_dir = Path(__file__).parent.parent.parent.resolve()
    except Exception:
        package_dir = Path(__file__).parent.parent.parent.resolve()

    return package_dir


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
