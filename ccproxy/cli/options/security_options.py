"""Security-related CLI options."""

from typing import Any

import typer


def validate_auth_token(
    ctx: typer.Context, param: typer.CallbackParam, value: str | None
) -> str | None:
    """Validate auth token."""
    if value is None:
        return None

    if not value.strip():
        raise typer.BadParameter("Auth token cannot be empty")

    return value


def auth_token_option() -> Any:
    """Auth token parameter."""
    return typer.Option(
        None,
        "--auth-token",
        help="Bearer token for API authentication",
        callback=validate_auth_token,
        rich_help_panel="Security Settings",
    )


class SecurityOptions:
    """Container for all security-related CLI options.

    This class provides a convenient way to include all security-related
    options in a command using typed attributes.
    """

    def __init__(
        self,
        auth_token: str | None = None,
    ):
        """Initialize security options.

        Args:
            auth_token: Bearer token for API authentication
        """
        self.auth_token = auth_token
