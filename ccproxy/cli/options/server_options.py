"""Server-related CLI options."""

import typer


def validate_port(
    ctx: typer.Context, param: typer.CallbackParam, value: int | None
) -> int | None:
    """Validate port number."""
    if value is None:
        return None

    if value < 1 or value > 65535:
        raise typer.BadParameter("Port must be between 1 and 65535")

    return value


def validate_log_level(
    ctx: typer.Context, param: typer.CallbackParam, value: str | None
) -> str | None:
    """Validate log level."""
    if value is None:
        return None

    valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
    if value.upper() not in valid_levels:
        raise typer.BadParameter(f"Log level must be one of: {', '.join(valid_levels)}")

    return value.upper()


# Factory functions removed - use Annotated syntax directly in commands


class ServerOptions:
    """Container for all server-related CLI options.

    This class provides a convenient way to include all server-related
    options in a command using typed attributes.
    """

    def __init__(
        self,
        port: int | None = None,
        host: str | None = None,
        reload: bool | None = None,
        log_level: str | None = None,
        log_file: str | None = None,
        use_terminal_confirmation_handler: bool | None = None,
    ):
        """Initialize server options.

        Args:
            port: Port to run the server on
            host: Host to bind the server to
            reload: Enable auto-reload for development
            log_level: Logging level
            log_file: Path to JSON log file
            use_terminal_confirmation_handler: Enable terminal UI for confirmation prompts
        """
        self.port = port
        self.host = host
        self.reload = reload
        self.log_level = log_level
        self.log_file = log_file
        self.use_terminal_confirmation_handler = use_terminal_confirmation_handler
