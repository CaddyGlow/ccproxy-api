"""Core CLI options for configuration and global settings."""

from pathlib import Path
from typing import Any

import typer


def config_option() -> Any:
    """Configuration file parameter."""
    return typer.Option(
        None,
        "--config",
        "-c",
        help="Path to configuration file (TOML, JSON, or YAML)",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        rich_help_panel="Configuration",
    )


class CoreOptions:
    """Container for core CLI options.

    This class provides a convenient way to include core options
    in a command using typed attributes.
    """

    def __init__(
        self,
        config: Path | None = None,
    ):
        """Initialize core options.

        Args:
            config: Path to configuration file
        """
        self.config = config
