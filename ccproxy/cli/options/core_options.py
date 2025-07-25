"""Core CLI options for configuration and global settings."""

from pathlib import Path


# Factory functions removed - use Annotated syntax directly in commands


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
