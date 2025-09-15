"""Configuration for the Command Replay plugin."""

from pydantic import BaseModel, ConfigDict, Field


class CommandReplayConfig(BaseModel):
    """Configuration for command replay generation.

    Generates curl and xh commands for provider requests to enable
    easy replay and debugging of API calls.
    """

    # Enable/disable entire plugin
    enabled: bool = Field(
        default=True, description="Enable or disable the command replay plugin"
    )

    # Command generation options
    generate_curl: bool = Field(default=True, description="Generate curl commands")
    generate_xh: bool = Field(default=True, description="Generate xh commands")

    # Formatting options
    pretty_format: bool = Field(
        default=True,
        description="Use pretty formatting with line continuations for readability",
    )

    # Request filtering
    include_url_patterns: list[str] = Field(
        default_factory=lambda: [
            "api.anthropic.com",
            "api.openai.com",
            "claude.ai",
            "chatgpt.com",
        ],
        description="Only generate commands for URLs matching these patterns",
    )
    exclude_url_patterns: list[str] = Field(
        default_factory=list,
        description="Skip generating commands for URLs matching these patterns",
    )

    # File output control
    log_dir: str = Field(
        default="/tmp/ccproxy/command_replay",
        description="Directory for command replay files",
    )
    write_to_files: bool = Field(default=True, description="Write commands to files")
    separate_files_per_command: bool = Field(
        default=True,
        description="Create separate files for curl and xh (False = single combined file)",
    )

    # Console output control
    log_to_console: bool = Field(
        default=True, description="Log commands to console via logger"
    )
    log_level: str = Field(
        default="INFO",
        description="Log level for command output (DEBUG, INFO, WARNING)",
    )

    # Request type filtering
    only_provider_requests: bool = Field(
        default=True,
        description="Only generate commands for provider requests (not client requests)",
    )

    model_config = ConfigDict()

    def should_generate_for_url(self, url: str) -> bool:
        """Check if commands should be generated for the given URL.

        Args:
            url: The request URL to check

        Returns:
            True if commands should be generated for this URL
        """
        # Check exclude patterns first
        if self.exclude_url_patterns:
            if any(pattern in url for pattern in self.exclude_url_patterns):
                return False

        # Check include patterns
        if self.include_url_patterns:
            return any(pattern in url for pattern in self.include_url_patterns)

        # Default: generate for all URLs if no patterns specified
        return True
