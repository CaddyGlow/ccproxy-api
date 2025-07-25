"""Server configuration settings."""

from pydantic import BaseModel, Field, field_validator


class ServerSettings(BaseModel):
    """Server-specific configuration settings."""

    host: str = Field(
        default="127.0.0.1",
        description="Server host address",
    )

    port: int = Field(
        default=8000,
        description="Server port number",
        ge=1,
        le=65535,
    )

    workers: int = Field(
        default=1,
        description="Number of worker processes",
        ge=1,
        le=32,
    )

    reload: bool = Field(
        default=False,
        description="Enable auto-reload for development",
    )

    log_level: str = Field(
        default="INFO",
        description="Logging level",
    )

    log_format: str = Field(
        default="auto",
        description="Logging output format: 'rich' for development, 'json' for production, 'auto' for automatic selection",
    )

    log_show_path: bool = Field(
        default=False,
        description="Whether to show module path in logs (automatically enabled for DEBUG level)",
    )

    log_show_time: bool = Field(
        default=True,
        description="Whether to show timestamps in logs",
    )

    log_console_width: int | None = Field(
        default=None,
        description="Optional console width override for Rich output",
    )

    log_file: str | None = Field(
        default=None,
        description="Path to JSON log file. If specified, logs will be written to this file in JSON format",
    )

    use_terminal_permission_handler: bool = Field(
        default=False,
        description="Enable terminal UI for permission prompts. Set to False to use external handler via SSE (not implemented)",
    )

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate and normalize log level."""
        upper_v = v.upper()
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if upper_v not in valid_levels:
            raise ValueError(f"Invalid log level: {v}. Must be one of {valid_levels}")
        return upper_v

    @field_validator("log_format")
    @classmethod
    def validate_log_format(cls, v: str) -> str:
        """Validate and normalize log format."""
        lower_v = v.lower()
        valid_formats = ["auto", "rich", "json", "plain"]
        if lower_v not in valid_formats:
            raise ValueError(f"Invalid log format: {v}. Must be one of {valid_formats}")
        return lower_v
