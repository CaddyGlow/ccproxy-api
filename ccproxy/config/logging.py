"""Centralized logging configuration settings."""

from pydantic import BaseModel, Field, field_validator


class LoggingSettings(BaseModel):
    """Centralized logging configuration - core app only."""

    # === Core Application Logging ===
    level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL, TRACE)",
    )

    format: str = Field(
        default="auto",
        description="Logging output format: 'rich' for development, 'json' for production, 'auto' for automatic selection",
    )

    file: str | None = Field(
        default=None,
        description="Path to JSON log file. If specified, logs will be written to this file in JSON format",
    )

    show_path: bool = Field(
        default=False,
        description="Whether to show module path in logs (automatically enabled for DEBUG level)",
    )

    show_time: bool = Field(
        default=True,
        description="Whether to show timestamps in logs",
    )

    console_width: int | None = Field(
        default=None,
        description="Optional console width override for Rich output",
    )

    # === API Request/Response Logging ===
    verbose_api: bool = Field(
        default=False,
        description="Enable verbose API request/response logging",
    )

    request_log_dir: str | None = Field(
        default=None,
        description="Directory to save individual request/response logs when verbose_api is enabled",
    )

    # === Hook System Logging ===
    use_hook_logging: bool = Field(
        default=True,
        description="Enable logging through the hook system",
    )

    enable_access_logging: bool = Field(
        default=True,
        description="Enable access logging for middleware",
    )

    enable_streaming_logging: bool = Field(
        default=True,
        description="Enable logging for streaming events",
    )

    parallel_run_mode: bool = Field(
        default=False,
        description="Enable parallel run mode for hooks",
    )

    disable_middleware_during_parallel: bool = Field(
        default=False,
        description="Disable middleware during parallel hook execution",
    )

    # === Observability Integration ===
    pipeline_enabled: bool = Field(
        default=True,
        description="Enable structlog pipeline integration for observability",
    )

    observability_format: str = Field(
        default="auto",
        description="Logging format for observability: 'rich', 'json', 'auto' (auto-detects based on environment)",
    )

    # === Plugin Logging Master Controls (Plugin-Agnostic) ===
    enable_plugin_logging: bool = Field(
        default=True,
        description="Global kill switch for ALL plugin logging features",
    )

    plugin_log_base_dir: str = Field(
        default="/tmp/ccproxy",
        description="Shared base directory for all plugin log outputs",
    )

    plugin_log_retention_days: int = Field(
        default=7,
        description="How long to keep plugin-generated logs (in days)",
    )

    # Scalable per-plugin control (based on Gemini's recommendation)
    plugin_overrides: dict[str, bool] = Field(
        default_factory=dict,
        description="Per-plugin enable/disable overrides. Key=plugin_name, Value=enabled. "
        "A plugin is enabled if not in dict or if value is True",
    )

    @field_validator("level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate and normalize log level."""
        upper_v = v.upper()
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL", "TRACE"]
        if upper_v not in valid_levels:
            raise ValueError(f"Invalid log level: {v}. Must be one of {valid_levels}")
        return upper_v

    @field_validator("format", "observability_format")
    @classmethod
    def validate_log_format(cls, v: str) -> str:
        """Validate and normalize log format."""
        lower_v = v.lower()
        valid_formats = ["auto", "rich", "json", "plain"]
        if lower_v not in valid_formats:
            raise ValueError(f"Invalid log format: {v}. Must be one of {valid_formats}")
        return lower_v
