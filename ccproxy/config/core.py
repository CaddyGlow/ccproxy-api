"""Core configuration settings - server, HTTP, CORS, and logging."""

from pydantic import BaseModel, Field, field_validator


# === Server Configuration ===


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

    use_terminal_permission_handler: bool = Field(
        default=False,
        description="Enable terminal UI for permission prompts. Set to False to use external handler via SSE (not implemented)",
    )

    bypass_mode: bool = Field(
        default=False,
        description="Enable bypass mode for testing (uses mock responses instead of real API calls)",
    )


# === HTTP Configuration ===


class HTTPSettings(BaseModel):
    """HTTP client configuration settings.

    Controls how the core HTTP client handles compression and other HTTP-level settings.
    """

    compression_enabled: bool = Field(
        default=True,
        description="Enable compression for provider requests (Accept-Encoding header)",
    )

    accept_encoding: str = Field(
        default="gzip, deflate",
        description="Accept-Encoding header value when compression is enabled",
    )

    # Future HTTP settings can be added here:
    # - Connection pooling parameters
    # - Retry policies
    # - Custom headers
    # - Proxy settings


# === CORS Configuration ===


class CORSSettings(BaseModel):
    """CORS-specific configuration settings."""

    origins: list[str] = Field(
        default_factory=lambda: [
            "http://localhost:3000",
            "http://localhost:8080",
            "http://127.0.0.1:3000",
            "http://127.0.0.1:8080",
        ],
        description="CORS allowed origins (avoid using '*' for security)",
    )

    credentials: bool = Field(
        default=True,
        description="CORS allow credentials",
    )

    methods: list[str] = Field(
        default_factory=lambda: ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        description="CORS allowed methods",
    )

    headers: list[str] = Field(
        default_factory=lambda: [
            "Content-Type",
            "Authorization",
            "Accept",
            "Origin",
            "X-Requested-With",
        ],
        description="CORS allowed headers",
    )

    origin_regex: str | None = Field(
        default=None,
        description="CORS origin regex pattern",
    )

    expose_headers: list[str] = Field(
        default_factory=list,
        description="CORS exposed headers",
    )

    max_age: int = Field(
        default=600,
        description="CORS preflight max age in seconds",
        ge=0,
    )

    @field_validator("origins", mode="before")
    @classmethod
    def validate_cors_origins(cls, v: str | list[str]) -> list[str]:
        """Parse CORS origins from string or list."""
        if isinstance(v, str):
            # Split comma-separated string
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v

    @field_validator("methods", mode="before")
    @classmethod
    def validate_cors_methods(cls, v: str | list[str]) -> list[str]:
        """Parse CORS methods from string or list."""
        if isinstance(v, str):
            # Split comma-separated string
            return [method.strip().upper() for method in v.split(",") if method.strip()]
        return [method.upper() for method in v]

    @field_validator("headers", mode="before")
    @classmethod
    def validate_cors_headers(cls, v: str | list[str]) -> list[str]:
        """Parse CORS headers from string or list."""
        if isinstance(v, str):
            # Split comma-separated string
            return [header.strip() for header in v.split(",") if header.strip()]
        return v

    @field_validator("expose_headers", mode="before")
    @classmethod
    def validate_cors_expose_headers(cls, v: str | list[str]) -> list[str]:
        """Parse CORS expose headers from string or list."""
        if isinstance(v, str):
            # Split comma-separated string
            return [header.strip() for header in v.split(",") if header.strip()]
        return v

    def is_origin_allowed(self, origin: str | None) -> bool:
        """Check if an origin is allowed by the CORS policy.

        Args:
            origin: The origin to check (from request Origin header)

        Returns:
            bool: True if origin is allowed, False otherwise
        """
        if not origin:
            return False

        # Check against explicit origins list
        if origin in self.origins:
            return True

        # Check if wildcard is explicitly configured
        if "*" in self.origins:
            return True

        # Check against regex pattern if configured
        if self.origin_regex:
            import re

            try:
                return bool(re.match(self.origin_regex, origin))
            except re.error:
                return False

        return False

    def get_allowed_origin(self, request_origin: str | None) -> str | None:
        """Get the appropriate CORS origin value for response headers.

        Args:
            request_origin: The origin from the request

        Returns:
            str | None: The origin to set in Access-Control-Allow-Origin header,
                       or None if origin is not allowed
        """
        if not request_origin:
            return None

        if self.is_origin_allowed(request_origin):
            # Return specific origin instead of wildcard for security
            # Only return "*" if explicitly configured and credentials are False
            if "*" in self.origins and not self.credentials:
                return "*"
            else:
                return request_origin

        return None


# === Logging Configuration ===


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

    # Scalable per-plugin control
    plugin_overrides: dict[str, bool] = Field(
        default_factory=dict,
        description="Per-plugin enable/disable overrides. Key=plugin_name, Value=enabled. "
        "A plugin is enabled if not in dict or if value is True",
    )

    # === Noise Reduction Flags ===
    reduce_startup_info: bool = Field(
        default=True,
        description="Reduce startup INFO noise by demoting initializer logs to DEBUG",
    )
    info_summaries_only: bool = Field(
        default=True,
        description="At INFO level, show only consolidated summaries (server_ready, plugins_initialized, hooks_registered, metrics_ready, access_log_ready)",
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
