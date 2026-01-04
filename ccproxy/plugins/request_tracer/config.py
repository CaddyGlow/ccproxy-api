"""Configuration for the RequestTracer plugin."""

from pydantic import BaseModel, ConfigDict, Field


class RequestTracerConfig(BaseModel):
    """Unified configuration for request tracing.

    Combines structured JSON tracing (from core_tracer) and raw HTTP logging
    (from raw_http_logger) into a single configuration.
    """

    # Enable/disable entire plugin
    enabled: bool = Field(
        default=True, description="Enable or disable the request tracer plugin"
    )

    # Structured tracing (from core_tracer)
    verbose_api: bool = Field(
        default=True,
        description="Enable verbose API logging with structured JSON output",
    )
    json_logs_enabled: bool = Field(
        default=True, description="Enable structured JSON logging to files"
    )

    # Raw HTTP logging (from raw_http_logger)
    raw_http_enabled: bool = Field(
        default=True, description="Enable raw HTTP protocol logging"
    )

    # OAuth tracing
    trace_oauth: bool = Field(
        default=True,
        description="Enable OAuth request/response tracing for CLI operations",
    )

    # Directory configuration
    log_dir: str = Field(
        default="/tmp/ccproxy/traces", description="Base directory for all trace logs"
    )
    request_log_dir: str | None = Field(
        default=None,
        description="Override directory for structured JSON logs (defaults to log_dir)",
    )
    raw_log_dir: str | None = Field(
        default=None,
        description="Override directory for raw HTTP logs (defaults to log_dir/raw)",
    )

    # Request filtering
    exclude_paths: list[str] = Field(
        default_factory=lambda: ["/health", "/metrics", "/readyz", "/livez"],
        description="Request paths to exclude from tracing",
    )
    include_paths: list[str] = Field(
        default_factory=list, description="If specified, only trace these paths"
    )

    # Privacy & security
    exclude_headers: list[str] = Field(
        default_factory=lambda: [
            "authorization",
            "x-api-key",
            "cookie",
            "x-auth-token",
        ],
        description="Headers to redact in raw logs",
    )
    redact_sensitive: bool = Field(
        default=True, description="Redact sensitive data in structured logs"
    )

    # Performance settings
    max_body_size: int = Field(
        default=10485760,  # 10MB
        description="Maximum body size to log (bytes)",
    )
    truncate_body_preview: int = Field(
        default=1024,
        description="Maximum body preview size for structured logs (chars)",
    )

    # Granular control
    log_client_request: bool = Field(default=True, description="Log client requests")
    log_client_response: bool = Field(default=True, description="Log client responses")
    log_provider_request: bool = Field(
        default=True, description="Log provider requests"
    )
    log_provider_response: bool = Field(
        default=True, description="Log provider responses"
    )

    # Streaming configuration
    log_streaming_chunks: bool = Field(
        default=False, description="Log individual streaming chunks (verbose)"
    )

    model_config = ConfigDict()

    def get_json_log_dir(self) -> str:
        """Get directory for structured JSON logs."""
        return self.request_log_dir or self.log_dir

    def get_raw_log_dir(self) -> str:
        """Get directory for raw HTTP logs."""
        return self.raw_log_dir or self.log_dir

    def should_trace_path(self, path: str) -> bool:
        """Check if a path should be traced based on include/exclude rules."""
        # First check exclude_paths (takes precedence)
        if any(path.startswith(exclude) for exclude in self.exclude_paths):
            return False

        # Then check include_paths (if specified, only log included paths)
        if self.include_paths:
            return any(path.startswith(include) for include in self.include_paths)

        # Default: trace all paths not explicitly excluded
        return True
