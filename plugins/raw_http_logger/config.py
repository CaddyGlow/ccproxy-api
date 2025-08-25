"""Configuration for raw HTTP logger plugin."""

from pydantic import BaseModel, Field


class RawHTTPLoggerConfig(BaseModel):
    """Configuration for raw HTTP logging plugin."""

    enabled: bool = Field(default=True, description="Enable raw HTTP logging")

    log_dir: str = Field(
        default="/tmp/ccproxy/raw", description="Directory to store raw HTTP logs"
    )

    log_client_request: bool = Field(
        default=True, description="Log raw client requests"
    )

    log_client_response: bool = Field(
        default=True, description="Log raw client responses"
    )

    log_provider_request: bool = Field(
        default=True, description="Log raw provider requests"
    )

    log_provider_response: bool = Field(
        default=True, description="Log raw provider responses"
    )

    max_body_size: int = Field(
        default=10485760,  # 10MB
        description="Maximum body size to log (in bytes)",
    )

    include_paths: list[str] = Field(
        default_factory=lambda: ["/api", "/claude", "/codex"],
        description="Paths to include in logging (if empty, all paths are included)",
    )

    exclude_paths: list[str] = Field(
        default_factory=list,
        description="Paths to exclude from logging (takes precedence over include_paths)",
    )

    exclude_headers: list[str] = Field(
        default_factory=lambda: ["authorization", "cookie", "x-api-key"],
        description="Headers to exclude from logging (for security)",
    )
