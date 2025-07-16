"""Observability configuration settings."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class ObservabilitySettings(BaseModel):
    """Observability configuration settings."""

    metrics_enabled: bool = Field(
        default=True,
        description="Enable metrics endpoint",
    )

    pushgateway_enabled: bool = Field(
        default=False,
        description="Enable Prometheus Pushgateway integration",
    )

    pushgateway_url: str | None = Field(
        default=None,
        description="Pushgateway URL (e.g., http://pushgateway:9091)",
    )

    pushgateway_job: str = Field(
        default="ccproxy",
        description="Job name for Pushgateway metrics",
    )

    pushgateway_batch_interval: float = Field(
        default=5.0,
        description="Interval in seconds for pushing metrics to Pushgateway",
    )

    duckdb_enabled: bool = Field(
        default=True,
        description="Enable DuckDB storage for metrics",
    )

    duckdb_path: str = Field(
        default="data/metrics.duckdb",
        description="Path to DuckDB database file",
    )

    # Enhanced logging integration
    logging_pipeline_enabled: bool = Field(
        default=True,
        description="Enable structlog pipeline integration for observability",
    )

    logging_format: str = Field(
        default="auto",
        description="Logging format for observability: 'rich', 'json', 'auto' (auto-detects based on environment)",
    )

    @field_validator("logging_format")
    @classmethod
    def validate_logging_format(cls, v: str) -> str:
        """Validate and normalize logging format."""
        lower_v = v.lower()
        valid_formats = ["auto", "rich", "json", "plain"]
        if lower_v not in valid_formats:
            raise ValueError(
                f"Invalid logging format: {v}. Must be one of {valid_formats}"
            )
        return lower_v

    @field_validator("pushgateway_batch_interval")
    @classmethod
    def validate_batch_interval(cls, v: float) -> float:
        """Validate batch interval is within reasonable bounds."""
        if v < 0.1:
            raise ValueError("Batch interval must be at least 0.1 seconds")
        if v > 300:
            raise ValueError("Batch interval must not exceed 300 seconds (5 minutes)")
        return v

    @property
    def enabled(self) -> bool:
        """Check if observability is enabled (backward compatibility property)."""
        return (
            self.metrics_enabled
            or self.pushgateway_enabled
            or self.duckdb_enabled
            or self.logging_pipeline_enabled
        )
