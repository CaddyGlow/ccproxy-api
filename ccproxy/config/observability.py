"""Observability configuration settings."""

from __future__ import annotations

from pydantic import BaseModel, Field


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

    duckdb_enabled: bool = Field(
        default=True,
        description="Enable DuckDB storage for metrics",
    )

    duckdb_path: str = Field(
        default="data/metrics.duckdb",
        description="Path to DuckDB database file",
    )
