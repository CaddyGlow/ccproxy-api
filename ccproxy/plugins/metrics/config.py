"""Configuration for the metrics plugin."""

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class MetricsConfig(BaseModel):
    """Configuration for the metrics plugin.

    This configuration controls Prometheus metrics collection,
    export endpoints, and Pushgateway integration.
    """

    # Basic settings
    enabled: bool = Field(default=True, description="Enable metrics collection")

    namespace: str = Field(
        default="ccproxy", description="Prometheus metric namespace prefix"
    )

    # Endpoint configuration
    metrics_endpoint_enabled: bool = Field(
        default=True, description="Enable /metrics endpoint for Prometheus scraping"
    )

    # Pushgateway configuration
    pushgateway_enabled: bool = Field(
        default=False, description="Enable Pushgateway integration for batch metrics"
    )

    pushgateway_url: str | None = Field(
        default=None, description="Pushgateway URL (e.g., http://localhost:9091)"
    )

    pushgateway_job: str = Field(
        default="ccproxy", description="Job name for Pushgateway"
    )

    pushgateway_push_interval: int = Field(
        default=60, description="Interval in seconds between pushes to Pushgateway"
    )

    # Collection settings
    collect_request_metrics: bool = Field(
        default=True, description="Collect request/response metrics"
    )

    collect_token_metrics: bool = Field(
        default=True, description="Collect token usage metrics"
    )

    collect_cost_metrics: bool = Field(default=True, description="Collect cost metrics")

    collect_error_metrics: bool = Field(
        default=True, description="Collect error metrics"
    )

    collect_pool_metrics: bool = Field(
        default=True, description="Collect connection pool metrics"
    )

    # Performance settings
    histogram_buckets: list[float] = Field(
        default=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 25.0],
        description="Histogram buckets for response time metrics (in seconds)",
    )

    # Grafana dashboard settings
    grafana_dashboards_path: Path | None = Field(
        default=None, description="Path to Grafana dashboards directory"
    )

    def model_post_init(self, __context: Any) -> None:
        """Post-initialization setup."""
        super().model_post_init(__context)

        # Set default Grafana path if not specified
        if self.grafana_dashboards_path is None:
            # Use plugin's grafana directory
            from pathlib import Path

            plugin_dir = Path(__file__).parent
            self.grafana_dashboards_path = plugin_dir / "grafana"
