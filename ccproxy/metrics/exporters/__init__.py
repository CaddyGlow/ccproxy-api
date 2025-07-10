"""Metrics exporters for various backends and formats."""

from .base import (
    BaseMetricsExporter,
    ExporterConnectionError,
    ExporterError,
    ExporterTimeoutError,
    ExporterValidationError,
)
from .json_api import JsonApiExporter


# Optional Prometheus support
try:
    from .prometheus import PrometheusExporter

    _PROMETHEUS_AVAILABLE = True
except ImportError:
    _PROMETHEUS_AVAILABLE = False
    PrometheusExporter = None  # type: ignore

__all__ = [
    "BaseMetricsExporter",
    "ExporterError",
    "ExporterConnectionError",
    "ExporterValidationError",
    "ExporterTimeoutError",
    "JsonApiExporter",
]

# Add Prometheus exporter if available
if _PROMETHEUS_AVAILABLE and PrometheusExporter is not None:
    __all__.append("PrometheusExporter")
