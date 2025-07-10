"""Base classes and interfaces for metrics exporters."""

from abc import ABC, abstractmethod
from typing import Any


class ExporterError(Exception):
    """Base exception for metrics exporters."""

    pass


class ExporterConnectionError(ExporterError):
    """Connection error for metrics exporters."""

    pass


class ExporterValidationError(ExporterError):
    """Validation error for metrics exporters."""

    pass


class ExporterTimeoutError(ExporterError):
    """Timeout error for metrics exporters."""

    pass


class BaseMetricsExporter(ABC):
    """Abstract base class for metrics exporters."""

    @abstractmethod
    async def export_metrics(self, metrics_data: Any) -> None:
        """Export metrics data.

        Args:
            metrics_data: The metrics data to export
        """
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the exporter is healthy.

        Returns:
            True if healthy, False otherwise
        """
        pass
