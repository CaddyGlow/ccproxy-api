"""Core interfaces and abstract base classes for the CCProxy API.

This module consolidates all abstract interfaces used throughout the application,
providing a single location for defining contracts and protocols.
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any, Protocol, TypeVar, runtime_checkable

from ccproxy.auth.models import ClaudeCredentials
from ccproxy.core.types import TransformContext


__all__ = [
    # Transformation interfaces
    "RequestTransformer",
    "ResponseTransformer",
    "StreamTransformer",
    "TransformerProtocol",
    # Storage interfaces
    "TokenStorage",
    # Metrics interfaces
    "MetricExporter",
]


T = TypeVar("T", contravariant=True)
R = TypeVar("R", covariant=True)


# === Transformation Interfaces ===


class RequestTransformer(ABC):
    """Abstract interface for request transformers."""

    @abstractmethod
    async def transform_request(self, request: dict[str, Any]) -> dict[str, Any]:
        """Transform a request from one format to another.

        Args:
            request: The request data to transform

        Returns:
            The transformed request data

        Raises:
            ValueError: If the request format is invalid or unsupported
        """
        pass


class ResponseTransformer(ABC):
    """Abstract interface for response transformers."""

    @abstractmethod
    async def transform_response(self, response: dict[str, Any]) -> dict[str, Any]:
        """Transform a response from one format to another.

        Args:
            response: The response data to transform

        Returns:
            The transformed response data

        Raises:
            ValueError: If the response format is invalid or unsupported
        """
        pass


class StreamTransformer(ABC):
    """Abstract interface for stream transformers."""

    @abstractmethod
    async def transform_stream(
        self, stream: AsyncIterator[dict[str, Any]]
    ) -> AsyncIterator[dict[str, Any]]:
        """Transform a streaming response from one format to another.

        Args:
            stream: The streaming response data to transform

        Yields:
            The transformed streaming response chunks

        Raises:
            ValueError: If the stream format is invalid or unsupported
        """
        pass


@runtime_checkable
class TransformerProtocol(Protocol[T, R]):
    """Protocol defining the transformer interface."""

    async def transform(self, data: T, context: TransformContext | None = None) -> R:
        """Transform the input data."""
        ...


# === Storage Interfaces ===


class TokenStorage(ABC):
    """Abstract interface for token storage backends."""

    @abstractmethod
    async def load(self) -> ClaudeCredentials | None:
        """Load credentials from storage.

        Returns:
            Parsed credentials if found and valid, None otherwise
        """
        pass

    @abstractmethod
    async def save(self, credentials: ClaudeCredentials) -> bool:
        """Save credentials to storage.

        Args:
            credentials: Credentials to save

        Returns:
            True if saved successfully, False otherwise
        """
        pass

    @abstractmethod
    async def exists(self) -> bool:
        """Check if credentials exist in storage.

        Returns:
            True if credentials exist, False otherwise
        """
        pass

    @abstractmethod
    async def delete(self) -> bool:
        """Delete credentials from storage.

        Returns:
            True if deleted successfully, False otherwise
        """
        pass

    @abstractmethod
    def get_location(self) -> str:
        """Get the storage location description.

        Returns:
            Human-readable description of where credentials are stored
        """
        pass


# === Metrics Interfaces ===


class MetricExporter(ABC):
    """Abstract interface for exporting metrics to external systems."""

    @abstractmethod
    async def export_metrics(self, metrics: dict[str, Any]) -> bool:
        """Export metrics to the target system.

        Args:
            metrics: Dictionary of metrics to export

        Returns:
            True if export was successful, False otherwise

        Raises:
            ConnectionError: If unable to connect to the metrics backend
            ValueError: If metrics format is invalid
        """
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the metrics export system is healthy.

        Returns:
            True if the system is healthy, False otherwise
        """
        pass
