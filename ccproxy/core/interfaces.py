"""Core interfaces and abstract base classes for the CCProxy API.

This module consolidates all abstract interfaces used throughout the application,
providing a single location for defining contracts and protocols.
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any, Protocol, TypeVar, runtime_checkable

from ccproxy.core.types import TransformContext


__all__ = [
    # Transformation interfaces
    "RequestTransformer",
    "ResponseTransformer",
    "StreamTransformer",
    "TransformerProtocol",
    # Storage interfaces
    # Metrics interfaces
    "MetricExporter",
    # Streaming configuration protocol
    "StreamingConfigurable",
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


@runtime_checkable
class StreamingConfigurable(Protocol):
    """Protocol for adapters that accept streaming-related configuration.

    Implementers can use this to receive DI-injected toggles such as whether
    to serialize thinking content as XML in OpenAI streams.
    """

    def configure_streaming(self, *, openai_thinking_xml: bool | None = None) -> None:
        """Apply streaming flags.

        Args:
            openai_thinking_xml: Enable/disable thinking-as-XML in OpenAI streams
        """
        ...
