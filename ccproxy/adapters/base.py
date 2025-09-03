"""Base adapter interface for API format conversion."""

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, AsyncIterator
from typing import Any


class APIAdapter(ABC):
    """Abstract base class for API format adapters.

    Combines all transformation interfaces to provide a complete adapter
    for converting between different API formats.
    """

    @abstractmethod
    async def adapt_request(self, request: dict[str, Any]) -> dict[str, Any]:
        """Convert a request from one API format to another.

        Args:
            request: The request data to convert

        Returns:
            The converted request data

        Raises:
            ValueError: If the request format is invalid or unsupported
        """
        pass

    @abstractmethod
    async def adapt_response(self, response: dict[str, Any]) -> dict[str, Any]:
        """Convert a response from one API format to another.

        Args:
            response: The response data to convert

        Returns:
            The converted response data

        Raises:
            ValueError: If the response format is invalid or unsupported
        """
        pass

    @abstractmethod
    def adapt_stream(
        self, stream: AsyncIterator[dict[str, Any]]
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Convert a streaming response from one API format to another.

        Args:
            stream: The streaming response data to convert

        Yields:
            The converted streaming response chunks

        Raises:
            ValueError: If the stream format is invalid or unsupported
        """
        # This should be implemented as an async generator
        # Subclasses must override this method
        ...


class BaseAPIAdapter(APIAdapter):
    """Base implementation with common functionality."""

    def __init__(self, name: str):
        self.name = name

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.name})"

    def __repr__(self) -> str:
        return self.__str__()


__all__ = ["APIAdapter", "BaseAPIAdapter"]
