"""Base adapter interface for API format conversion."""

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, AsyncIterator
from typing import Any

from pydantic import BaseModel

from ccproxy.core.interfaces import StreamingConfigurable


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

    @abstractmethod
    async def adapt_error(self, error: dict[str, Any]) -> dict[str, Any]:
        """Convert an error response from one API format to another.

        Args:
            error: The error response data to convert

        Returns:
            The converted error response data

        Raises:
            ValueError: If the error format is invalid or unsupported
        """
        pass


class BaseAPIAdapter(APIAdapter, StreamingConfigurable):
    """Base implementation with common functionality.

    Provides dual interface support:
    - Legacy dict-based methods for backward compatibility
    - New strongly-typed methods for better type safety

    Subclasses should implement the typed methods, and the legacy methods
    will automatically delegate to them with appropriate conversions.
    """

    def __init__(self, name: str):
        self.name = name
        # Optional streaming flags that subclasses may use
        self._openai_thinking_xml: bool | None = None

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.name})"

    def __repr__(self) -> str:
        return self.__str__()

    # StreamingConfigurable
    def configure_streaming(self, *, openai_thinking_xml: bool | None = None) -> None:
        self._openai_thinking_xml = openai_thinking_xml

    # Legacy dict interface - delegates to typed methods
    async def adapt_request(self, request: dict[str, Any]) -> dict[str, Any]:
        """Legacy dict interface - calls typed implementation internally."""
        typed_request = self._dict_to_request_model(request)
        typed_response = await self.adapt_request_typed(typed_request)
        return typed_response.model_dump()

    async def adapt_response(self, response: dict[str, Any]) -> dict[str, Any]:
        """Legacy dict interface - calls typed implementation internally."""
        typed_response = self._dict_to_response_model(response)
        typed_result = await self.adapt_response_typed(typed_response)
        return typed_result.model_dump()

    def adapt_stream(
        self, stream: AsyncIterator[dict[str, Any]]
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Legacy dict interface - calls typed implementation internally."""

        async def dict_generator() -> AsyncGenerator[dict[str, Any], None]:
            typed_stream = self._dict_stream_to_typed_stream(stream)
            async for typed_chunk in self.adapt_stream_typed(typed_stream):
                yield typed_chunk.model_dump()

        return dict_generator()

    async def adapt_error(self, error: dict[str, Any]) -> dict[str, Any]:
        """Legacy dict interface - calls typed implementation internally."""
        typed_error = self._dict_to_error_model(error)
        typed_result = await self.adapt_error_typed(typed_error)
        return typed_result.model_dump()

    # New strongly-typed interface - subclasses implement these
    @abstractmethod
    async def adapt_request_typed(self, request: BaseModel) -> BaseModel:
        """Convert a request using strongly-typed Pydantic models."""
        pass

    @abstractmethod
    async def adapt_response_typed(self, response: BaseModel) -> BaseModel:
        """Convert a response using strongly-typed Pydantic models."""
        pass

    @abstractmethod
    def adapt_stream_typed(
        self, stream: AsyncIterator[BaseModel]
    ) -> AsyncGenerator[BaseModel, None]:
        """Convert a streaming response using strongly-typed Pydantic models."""
        # This should be implemented as an async generator
        # Subclasses must override this method
        ...

    @abstractmethod
    async def adapt_error_typed(self, error: BaseModel) -> BaseModel:
        """Convert an error response using strongly-typed Pydantic models."""
        pass

    # Helper methods for model conversion - subclasses implement these
    @abstractmethod
    def _dict_to_request_model(self, request: dict[str, Any]) -> BaseModel:
        """Convert dict to appropriate request model."""
        pass

    @abstractmethod
    def _dict_to_response_model(self, response: dict[str, Any]) -> BaseModel:
        """Convert dict to appropriate response model."""
        pass

    @abstractmethod
    def _dict_to_error_model(self, error: dict[str, Any]) -> BaseModel:
        """Convert dict to appropriate error model."""
        pass

    @abstractmethod
    def _dict_stream_to_typed_stream(
        self, stream: AsyncIterator[dict[str, Any]]
    ) -> AsyncIterator[BaseModel]:
        """Convert dict stream to typed stream."""
        pass


__all__ = ["APIAdapter", "BaseAPIAdapter"]
