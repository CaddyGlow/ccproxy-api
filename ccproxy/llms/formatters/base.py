"""Base adapter interface for API format conversion."""

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, AsyncIterator
from typing import Generic, TypeVar

from pydantic import BaseModel

from ccproxy.core.interfaces import StreamingConfigurable


RequestType = TypeVar("RequestType", bound=BaseModel)
ResponseType = TypeVar("ResponseType", bound=BaseModel)
StreamEventType = TypeVar("StreamEventType", bound=BaseModel)


class APIAdapter(ABC, Generic[RequestType, ResponseType, StreamEventType]):
    """Abstract base class for API format adapters.

    Provides strongly-typed interface for converting between different API formats
    with full type safety and validation.
    """

    @abstractmethod
    async def adapt_request(self, request: RequestType) -> BaseModel:
        """Convert a request using strongly-typed Pydantic models.

        Args:
            request: The typed request model to convert

        Returns:
            The converted typed request model

        Raises:
            ValueError: If the request format is invalid or unsupported
        """
        pass

    @abstractmethod
    async def adapt_response(self, response: ResponseType) -> BaseModel:
        """Convert a response using strongly-typed Pydantic models.

        Args:
            response: The typed response model to convert

        Returns:
            The converted typed response model

        Raises:
            ValueError: If the response format is invalid or unsupported
        """
        pass

    @abstractmethod
    def adapt_stream(
        self, stream: AsyncIterator[StreamEventType]
    ) -> AsyncGenerator[BaseModel, None]:
        """Convert a streaming response using strongly-typed Pydantic models.

        Args:
            stream: The typed streaming response data to convert

        Yields:
            The converted typed streaming response chunks

        Raises:
            ValueError: If the stream format is invalid or unsupported
        """
        # This should be implemented as an async generator
        # Subclasses must override this method
        ...

    @abstractmethod
    async def adapt_error(self, error: BaseModel) -> BaseModel:
        """Convert an error response using strongly-typed Pydantic models.

        Args:
            error: The typed error response model to convert

        Returns:
            The converted typed error response model

        Raises:
            ValueError: If the error format is invalid or unsupported
        """
        pass


class BaseAPIAdapter(
    APIAdapter[RequestType, ResponseType, StreamEventType],
    StreamingConfigurable,
):
    """Base implementation with common functionality.

    Provides strongly-typed interface for API format conversion with
    better type safety and validation.
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

    # Strongly-typed interface - subclasses implement these
    @abstractmethod
    async def adapt_request(self, request: RequestType) -> BaseModel:
        """Convert a request using strongly-typed Pydantic models."""
        pass

    @abstractmethod
    async def adapt_response(self, response: ResponseType) -> BaseModel:
        """Convert a response using strongly-typed Pydantic models."""
        pass

    @abstractmethod
    def adapt_stream(
        self, stream: AsyncIterator[StreamEventType]
    ) -> AsyncGenerator[BaseModel, None]:
        """Convert a streaming response using strongly-typed Pydantic models."""
        # This should be implemented as an async generator
        # Subclasses must override this method
        ...

    @abstractmethod
    async def adapt_error(self, error: BaseModel) -> BaseModel:
        """Convert an error response using strongly-typed Pydantic models."""
        pass


__all__ = ["APIAdapter", "BaseAPIAdapter"]
