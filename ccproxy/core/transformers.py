"""Core transformer abstractions for request/response transformation."""

from abc import ABC, abstractmethod
from typing import Any, Optional, Protocol, TypeVar, runtime_checkable

from ccproxy.core.types import ProxyRequest, ProxyResponse, TransformContext


T = TypeVar("T", contravariant=True)
R = TypeVar("R", covariant=True)


class BaseTransformer(ABC):
    """Abstract base class for all transformers."""

    @abstractmethod
    async def transform(
        self, data: Any, context: TransformContext | None = None
    ) -> Any:
        """Transform the input data.

        Args:
            data: The data to transform
            context: Optional transformation context

        Returns:
            The transformed data

        Raises:
            TransformationError: If transformation fails
        """
        pass


class RequestTransformer(BaseTransformer):
    """Base class for request transformers."""

    @abstractmethod
    async def transform(
        self, request: ProxyRequest, context: TransformContext | None = None
    ) -> ProxyRequest:
        """Transform a proxy request.

        Args:
            request: The request to transform
            context: Optional transformation context

        Returns:
            The transformed request
        """
        pass


class ResponseTransformer(BaseTransformer):
    """Base class for response transformers."""

    @abstractmethod
    async def transform(
        self, response: ProxyResponse, context: TransformContext | None = None
    ) -> ProxyResponse:
        """Transform a proxy response.

        Args:
            response: The response to transform
            context: Optional transformation context

        Returns:
            The transformed response
        """
        pass


@runtime_checkable
class TransformerProtocol(Protocol[T, R]):
    """Protocol defining the transformer interface."""

    async def transform(self, data: T, context: TransformContext | None = None) -> R:
        """Transform the input data."""
        ...


class ChainedTransformer(BaseTransformer):
    """Transformer that chains multiple transformers together."""

    def __init__(self, transformers: list[BaseTransformer]):
        """Initialize with a list of transformers to chain.

        Args:
            transformers: List of transformers to apply in sequence
        """
        self.transformers = transformers

    async def transform(
        self, data: Any, context: TransformContext | None = None
    ) -> Any:
        """Apply all transformers in sequence.

        Args:
            data: The data to transform
            context: Optional transformation context

        Returns:
            The result of applying all transformers
        """
        result = data
        for transformer in self.transformers:
            result = await transformer.transform(result, context)
        return result
