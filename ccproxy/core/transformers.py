"""Core transformer abstractions for request/response transformation."""

import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Protocol, TypeVar, runtime_checkable

from ccproxy.core.logging import get_logger
from ccproxy.core.types import ProxyRequest, ProxyResponse, TransformContext


if TYPE_CHECKING:
    pass


T = TypeVar("T", contravariant=True)
R = TypeVar("R", covariant=True)


class BaseTransformer(ABC):
    """Abstract base class for all transformers."""

    def __init__(self) -> None:
        """Initialize transformer."""
        self.metrics_collector: Any = None

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

    async def _collect_transformation_metrics(
        self,
        transformation_type: str,
        input_data: Any,
        output_data: Any,
        duration_ms: float,
        success: bool = True,
        error: str | None = None,
    ) -> None:
        """Collect metrics for transformation operations.

        Args:
            transformation_type: Type of transformation (request/response)
            input_data: Original input data
            output_data: Transformed output data
            duration_ms: Time taken for transformation in milliseconds
            success: Whether transformation was successful
            error: Error message if transformation failed
        """
        if not self.metrics_collector:
            return

        try:
            # Calculate data sizes
            # input_size = self._calculate_data_size(input_data)
            # output_size = self._calculate_data_size(output_data) if output_data else 0

            # Create a unique request ID for this transformation
            request_id = (
                f"transformer_{id(self)}_{transformation_type}_{id(input_data)}"
            )

            # Use existing latency collection method with timing data
            await self.metrics_collector.collect_latency(
                request_id=request_id,
                transformation_duration=duration_ms,
                processing_time=duration_ms,
            )

        except (AttributeError, TypeError) as e:
            # Don't let metrics collection fail the transformation
            logger = get_logger(__name__)
            # logger = logging.getLogger(__name__)
            logger.debug(
                "transformation_metrics_attribute_error",
                error=str(e),
                exc_info=e,
            )
        except Exception as e:
            # Don't let metrics collection fail the transformation
            logger = get_logger(__name__)
            # logger = logging.getLogger(__name__)
            logger.debug(
                "transformation_metrics_failed",
                error=str(e),
                operation="collect_transformation_metrics",
            )

    def _calculate_data_size(self, data: Any) -> int:
        """Calculate the size of data in bytes.

        Args:
            data: The data to measure

        Returns:
            Size in bytes
        """
        if data is None:
            return 0
        elif isinstance(data, bytes):
            return len(data)
        elif isinstance(data, str):
            return len(data.encode("utf-8"))
        elif hasattr(data, "__len__"):
            return len(str(data))
        else:
            return len(str(data))


class RequestTransformer(BaseTransformer):
    """Base class for request transformers."""

    def __init__(self, proxy_mode: str = "full") -> None:
        """Initialize request transformer with proxy mode."""
        super().__init__()
        self.proxy_mode = proxy_mode

    async def transform(
        self, request: ProxyRequest, context: TransformContext | None = None
    ) -> ProxyRequest:
        """Transform a proxy request with metrics collection.

        Args:
            request: The request to transform
            context: Optional transformation context

        Returns:
            The transformed request
        """
        start_time = time.perf_counter()
        error_msg = None
        result = None

        try:
            result = await self._transform_request(request, context)
            return result
        except Exception as e:
            error_msg = str(e)
            raise
        finally:
            # Collect metrics regardless of success/failure
            duration_ms = (time.perf_counter() - start_time) * 1000
            await self._collect_transformation_metrics(
                transformation_type="request",
                input_data=request,
                output_data=result,
                duration_ms=duration_ms,
                success=error_msg is None,
                error=error_msg,
            )

    @abstractmethod
    async def _transform_request(
        self, request: ProxyRequest, context: TransformContext | None = None
    ) -> ProxyRequest:
        """Transform a proxy request implementation.

        Args:
            request: The request to transform
            context: Optional transformation context

        Returns:
            The transformed request
        """
        pass


class ResponseTransformer(BaseTransformer):
    """Base class for response transformers."""

    async def transform(
        self, response: ProxyResponse, context: TransformContext | None = None
    ) -> ProxyResponse:
        """Transform a proxy response with metrics collection.

        Args:
            response: The response to transform
            context: Optional transformation context

        Returns:
            The transformed response
        """
        start_time = time.perf_counter()
        error_msg = None
        result = None

        try:
            result = await self._transform_response(response, context)
            return result
        except Exception as e:
            error_msg = str(e)
            raise
        finally:
            # Collect metrics regardless of success/failure
            duration_ms = (time.perf_counter() - start_time) * 1000
            await self._collect_transformation_metrics(
                transformation_type="response",
                input_data=response,
                output_data=result,
                duration_ms=duration_ms,
                success=error_msg is None,
                error=error_msg,
            )

    @abstractmethod
    async def _transform_response(
        self, response: ProxyResponse, context: TransformContext | None = None
    ) -> ProxyResponse:
        """Transform a proxy response implementation.

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
