"""Transformer pipeline service for orchestrating multiple transformations."""

from typing import Any

from ccproxy.core.interfaces import RequestTransformer, ResponseTransformer
from ccproxy.core.types import TransformContext


class TransformerPipeline:
    """Orchestrates multiple transformers in sequence.

    This service-layer class handles the business logic of applying
    multiple transformations in a specific order.
    """

    def __init__(
        self, transformers: list[RequestTransformer | ResponseTransformer]
    ) -> None:
        """Initialize with a list of transformers to chain.

        Args:
            transformers: List of transformers to apply in sequence
        """
        self.transformers = transformers

    async def transform_request(
        self, request: dict[str, Any], context: TransformContext | None = None
    ) -> dict[str, Any]:
        """Apply all request transformers in sequence.

        Args:
            request: The request data to transform
            context: Optional transformation context

        Returns:
            The result of applying all request transformers

        Raises:
            ValueError: If transformer sequence contains non-request transformers
        """
        result = request
        for transformer in self.transformers:
            if not isinstance(transformer, RequestTransformer):
                msg = f"Expected RequestTransformer, got {type(transformer)}"
                raise ValueError(msg)
            result = await transformer.transform_request(result)
        return result

    async def transform_response(
        self, response: dict[str, Any], context: TransformContext | None = None
    ) -> dict[str, Any]:
        """Apply all response transformers in sequence.

        Args:
            response: The response data to transform
            context: Optional transformation context

        Returns:
            The result of applying all response transformers

        Raises:
            ValueError: If transformer sequence contains non-response transformers
        """
        result = response
        for transformer in self.transformers:
            if not isinstance(transformer, ResponseTransformer):
                msg = f"Expected ResponseTransformer, got {type(transformer)}"
                raise ValueError(msg)
            result = await transformer.transform_response(result)
        return result
