"""Compatibility shim for converting between dict-based and typed adapter interfaces."""

from __future__ import annotations

from collections.abc import AsyncGenerator, AsyncIterator
from typing import Any, TYPE_CHECKING

from pydantic import BaseModel, ValidationError

from ccproxy.adapters.base import APIAdapter as LegacyAPIAdapter, BaseAPIAdapter as LegacyBaseAPIAdapter

if TYPE_CHECKING:
    from ccproxy.llms.adapters.base import BaseAPIAdapter
else:
    from ccproxy.llms.adapters.base import BaseAPIAdapter


class AdapterShim(LegacyBaseAPIAdapter):
    """Shim that wraps typed adapters to provide legacy dict-based interface.

    This allows the new strongly-typed adapters from ccproxy.llms.adapters
    to work with existing code that expects dict[str, Any] interfaces.

    The shim automatically converts between dict and BaseModel formats:
    - Incoming dicts are converted to generic BaseModels
    - Outgoing BaseModels are converted back to dicts
    - All error handling is preserved with meaningful messages
    """

    def __init__(self, typed_adapter: BaseAPIAdapter[Any, Any, Any]):
        """Initialize shim with a typed adapter.

        Args:
            typed_adapter: The strongly-typed adapter to wrap
        """
        super().__init__(name=f"shim_{typed_adapter.name}")
        self._typed_adapter = typed_adapter

    async def adapt_request(self, request: dict[str, Any]) -> dict[str, Any]:
        """Convert request using shim - dict to BaseModel and back."""
        try:
            # Convert dict to generic BaseModel
            typed_request = self._dict_to_model(request, "request")

            # Call the typed adapter
            typed_response = await self._typed_adapter.adapt_request(typed_request)

            # Convert back to dict
            return self._model_to_dict(typed_response)

        except ValidationError as e:
            raise ValueError(f"Invalid request format for {self._typed_adapter.name}: {e}") from e
        except Exception as e:
            raise ValueError(f"Request adaptation failed in {self._typed_adapter.name}: {e}") from e

    async def adapt_response(self, response: dict[str, Any]) -> dict[str, Any]:
        """Convert response using shim - dict to BaseModel and back."""
        try:
            # Convert dict to generic BaseModel
            typed_response = self._dict_to_model(response, "response")

            # Call the typed adapter
            typed_result = await self._typed_adapter.adapt_response(typed_response)

            # Convert back to dict
            return self._model_to_dict(typed_result)

        except ValidationError as e:
            raise ValueError(f"Invalid response format for {self._typed_adapter.name}: {e}") from e
        except Exception as e:
            raise ValueError(f"Response adaptation failed in {self._typed_adapter.name}: {e}") from e

    async def adapt_stream(
        self, stream: AsyncIterator[dict[str, Any]]
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Convert streaming response using shim."""
        async def typed_stream() -> AsyncGenerator[BaseModel, None]:
            """Convert dict stream to typed stream."""
            async for chunk in stream:
                try:
                    yield self._dict_to_model(chunk, "stream_chunk")
                except ValidationError as e:
                    raise ValueError(f"Invalid stream chunk format for {self._typed_adapter.name}: {e}") from e

        # Get the typed stream from the adapter
        typed_stream_result = self._typed_adapter.adapt_stream(typed_stream())

        # Convert back to dict stream
        async for typed_chunk in typed_stream_result:
            try:
                yield self._model_to_dict(typed_chunk)
            except Exception as e:
                raise ValueError(f"Stream chunk conversion failed in {self._typed_adapter.name}: {e}") from e

    async def adapt_error(self, error: dict[str, Any]) -> dict[str, Any]:
        """Convert error using shim - dict to BaseModel and back."""
        try:
            # Convert dict to generic BaseModel
            typed_error = self._dict_to_model(error, "error")

            # Call the typed adapter
            typed_result = await self._typed_adapter.adapt_error(typed_error)

            # Convert back to dict
            return self._model_to_dict(typed_result)

        except ValidationError as e:
            raise ValueError(f"Invalid error format for {self._typed_adapter.name}: {e}") from e
        except Exception as e:
            raise ValueError(f"Error adaptation failed in {self._typed_adapter.name}: {e}") from e

    def _dict_to_model(self, data: dict[str, Any], context: str) -> BaseModel:
        """Convert dict to BaseModel.

        Creates a generic BaseModel that accepts arbitrary fields from the dict.
        This allows the typed adapter to receive a BaseModel interface while
        preserving all the original dict data.

        Args:
            data: Dictionary to convert
            context: Context string for error messages

        Returns:
            BaseModel instance with all dict data as fields
        """
        try:
            # Create a dynamic model class that accepts all fields
            class GenericModel(BaseModel):
                """Generic model that accepts arbitrary fields."""
                class Config:
                    extra = "allow"  # Allow additional fields
                    arbitrary_types_allowed = True  # Allow arbitrary types

            # Create instance with all dict items as fields
            return GenericModel(**data)
        except Exception as e:
            raise ValueError(f"Failed to convert {context} dict to BaseModel: {e}") from e

    def _model_to_dict(self, model: BaseModel) -> dict[str, Any]:
        """Convert BaseModel to dict.

        Args:
            model: BaseModel instance to convert

        Returns:
            Dictionary representation of the model
        """
        try:
            return model.model_dump()
        except Exception as e:
            raise ValueError(f"Failed to convert BaseModel to dict: {e}") from e

    def __str__(self) -> str:
        return f"AdapterShim({self._typed_adapter})"

    def __repr__(self) -> str:
        return self.__str__()

    @property
    def wrapped_adapter(self) -> BaseAPIAdapter:
        """Get the underlying typed adapter.

        This allows code to access the original typed adapter if needed
        for direct typed operations.
        """
        return self._typed_adapter