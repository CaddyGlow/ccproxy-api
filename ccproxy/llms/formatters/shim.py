"""Compatibility shim for converting between dict-based and typed adapter interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, AsyncIterator
from typing import Any, get_args, get_origin

from pydantic import BaseModel, ConfigDict, ValidationError

from ccproxy.llms.formatters.base import BaseAPIAdapter

from ..models.openai import AnyStreamEvent


class DictBasedAdapterProtocol(ABC):
    """Protocol for adapters that work with dict interfaces."""

    @abstractmethod
    async def adapt_request(self, request: dict[str, Any]) -> dict[str, Any]:
        """Convert a request using dict interface."""
        pass

    @abstractmethod
    async def adapt_response(self, response: dict[str, Any]) -> dict[str, Any]:
        """Convert a response using dict interface."""
        pass

    @abstractmethod
    def adapt_stream(
        self, stream: AsyncIterator[dict[str, Any]]
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Convert a streaming response using dict interface."""
        pass

    @abstractmethod
    async def adapt_error(self, error: dict[str, Any]) -> dict[str, Any]:
        """Convert an error response using dict interface."""
        pass


class AdapterShim(DictBasedAdapterProtocol):
    """Shim that wraps typed adapters to provide legacy dict-based interface.

    This allows the new strongly-typed adapters from ccproxy.llms.formatters
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
        self.name = f"shim_{typed_adapter.name}"
        self._typed_adapter = typed_adapter
        # Discovered model types from the typed adapter's generic parameters
        self._request_model: type[BaseModel] | None = None
        self._response_model: type[BaseModel] | None = None
        self._stream_event_model: type[BaseModel] | None = None

        self._introspect_model_types()

    def _introspect_model_types(self) -> None:
        """Discover the generic type arguments declared by the typed adapter.

        Reads BaseAPIAdapter[Req, Resp, Stream] from the class to avoid guesswork.
        """
        try:
            for base in getattr(self._typed_adapter.__class__, "__orig_bases__", ()):
                if get_origin(base) is BaseAPIAdapter:
                    args = get_args(base)
                    if len(args) == 3:
                        req, resp, stream = args
                        if (
                            isinstance(req, type)
                            and issubclass(req, BaseModel)
                            and req is not BaseModel
                        ):
                            self._request_model = req
                        if (
                            isinstance(resp, type)
                            and issubclass(resp, BaseModel)
                            and resp is not BaseModel
                        ):
                            self._response_model = resp
                        if (
                            isinstance(stream, type)
                            and issubclass(stream, BaseModel)
                            and stream is not BaseModel
                        ):
                            self._stream_event_model = stream
                    break
        except Exception:
            # Best-effort only; fall back to inference/generic model path
            pass

    async def adapt_request(self, request: dict[str, Any]) -> dict[str, Any]:
        """Convert request using shim - dict to BaseModel and back."""
        try:
            # Convert dict to typed model (strict: requires declared model)
            typed_request = self._dict_to_model(
                request, "request", preferred_model=self._request_model
            )

            # Call the typed adapter
            typed_response = await self._typed_adapter.adapt_request(typed_request)

            # Convert back to dict
            return self._model_to_dict(typed_response)

        except ValidationError as e:
            raise ValueError(
                f"Invalid request format for {self._typed_adapter.name}: {e}"
            ) from e
        except Exception as e:
            raise ValueError(
                f"Request adaptation failed in {self._typed_adapter.name}: {e}"
            ) from e

    async def adapt_response(self, response: dict[str, Any]) -> dict[str, Any]:
        """Convert response using shim - dict to BaseModel and back."""
        try:
            # Convert dict to typed model (strict: requires declared model)
            typed_response = self._dict_to_model(
                response, "response", preferred_model=self._response_model
            )

            # Call the typed adapter
            typed_result = await self._typed_adapter.adapt_response(typed_response)

            # Convert back to dict
            return self._model_to_dict(typed_result)

        except ValidationError as e:
            raise ValueError(
                f"Invalid response format for {self._typed_adapter.name}: {e}"
            ) from e
        except Exception as e:
            raise ValueError(
                f"Response adaptation failed in {self._typed_adapter.name}: {e}"
            ) from e

    def adapt_stream(
        self, stream: AsyncIterator[dict[str, Any]]
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Convert streaming response using shim."""
        return self._adapt_stream_impl(stream)

    async def _adapt_stream_impl(
        self, stream: AsyncIterator[dict[str, Any]]
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Internal implementation for stream adaptation."""

        async def typed_stream() -> AsyncGenerator[BaseModel, None]:
            """Convert dict stream to typed stream."""
            async for chunk in stream:
                try:
                    yield self._dict_to_model(
                        chunk, "stream_chunk", preferred_model=self._stream_event_model
                    )
                except ValidationError as e:
                    raise ValueError(
                        f"Invalid stream chunk format for {self._typed_adapter.name}: {e}"
                    ) from e

        # Get the typed stream from the adapter
        typed_stream_result = self._typed_adapter.adapt_stream(typed_stream())

        # Convert back to dict stream
        async for typed_chunk in typed_stream_result:
            try:
                yield self._model_to_dict(typed_chunk)
            except Exception as e:
                raise ValueError(
                    f"Stream chunk conversion failed in {self._typed_adapter.name}: {e}"
                ) from e

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
            raise ValueError(
                f"Invalid error format for {self._typed_adapter.name}: {e}"
            ) from e
        except Exception as e:
            raise ValueError(
                f"Error adaptation failed in {self._typed_adapter.name}: {e}"
            ) from e

    def _dict_to_model(
        self,
        data: dict[str, Any],
        context: str,
        *,
        preferred_model: type[BaseModel] | None = None,
    ) -> BaseModel:
        """Convert dict to appropriate BaseModel based on content.

        This method intelligently determines the correct Pydantic model type
        based on the dictionary contents and converts accordingly.

        Args:
            data: Dictionary to convert
            context: Context string for error messages

        Returns:
            BaseModel instance of the appropriate type
        """
        try:
            # Use the discovered model type when available
            if preferred_model is not None:
                return preferred_model.model_validate(data)

            # Strict mode: require declared model types for request/response/stream
            if context != "error":
                if context == "stream_chunk":
                    try:
                        return AnyStreamEvent.model_validate(data)
                    except Exception:
                        pass
                raise ValueError(
                    f"Strict shim: {context} model type not declared by {type(self._typed_adapter).__name__}. "
                    "Ensure the adapter specifies concrete generic type parameters."
                )

            # Error context: build a minimal structured error model so nested
            # attributes like `error.message` are accessible to consumers.
            class SimpleErrorDetail(BaseModel):
                message: str | None = None
                type: str | None = None
                code: str | None = None
                param: str | None = None

            class SimpleError(BaseModel):
                error: SimpleErrorDetail

            try:
                return SimpleError.model_validate(data)
            except Exception:
                # Fallback to permissive generic model if structure is unexpected
                class GenericModel(BaseModel):
                    model_config = ConfigDict(
                        extra="allow", arbitrary_types_allowed=True
                    )

                return GenericModel(**data)
        except Exception as e:
            raise ValueError(
                f"Failed to convert {context} dict to BaseModel: {e}"
            ) from e

    # Heuristic inference removed for strictness

    def _model_to_dict(self, model: BaseModel) -> dict[str, Any]:
        """Convert BaseModel to dict.

        Args:
            model: BaseModel instance to convert

        Returns:
            Dictionary representation of the model
        """
        try:
            # Don't pass exclude_none here as LlmBaseModel handles it internally
            # to avoid "multiple values for keyword argument 'exclude_none'" error
            return model.model_dump(mode="json", exclude_unset=True)
        except Exception as e:
            raise ValueError(f"Failed to convert BaseModel to dict: {e}") from e

    def __str__(self) -> str:
        return f"AdapterShim({self._typed_adapter})"

    def __repr__(self) -> str:
        return self.__str__()

    @property
    def wrapped_adapter(self) -> BaseAPIAdapter[Any, Any, Any]:
        """Get the underlying typed adapter.

        This allows code to access the original typed adapter if needed
        for direct typed operations.
        """
        return self._typed_adapter
