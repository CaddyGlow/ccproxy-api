"""Adapter wrapper that uses FormatterRegistry for format conversions."""

from __future__ import annotations

from collections.abc import AsyncGenerator, AsyncIterator, Callable
from typing import Any

from pydantic import BaseModel, ConfigDict, TypeAdapter

from ccproxy.llms.formatters.base import BaseAPIAdapter
from ccproxy.llms.formatters.formatter_registry import FormatterRegistry
from ccproxy.services.adapters.format_adapter import FormatAdapterProtocol


class FormatterGenericModel(BaseModel):
    """Generic model for FormatterRegistryAdapter with flexible field support."""

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)


class FormatterRegistryAdapter(
    BaseAPIAdapter[FormatterGenericModel, FormatterGenericModel, FormatterGenericModel],
    FormatAdapterProtocol,
):
    """Adapter that uses FormatterRegistry for format conversions.

    This adapter wraps formatter functions from the registry to provide
    the APIAdapter interface expected by the plugin system.
    """

    def __init__(
        self,
        formatter_registry: FormatterRegistry,
        source_format: str,
        target_format: str,
    ):
        """Initialize adapter with formatter registry and format pair.

        Args:
            formatter_registry: Registry containing conversion functions
            source_format: Source format (e.g., "anthropic.messages")
            target_format: Target format (e.g., "openai.responses")
        """
        # Initialize parent with a descriptive name
        super().__init__(name=f"{source_format}_to_{target_format}_formatter")

        self.formatter_registry = formatter_registry
        self.source_format = source_format
        self.target_format = target_format

        # Cache formatters for this conversion direction
        try:
            self.formatters = formatter_registry.get_all(source_format, target_format)
        except ValueError as e:
            raise ValueError(
                f"No formatters available for {source_format} -> {target_format}: {e}"
            ) from e

    # Required abstract methods from BaseAPIAdapter
    async def adapt_request(self, request: FormatterGenericModel) -> BaseModel:
        """Convert request using registry formatter."""
        if "request" not in self.formatters:
            raise ValueError(
                f"No request formatter available for {self.source_format} -> {self.target_format}"
            )

        # Extract dict from the generic model for formatter functions
        request_dict = request.model_dump()

        # Call the formatter function - but first convert dict to appropriate typed model
        converter = self.formatters["request"]

        # We need to convert the dict to the appropriate input model for the formatter
        # Based on the format pair, determine the correct input model type
        typed_request = self._dict_to_input_model(
            request_dict, self.source_format, "request"
        )

        import inspect

        if inspect.iscoroutinefunction(converter):
            result = await converter(typed_request)
        else:
            result = converter(typed_request)

        # Convert result to FormatterGenericModel for compatibility
        if isinstance(result, BaseModel) and not isinstance(
            result, FormatterGenericModel
        ):
            # Convert typed model to dict then to FormatterGenericModel
            result_dict = result.model_dump()
            return FormatterGenericModel(**result_dict)
        elif isinstance(result, dict):
            return FormatterGenericModel(**result)
        else:
            # Return as-is if it's already FormatterGenericModel or handle unexpected types
            return (
                result
                if isinstance(result, FormatterGenericModel)
                else FormatterGenericModel(data=result)
            )

    async def adapt_response(self, response: FormatterGenericModel) -> BaseModel:
        """Convert response using registry formatter."""
        if "response" not in self.formatters:
            raise ValueError(
                f"No response formatter available for {self.source_format} -> {self.target_format}"
            )

        # Extract dict from the generic model for formatter
        response_dict = response.model_dump()

        # Detect actual data format and choose the right formatter direction
        detected_format = self._detect_data_format(response_dict)

        # Debug logging
        import logging

        logger = logging.getLogger(__name__)
        logger.info(
            f"FormatterAdapter: source_format={self.source_format}, target_format={self.target_format}, detected_format={detected_format}"
        )

        # If the detected format doesn't match our source format, we need to use the reverse formatter
        if detected_format != self.source_format and detected_format:
            # Try to get the reverse formatter
            reverse_formatter_key = f"{detected_format}_to_{self.target_format}"
            try:
                from ccproxy.llms.formatters.formatter_registry import (
                    iter_registered_formatters,
                    load_builtin_formatter_modules,
                )

                # Load formatters and find the reverse one
                load_builtin_formatter_modules()
                all_formatters = dict(iter_registered_formatters())

                reverse_formatter = all_formatters.get(
                    (detected_format, self.target_format, "response")
                )
                if reverse_formatter:
                    # Use the reverse formatter with correct input type
                    typed_response = self._dict_to_input_model(
                        response_dict, detected_format, "response"
                    )

                    import inspect

                    if inspect.iscoroutinefunction(reverse_formatter):
                        result = await reverse_formatter(typed_response)
                    else:
                        result = reverse_formatter(typed_response)

                    # Convert result to FormatterGenericModel for compatibility
                    if isinstance(result, BaseModel) and not isinstance(
                        result, FormatterGenericModel
                    ):
                        result_dict = result.model_dump()
                        return FormatterGenericModel(**result_dict)
                    elif isinstance(result, dict):
                        return FormatterGenericModel(**result)
                    else:
                        return (
                            result
                            if isinstance(result, FormatterGenericModel)
                            else FormatterGenericModel(data=result)
                        )
            except Exception:
                # Fall back to original approach if reverse formatter lookup fails
                pass

        # Original approach: Call the formatter function - convert dict to appropriate typed model
        converter = self.formatters["response"]

        # Convert dict to appropriate input model for the formatter
        typed_response = self._dict_to_input_model(
            response_dict, self.source_format, "response"
        )

        import inspect

        if inspect.iscoroutinefunction(converter):
            result = await converter(typed_response)
        else:
            result = converter(typed_response)

        # Convert result to FormatterGenericModel for compatibility
        if isinstance(result, BaseModel) and not isinstance(
            result, FormatterGenericModel
        ):
            # Convert typed model to dict then to FormatterGenericModel
            result_dict = result.model_dump()
            return FormatterGenericModel(**result_dict)
        elif isinstance(result, dict):
            return FormatterGenericModel(**result)
        else:
            # Return as-is if it's already FormatterGenericModel or handle unexpected types
            return (
                result
                if isinstance(result, FormatterGenericModel)
                else FormatterGenericModel(data=result)
            )

    def adapt_stream(
        self, stream: AsyncIterator[FormatterGenericModel]
    ) -> AsyncGenerator[BaseModel, None]:
        """Convert stream using registry formatter."""
        if "stream" not in self.formatters:
            raise ValueError(
                f"No stream formatter available for {self.source_format} -> {self.target_format}"
            )

        # Convert FormatterGenericModel stream to dict stream for formatter
        async def typed_stream() -> AsyncIterator[Any]:
            async for chunk in stream:
                data = chunk.model_dump()
                yield self._to_stream_model(data, self.source_format)

        converter = self.formatters["stream"]

        async def converted_stream() -> AsyncGenerator[BaseModel, None]:
            converted = converter(typed_stream())
            async for result in converted:
                if isinstance(result, BaseModel):
                    yield result
                elif isinstance(result, dict):
                    yield FormatterGenericModel(**result)
                else:
                    yield FormatterGenericModel(data=result)

        return converted_stream()

    async def adapt_error(self, error: BaseModel) -> BaseModel:
        """Convert error using registry formatter if available."""
        if "error" not in self.formatters:
            # If no error formatter, return error as-is
            return error

        # Extract dict from the model for formatter (error can be any BaseModel type)
        if hasattr(error, "model_dump"):
            error_dict = error.model_dump()
        else:
            # Fallback for other BaseModel types
            error_dict = dict(error)

        converter = self.formatters["error"]
        if hasattr(converter, "__await__"):
            result = await converter(error_dict)
        else:
            result = converter(error_dict)

        # Return result as FormatterGenericModel if it's a dict, otherwise pass through
        if isinstance(result, dict):
            return FormatterGenericModel(**result)
        return result  # type: ignore[no-any-return]

    async def convert_request(self, data: dict[str, Any]) -> dict[str, Any]:
        model = FormatterGenericModel(**data)
        converted = await self.adapt_request(model)
        return self._to_dict(converted)

    async def convert_response(self, data: dict[str, Any]) -> dict[str, Any]:
        model = FormatterGenericModel(**data)
        converted = await self.adapt_response(model)
        return self._to_dict(converted)

    async def convert_error(self, data: dict[str, Any]) -> dict[str, Any]:
        model = FormatterGenericModel(**data)
        converted = await self.adapt_error(model)
        return self._to_dict(converted)

    async def convert_stream(
        self, stream: AsyncIterator[dict[str, Any]]
    ) -> AsyncIterator[dict[str, Any]]:
        async def model_stream() -> AsyncIterator[FormatterGenericModel]:
            async for chunk in stream:
                yield FormatterGenericModel(**chunk)

        converted_stream = self.adapt_stream(model_stream())

        async def dict_stream() -> AsyncGenerator[dict[str, Any], None]:
            async for item in converted_stream:
                yield self._to_dict(item)

        return dict_stream()

    # Additional convenience methods (not abstract)
    def convert_usage(self, usage: Any) -> Any:
        """Convert usage using registry formatter if available."""
        if "usage" not in self.formatters:
            # If no usage formatter, return usage as-is
            return usage

        converter = self.formatters["usage"]
        return converter(usage)

    @property
    def supported_operations(self) -> list[str]:
        """Get list of supported operations for this format pair."""
        return list(self.formatters.keys())

    def _detect_data_format(self, data: dict[str, Any]) -> str:
        """Detect the actual format of the data based on its structure."""
        # Detect Anthropic MessageResponse format
        if "content" in data and "type" in data and data.get("type") == "message":
            return "anthropic.messages"

        # Detect OpenAI ChatCompletion format
        if (
            "choices" in data
            and "object" in data
            and data.get("object") == "chat.completion"
        ):
            return "openai.chat_completions"

        # Detect Response API format
        if (
            "output" in data
            and "status" in data
            and "object" in data
            and data.get("object") == "response"
        ):
            return "openai.responses"

        # Default fallback - return empty to indicate unknown
        return ""

    def _dict_to_input_model(
        self, data: dict[str, Any], format_name: str, operation: str
    ) -> BaseModel:
        """Convert dict to appropriate input model based on format and operation."""
        # Map format names to their corresponding model types for requests
        if operation == "request":
            if format_name == "openai.chat_completions":
                from ccproxy.llms.models import openai as openai_models

                return openai_models.ChatCompletionRequest.model_validate(data)
            elif format_name == "openai.responses":
                from ccproxy.llms.models.openai import ResponseRequest

                return ResponseRequest.model_validate(data)
            elif format_name == "anthropic.messages":
                from ccproxy.llms.models import anthropic as anthropic_models

                return anthropic_models.CreateMessageRequest.model_validate(data)
        elif operation == "response":
            if format_name == "openai.chat_completions":
                from ccproxy.llms.models import openai as openai_models

                return openai_models.ChatCompletionResponse.model_validate(data)
            elif format_name == "openai.responses":
                # For Response API format, check the actual data structure to determine the correct model
                # If data has "choices" field, it's OpenAI ChatCompletion format
                if "choices" in data:
                    from ccproxy.llms.models import openai as openai_models

                    return openai_models.ChatCompletionResponse.model_validate(data)
                # If data has "content" and "type" fields, it's Anthropic MessageResponse format
                elif (
                    "content" in data
                    and "type" in data
                    and data.get("type") == "message"
                ):
                    from ccproxy.llms.models import anthropic as anthropic_models

                    return anthropic_models.MessageResponse.model_validate(data)
                else:
                    # Otherwise, it's true Response API format
                    from ccproxy.llms.models.openai import ResponseObject

                    return ResponseObject.model_validate(data)
            elif format_name == "anthropic.messages":
                from ccproxy.llms.models import anthropic as anthropic_models

                return anthropic_models.MessageResponse.model_validate(data)

        # Fallback to FormatterGenericModel for unknown types
        return FormatterGenericModel(**data)

    def _to_stream_model(self, data: dict[str, Any], format_name: str) -> BaseModel:
        try:
            if format_name == "anthropic.messages":
                from ccproxy.llms.models import anthropic as anthropic_models

                adapter = TypeAdapter(anthropic_models.MessageStreamEvent)
                return adapter.validate_python(data)
            if format_name == "openai.responses":
                from ccproxy.llms.models import openai as openai_models

                adapter = TypeAdapter(openai_models.AnyStreamEvent)
                result = adapter.validate_python(data)
                return getattr(result, "root", result)
            if format_name == "openai.chat_completions":
                from ccproxy.llms.models import openai as openai_models

                return openai_models.ChatCompletionChunk.model_validate(data)
        except Exception as exc:  # pragma: no cover - best effort logging
            try:
                from structlog import get_logger

                get_logger(__name__).debug(
                    "stream_chunk_type_adapter_failed",
                    source_format=format_name,
                    error=str(exc),
                    keys=list(data.keys()),
                )
            except Exception:
                pass

        return FormatterGenericModel(**data)

    def _to_dict(self, model: BaseModel | dict[str, Any]) -> dict[str, Any]:
        if isinstance(model, FormatterGenericModel):
            return model.model_dump()
        if isinstance(model, BaseModel):
            return model.model_dump(mode="json", exclude_unset=True)
        if isinstance(model, dict):
            return model
        raise TypeError(
            f"FormatterRegistryAdapter produced unsupported result type {type(model).__name__}"
        )

    def __repr__(self) -> str:
        return f"FormatterRegistryAdapter({self.source_format} -> {self.target_format})"


def create_formatter_adapter_factory(
    source_format: str, target_format: str
) -> Callable[[], FormatterRegistryAdapter]:
    """Create an adapter factory function for use in FormatAdapterSpec.

    This function returns a factory that can be used in plugin FormatAdapterSpec
    declarations to create FormatterRegistryAdapter instances.

    Args:
        source_format: Source format identifier
        target_format: Target format identifier

    Returns:
        Factory function that creates FormatterRegistryAdapter instances

    Example:
        format_adapters = [
            FormatAdapterSpec(
                from_format="anthropic.messages",
                to_format="openai.chat_completions",
                adapter_factory=create_formatter_adapter_factory(
                    "anthropic.messages", "openai.chat_completions"
                ),
                description="Anthropic Messages to OpenAI Chat via FormatterRegistry",
            ),
        ]
    """

    def factory() -> FormatterRegistryAdapter:
        # Import here to avoid circular imports during module loading
        from ccproxy.llms.formatters.formatter_registry import (
            iter_registered_formatters,
            load_builtin_formatter_modules,
        )

        # Load formatter modules and create registry
        load_builtin_formatter_modules()
        formatter_registry = FormatterRegistry()
        formatter_registry.register_many(iter_registered_formatters())

        return FormatterRegistryAdapter(
            formatter_registry=formatter_registry,
            source_format=source_format,
            target_format=target_format,
        )

    return factory


__all__ = [
    "FormatterRegistryAdapter",
    "create_formatter_adapter_factory",
]
