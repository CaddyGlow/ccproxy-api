"""Direct dict-based conversion functions for use with DictFormatAdapter.

This module provides simple wrapper functions around the existing formatter functions
that operate directly on dictionaries instead of typed Pydantic models.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from .protocols import StreamAccumulatorProtocol

from pydantic import TypeAdapter, ValidationError

from ccproxy.core import logging
from ccproxy.core.constants import (
    FORMAT_ANTHROPIC_MESSAGES as ANTHROPIC_MESSAGES,
)
from ccproxy.core.constants import (
    FORMAT_OPENAI_CHAT as OPENAI_CHAT,
)
from ccproxy.core.constants import (
    FORMAT_OPENAI_RESPONSES as OPENAI_RESPONSES,
)
from ccproxy.llms.formatters import (
    anthropic_to_openai,
    openai_to_anthropic,
    openai_to_openai,
)
from ccproxy.llms.formatters import anthropic_to_openai as a2o
from ccproxy.llms.models import anthropic as anthropic_models
from ccproxy.llms.models import openai as openai_models
from ccproxy.llms.models.anthropic import MessageStreamEvent

from .format_adapter import DictFormatAdapter
from .format_registry import FormatRegistry


FormatDict = dict[str, Any]

logger = logging.get_logger(__name__)


_type_adapter_cache: dict[Any, TypeAdapter[Any]] = {}


def _validate_stream_event(model: Any, data: dict[str, Any]) -> Any:
    """Validate a streaming event against the provided model.

    Raises ValidationError when the payload does not conform so callers can fail fast.
    """

    adapter = _type_adapter_cache.get(model)
    if adapter is None:
        adapter = TypeAdapter(model)
        _type_adapter_cache[model] = adapter
    return adapter.validate_python(data)


# Generic stream mapper to DRY conversion loops
async def map_stream(
    stream: AsyncIterator[FormatDict],
    *,
    validator_model: Any,
    converter: Any,
    accumulator: StreamAccumulatorProtocol | None = None,
) -> AsyncIterator[FormatDict]:
    """Map stream with optional accumulation before validation.

    Args:
        stream: Input stream of format dictionaries
        validator_model: Pydantic model for validation
        converter: Converter function to apply to validated stream
        accumulator: Optional accumulator for handling partial chunks

    Returns:
        Converted stream
    """

    async def _typed_stream() -> AsyncIterator[Any]:
        async for chunk_data in stream:
            if accumulator:
                # Accumulate chunk and get complete object if ready
                complete_object = accumulator.accumulate_chunk(chunk_data)

                if complete_object is None:
                    # Still accumulating, skip this chunk
                    continue

                # Have complete object, validate it
                try:
                    yield _validate_stream_event(validator_model, complete_object)
                except ValidationError as exc:
                    logger.debug(
                        "stream_chunk_validation_failed",
                        model=str(validator_model),
                        error=str(exc),
                        action="raise",
                    )
                    raise
            else:
                # No accumulator, validate directly
                try:
                    yield _validate_stream_event(validator_model, chunk_data)
                except ValidationError as exc:
                    logger.debug(
                        "stream_chunk_validation_failed",
                        model=str(validator_model),
                        error=str(exc),
                        action="raise",
                    )
                    raise

    converted_chunks = converter(_typed_stream())
    async for converted_chunk in converted_chunks:
        if hasattr(converted_chunk, "model_dump"):
            yield converted_chunk.model_dump(exclude_unset=True)
        else:
            yield converted_chunk


# OpenAI to Anthropic converters (for plugins that target Anthropic APIs)
async def convert_openai_to_anthropic_request(data: FormatDict) -> FormatDict:
    """Convert OpenAI ChatCompletion request to Anthropic CreateMessage request."""
    # Convert dict to typed model
    request = openai_models.ChatCompletionRequest.model_validate(data)

    # Use existing formatter function
    result = (
        await openai_to_anthropic.convert__openai_chat_to_anthropic_message__request(
            request
        )
    )

    # Convert back to dict
    result_dict: FormatDict = result.model_dump(exclude_unset=True)
    return result_dict


async def convert_anthropic_to_openai_response(data: FormatDict) -> FormatDict:
    """Convert Anthropic MessageResponse to OpenAI ChatCompletion response."""
    # Convert dict to typed model
    response = anthropic_models.MessageResponse.model_validate(data)

    # Use existing formatter function
    result = anthropic_to_openai.convert__anthropic_message_to_openai_chat__response(
        response
    )

    # Convert back to dict
    result_dict: FormatDict = result.model_dump(exclude_unset=True)
    return result_dict


async def convert_anthropic_to_openai_stream(
    stream: AsyncIterator[FormatDict],
) -> AsyncIterator[FormatDict]:
    """Convert Anthropic MessageStream to OpenAI ChatCompletion stream."""

    async for out_chunk in map_stream(
        stream,
        validator_model=anthropic_models.MessageStreamEvent,
        converter=anthropic_to_openai.convert__anthropic_message_to_openai_chat__stream,
    ):
        yield out_chunk


async def convert_openai_to_anthropic_error(data: FormatDict) -> FormatDict:
    """Convert OpenAI error to Anthropic error."""
    # Convert dict to typed model
    error = openai_models.ErrorResponse.model_validate(data)

    # Use existing formatter function
    result = openai_to_anthropic.convert__openai_to_anthropic__error(error)

    # Convert back to dict
    result_dict: FormatDict = result.model_dump(exclude_unset=True)
    return result_dict


# Anthropic to OpenAI converters (reverse direction, if needed)
async def convert_anthropic_to_openai_request(data: FormatDict) -> FormatDict:
    """Convert Anthropic CreateMessage request to OpenAI ChatCompletion request."""
    # Convert dict to typed model
    request = anthropic_models.CreateMessageRequest.model_validate(data)

    # Use existing formatter function
    result = anthropic_to_openai.convert__anthropic_message_to_openai_chat__request(
        request
    )

    # Convert back to dict
    result_dict: FormatDict = result.model_dump(exclude_unset=True)
    return result_dict


async def convert_openai_to_anthropic_response(data: FormatDict) -> FormatDict:
    """Convert OpenAI ChatCompletion response to Anthropic MessageResponse."""
    # Convert dict to typed model
    response = openai_models.ChatCompletionResponse.model_validate(data)

    # Use existing formatter function
    result = openai_to_anthropic.convert__openai_chat_to_anthropic_messages__response(
        response
    )

    # Convert back to dict
    result_dict: FormatDict = result.model_dump(exclude_unset=True)
    return result_dict


async def convert_openai_to_anthropic_stream(
    stream: AsyncIterator[FormatDict],
) -> AsyncIterator[FormatDict]:
    """Convert OpenAI ChatCompletion stream to Anthropic MessageStream."""
    from .chat_accumulator import ChatCompletionAccumulator

    # Use accumulator to handle partial tool calls
    accumulator = ChatCompletionAccumulator()

    async for out_chunk in map_stream(
        stream,
        validator_model=openai_models.ChatCompletionChunk,
        converter=openai_to_anthropic.convert__openai_chat_to_anthropic_messages__stream,
        accumulator=accumulator,
    ):
        yield out_chunk


async def convert_anthropic_to_openai_error(data: FormatDict) -> FormatDict:
    """Convert Anthropic error to OpenAI error."""
    # Convert dict to typed model
    error = anthropic_models.ErrorResponse.model_validate(data)

    # Use existing formatter function
    result = anthropic_to_openai.convert__anthropic_to_openai__error(error)

    # Convert back to dict
    result_dict: FormatDict = result.model_dump(exclude_unset=True)
    return result_dict


# OpenAI Responses format converters (for Codex plugin)
async def convert_openai_responses_to_anthropic_request(data: FormatDict) -> FormatDict:
    """Convert OpenAI Responses request to Anthropic CreateMessage request."""
    # Convert dict to typed model
    request = openai_models.ResponseRequest.model_validate(data)

    # Use existing formatter function
    result = (
        openai_to_anthropic.convert__openai_responses_to_anthropic_message__request(
            request
        )
    )

    # Convert back to dict
    result_dict: FormatDict = result.model_dump(exclude_unset=True)
    return result_dict


async def convert_openai_responses_to_anthropic_response(
    data: FormatDict,
) -> FormatDict:
    """Convert OpenAI Responses response to Anthropic MessageResponse."""
    # Convert dict to typed model
    response = openai_models.ResponseObject.model_validate(data)

    # Use existing formatter function
    result = (
        openai_to_anthropic.convert__openai_responses_to_anthropic_message__response(
            response
        )
    )

    # Convert back to dict
    result_dict: FormatDict = result.model_dump(exclude_unset=True)
    return result_dict


async def convert_anthropic_to_openai_responses_request(data: FormatDict) -> FormatDict:
    """Convert Anthropic CreateMessage request to OpenAI Responses request."""
    # Convert dict to typed model
    request = anthropic_models.CreateMessageRequest.model_validate(data)

    # Use existing formatter function
    result = (
        anthropic_to_openai.convert__anthropic_message_to_openai_responses__request(
            request
        )
    )

    # Convert back to dict
    result_dict: FormatDict = result.model_dump(exclude_unset=True)
    return result_dict


async def convert_anthropic_to_openai_responses_response(
    data: FormatDict,
) -> FormatDict:
    """Convert Anthropic MessageResponse to OpenAI Responses response."""
    # Convert dict to typed model
    response = anthropic_models.MessageResponse.model_validate(data)

    # Use existing formatter function
    result = (
        anthropic_to_openai.convert__anthropic_message_to_openai_responses__response(
            response
        )
    )

    # Convert back to dict
    result_dict: FormatDict = result.model_dump(exclude_unset=True)
    return result_dict


# OpenAI Chat ↔ OpenAI Responses converters (for Codex plugin)
async def convert_openai_chat_to_openai_responses_request(
    data: FormatDict,
) -> FormatDict:
    """Convert OpenAI ChatCompletion request to OpenAI Responses request."""
    # Convert dict to typed model
    request = openai_models.ChatCompletionRequest.model_validate(data)

    # Use existing formatter function
    result = await openai_to_openai.convert__openai_chat_to_openai_responses__request(
        request
    )

    # Convert back to dict
    result_dict: FormatDict = result.model_dump(exclude_unset=True)
    return result_dict


async def convert_openai_responses_to_openai_chat_response(
    data: FormatDict,
) -> FormatDict:
    """Convert OpenAI Responses response to OpenAI ChatCompletion response."""
    if isinstance(data, dict):
        if data.get("object") == "chat.completion" or (
            "choices" in data and "response" not in data and "model" in data
        ):
            return data

    # Convert dict to typed model
    response = openai_models.ResponseObject.model_validate(data)

    # Use existing formatter function
    result = openai_to_openai.convert__openai_responses_to_openai_chat__response(
        response
    )

    # Convert back to dict
    result_dict: FormatDict = result.model_dump(exclude_unset=True)
    return result_dict


async def convert_openai_chat_to_openai_responses_response(
    data: FormatDict,
) -> FormatDict:
    """Convert OpenAI ChatCompletion response to OpenAI Responses response."""
    # Convert dict to typed model
    response = openai_models.ChatCompletionResponse.model_validate(data)

    # Use existing formatter function
    result = await openai_to_openai.convert__openai_chat_to_openai_responses__response(
        response
    )

    # Convert back to dict
    result_dict: FormatDict = result.model_dump(exclude_unset=True)
    return result_dict


async def convert_openai_responses_to_openai_chat_stream(
    stream: AsyncIterator[FormatDict],
) -> AsyncIterator[FormatDict]:
    """Convert OpenAI Responses stream to OpenAI ChatCompletion stream."""
    from ccproxy.llms.models.openai import AnyStreamEvent

    async for out_chunk in map_stream(
        stream,
        validator_model=AnyStreamEvent,
        converter=openai_to_openai.convert__openai_responses_to_openai_chat__stream,
    ):
        yield out_chunk


async def convert_openai_chat_to_openai_responses_stream(
    stream: AsyncIterator[FormatDict],
) -> AsyncIterator[FormatDict]:
    """Convert OpenAI ChatCompletion stream to OpenAI Responses stream."""
    from .chat_accumulator import ChatCompletionAccumulator

    # Use accumulator to handle partial tool calls
    accumulator = ChatCompletionAccumulator()

    async for out_chunk in map_stream(
        stream,
        validator_model=openai_models.ChatCompletionChunk,
        converter=openai_to_openai.convert__openai_chat_to_openai_responses__stream,
        accumulator=accumulator,
    ):
        yield out_chunk


async def convert_anthropic_to_openai_responses_stream(
    stream: AsyncIterator[FormatDict],
) -> AsyncIterator[FormatDict]:
    """Convert Anthropic MessageStream to OpenAI Responses stream.

    Avoid dict→model→dict churn by using the shared map_stream helper.
    """

    async for out_chunk in map_stream(
        stream,
        validator_model=MessageStreamEvent,
        converter=a2o.convert__anthropic_message_to_openai_responses__stream,
    ):
        yield out_chunk


async def convert_openai_responses_to_anthropic_stream(
    stream: AsyncIterator[FormatDict],
) -> AsyncIterator[FormatDict]:
    """Convert OpenAI Responses stream to Anthropic MessageStream."""
    # Since there's no direct openai.responses -> anthropic stream converter,
    # we'll convert responses -> chat -> anthropic
    chat_stream = convert_openai_responses_to_openai_chat_stream(stream)
    anthropic_stream = convert_openai_to_anthropic_stream(chat_stream)
    async for chunk in anthropic_stream:
        yield chunk


async def convert_openai_responses_to_openai_chat_request(
    data: FormatDict,
) -> FormatDict:
    """Convert OpenAI Responses request to OpenAI ChatCompletion request."""
    # Convert dict to typed model
    request = openai_models.ResponseRequest.model_validate(data)

    # Use existing formatter function
    result = await openai_to_openai.convert__openai_responses_to_openaichat__request(
        request
    )

    # Convert back to dict
    result_dict: FormatDict = result.model_dump(exclude_unset=True)
    return result_dict


# Passthrough and additional error conversion functions
# OpenAI↔OpenAI error formats are identical; return input unchanged.
async def convert_openai_responses_to_anthropic_error(data: FormatDict) -> FormatDict:
    """Convert OpenAI Responses error to Anthropic error."""
    # OpenAI errors are similar across formats - use existing converter
    return await convert_openai_to_anthropic_error(data)


async def convert_anthropic_to_openai_responses_error(data: FormatDict) -> FormatDict:
    """Convert Anthropic error to OpenAI Responses error."""
    # Use existing anthropic -> openai error converter (errors are same format)
    return await convert_anthropic_to_openai_error(data)


async def convert_openai_responses_to_openai_chat_error(data: FormatDict) -> FormatDict:
    """Convert OpenAI Responses error to OpenAI ChatCompletion error."""
    # Errors have the same format between OpenAI endpoints - passthrough
    return data


async def convert_openai_chat_to_openai_responses_error(data: FormatDict) -> FormatDict:
    """Convert OpenAI ChatCompletion error to OpenAI Responses error."""
    # Errors have the same format between OpenAI endpoints - passthrough
    return data


__all__ = [
    "convert_openai_to_anthropic_request",
    "convert_anthropic_to_openai_response",
    "convert_anthropic_to_openai_stream",
    "convert_openai_to_anthropic_error",
    "convert_anthropic_to_openai_request",
    "convert_openai_to_anthropic_response",
    "convert_openai_to_anthropic_stream",
    "convert_anthropic_to_openai_error",
    "convert_openai_responses_to_anthropic_request",
    "convert_openai_responses_to_anthropic_response",
    "convert_openai_responses_to_anthropic_error",
    "convert_anthropic_to_openai_responses_request",
    "convert_anthropic_to_openai_responses_response",
    "convert_anthropic_to_openai_responses_error",
    "convert_anthropic_to_openai_responses_stream",
    "convert_openai_responses_to_anthropic_stream",
    "convert_openai_chat_to_openai_responses_request",
    "convert_openai_responses_to_openai_chat_response",
    "convert_openai_responses_to_openai_chat_error",
    "convert_openai_chat_to_openai_responses_response",
    "convert_openai_chat_to_openai_responses_error",
    "convert_openai_chat_to_openai_responses_stream",
    "convert_openai_responses_to_openai_chat_stream",
    "convert_openai_responses_to_openai_chat_request",
]

# Centralized pair→stage mapping and registration helpers


def get_converter_map() -> dict[tuple[str, str], dict[str, Any]]:
    """Return a mapping of (from, to) → {request, response, error, stream} callables.

    Missing stages are allowed (e.g., error), and will default to passthrough in composition.
    """
    return {
        # OpenAI Chat → Anthropic Messages
        (OPENAI_CHAT, ANTHROPIC_MESSAGES): {
            "request": convert_openai_to_anthropic_request,
            "response": convert_anthropic_to_openai_response,
            "error": convert_anthropic_to_openai_error,
            "stream": convert_anthropic_to_openai_stream,
        },
        # Anthropic Messages → OpenAI Chat
        (ANTHROPIC_MESSAGES, OPENAI_CHAT): {
            "request": convert_anthropic_to_openai_request,
            "response": convert_openai_to_anthropic_response,
            "error": convert_openai_to_anthropic_error,
            "stream": convert_openai_to_anthropic_stream,
        },
        # OpenAI Chat ↔ OpenAI Responses
        (OPENAI_CHAT, OPENAI_RESPONSES): {
            "request": convert_openai_chat_to_openai_responses_request,
            "response": convert_openai_chat_to_openai_responses_response,
            "error": convert_openai_chat_to_openai_responses_error,
            "stream": convert_openai_chat_to_openai_responses_stream,
        },
        (OPENAI_RESPONSES, OPENAI_CHAT): {
            "request": convert_openai_responses_to_openai_chat_request,
            "response": convert_openai_responses_to_openai_chat_response,
            "error": convert_openai_responses_to_openai_chat_error,
            "stream": convert_openai_responses_to_openai_chat_stream,
        },
        # OpenAI Responses ↔ Anthropic Messages
        (OPENAI_RESPONSES, ANTHROPIC_MESSAGES): {
            "request": convert_openai_responses_to_anthropic_request,
            "response": convert_openai_responses_to_anthropic_response,
            "error": convert_openai_responses_to_anthropic_error,
            "stream": convert_openai_responses_to_anthropic_stream,
        },
        (ANTHROPIC_MESSAGES, OPENAI_RESPONSES): {
            "request": convert_anthropic_to_openai_responses_request,
            "response": convert_anthropic_to_openai_responses_response,
            "error": convert_anthropic_to_openai_responses_error,
            "stream": convert_anthropic_to_openai_responses_stream,
        },
    }


def register_converters(registry: FormatRegistry, *, plugin_name: str = "core") -> None:
    """Register DictFormatAdapter instances for all known pairs into the registry."""
    for (src, dst), stages in get_converter_map().items():
        adapter = DictFormatAdapter(
            request=stages.get("request"),
            response=stages.get("response"),
            error=stages.get("error"),
            stream=stages.get("stream"),
            name=f"{src}->{dst}",
        )
        registry.register(
            from_format=src, to_format=dst, adapter=adapter, plugin_name=plugin_name
        )
