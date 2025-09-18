"""Direct dict-based conversion functions for use with SimpleFormatAdapter.

This module provides simple wrapper functions around the existing formatter functions
that operate directly on dictionaries instead of typed Pydantic models. This eliminates
the need for the complex FormatterRegistryAdapter.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from ccproxy.llms.formatters.anthropic_to_openai import helpers as anthropic_to_openai
from ccproxy.llms.formatters.openai_to_anthropic import helpers as openai_to_anthropic
from ccproxy.llms.formatters.openai_to_openai import helpers as openai_to_openai
from ccproxy.llms.models import anthropic as anthropic_models
from ccproxy.llms.models import openai as openai_models


FormatDict = dict[str, Any]


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
    return result.model_dump(exclude_unset=True)


async def convert_anthropic_to_openai_response(data: FormatDict) -> FormatDict:
    """Convert Anthropic MessageResponse to OpenAI ChatCompletion response."""
    # Convert dict to typed model
    response = anthropic_models.MessageResponse.model_validate(data)

    # Use existing formatter function
    result = anthropic_to_openai.convert__anthropic_message_to_openai_chat__response(
        response
    )

    # Convert back to dict
    return result.model_dump(exclude_unset=True)


async def convert_anthropic_to_openai_stream(
    stream: AsyncIterator[FormatDict],
) -> AsyncIterator[FormatDict]:
    """Convert Anthropic MessageStream to OpenAI ChatCompletion stream."""
    async for chunk_data in stream:
        # Try to convert dict to typed model with graceful fallback
        try:
            from pydantic import TypeAdapter

            from ccproxy.llms.models.anthropic import MessageStreamEvent

            adapter = TypeAdapter(MessageStreamEvent)
            chunk = adapter.validate_python(chunk_data)
        except Exception:
            # For unknown event types, create a simple object with the data
            # The downstream formatter will handle unknown events gracefully
            from types import SimpleNamespace

            chunk = SimpleNamespace(**chunk_data)

        # Create async iterator from single chunk
        async def single_chunk_stream(chunk_value=chunk):
            yield chunk_value

        # Use existing formatter function
        converted_chunks = (
            anthropic_to_openai.convert__anthropic_message_to_openai_chat__stream(
                single_chunk_stream()
            )
        )

        # Yield converted chunks as dicts
        async for converted_chunk in converted_chunks:
            yield converted_chunk.model_dump(exclude_unset=True)


async def convert_openai_to_anthropic_error(data: FormatDict) -> FormatDict:
    """Convert OpenAI error to Anthropic error."""
    # Convert dict to typed model
    error = openai_models.ErrorResponse.model_validate(data)

    # Use existing formatter function
    result = openai_to_anthropic.convert__openai_to_anthropic__error(error)

    # Convert back to dict
    return result.model_dump(exclude_unset=True)


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
    return result.model_dump(exclude_unset=True)


async def convert_openai_to_anthropic_response(data: FormatDict) -> FormatDict:
    """Convert OpenAI ChatCompletion response to Anthropic MessageResponse."""
    # Convert dict to typed model
    response = openai_models.ChatCompletionResponse.model_validate(data)

    # Use existing formatter function
    result = openai_to_anthropic.convert__openai_chat_to_anthropic_messages__response(
        response
    )

    # Convert back to dict
    return result.model_dump(exclude_unset=True)


async def convert_openai_to_anthropic_stream(
    stream: AsyncIterator[FormatDict],
) -> AsyncIterator[FormatDict]:
    """Convert OpenAI ChatCompletion stream to Anthropic MessageStream."""
    async for chunk_data in stream:
        # Convert dict to typed model with graceful fallback
        try:
            chunk = openai_models.ChatCompletionChunk.model_validate(chunk_data)
        except Exception:
            # For unknown event types, create a simple object with the data
            # The downstream formatter will handle unknown events gracefully
            from types import SimpleNamespace

            chunk = SimpleNamespace(**chunk_data)

        # Create async iterator from single chunk
        async def single_chunk_stream(chunk_value=chunk):
            yield chunk_value

        # Use existing formatter function
        converted_chunks = (
            openai_to_anthropic.convert__openai_chat_to_anthropic_messages__stream(
                single_chunk_stream()
            )
        )

        # Yield converted chunks as dicts
        async for converted_chunk in converted_chunks:
            yield converted_chunk.model_dump(exclude_unset=True)


async def convert_anthropic_to_openai_error(data: FormatDict) -> FormatDict:
    """Convert Anthropic error to OpenAI error."""
    # Convert dict to typed model
    error = anthropic_models.ErrorResponse.model_validate(data)

    # Use existing formatter function
    result = anthropic_to_openai.convert__anthropic_to_openai__error(error)

    # Convert back to dict
    return result.model_dump(exclude_unset=True)


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
    return result.model_dump(exclude_unset=True)


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
    return result.model_dump(exclude_unset=True)


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
    return result.model_dump(exclude_unset=True)


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
    return result.model_dump(exclude_unset=True)


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
    return result.model_dump(exclude_unset=True)


async def convert_openai_responses_to_openai_chat_response(
    data: FormatDict,
) -> FormatDict:
    """Convert OpenAI Responses response to OpenAI ChatCompletion response."""
    # Convert dict to typed model
    response = openai_models.ResponseObject.model_validate(data)

    # Use existing formatter function
    result = openai_to_openai.convert__openai_responses_to_openai_chat__response(
        response
    )

    # Convert back to dict
    return result.model_dump(exclude_unset=True)


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
    return result.model_dump(exclude_unset=True)


async def convert_openai_responses_to_openai_chat_stream(
    stream: AsyncIterator[FormatDict],
) -> AsyncIterator[FormatDict]:
    """Convert OpenAI Responses stream to OpenAI ChatCompletion stream."""
    async for chunk_data in stream:
        # Convert dict to typed model with graceful fallback
        try:
            from pydantic import TypeAdapter

            from ccproxy.llms.models.openai import AnyStreamEvent

            adapter = TypeAdapter(AnyStreamEvent)
            chunk = adapter.validate_python(chunk_data)
        except Exception:
            # For unknown event types, create a simple object with the data
            # The downstream formatter will handle unknown events gracefully
            from types import SimpleNamespace

            chunk = SimpleNamespace(**chunk_data)

        # Create async iterator from single chunk
        async def single_chunk_stream(chunk_value=chunk):
            yield chunk_value

        # Use existing formatter function
        converted_chunks = (
            openai_to_openai.convert__openai_responses_to_openai_chat__stream(
                single_chunk_stream()
            )
        )

        # Yield converted chunks as dicts
        async for converted_chunk in converted_chunks:
            yield converted_chunk.model_dump(exclude_unset=True)


async def convert_openai_chat_to_openai_responses_stream(
    stream: AsyncIterator[FormatDict],
) -> AsyncIterator[FormatDict]:
    """Convert OpenAI ChatCompletion stream to OpenAI Responses stream."""
    async for chunk_data in stream:
        # Convert dict to typed model with graceful fallback
        try:
            chunk = openai_models.ChatCompletionChunk.model_validate(chunk_data)
        except Exception:
            # For unknown event types, create a simple object with the data
            # The downstream formatter will handle unknown events gracefully
            from types import SimpleNamespace

            chunk = SimpleNamespace(**chunk_data)

        # Create async iterator from single chunk
        async def single_chunk_stream(chunk_value=chunk):
            yield chunk_value

        # Use existing formatter function
        converted_chunks = (
            openai_to_openai.convert__openai_chat_to_openai_responses__stream(
                single_chunk_stream()
            )
        )

        # Yield converted chunks as dicts
        async for converted_chunk in converted_chunks:
            yield converted_chunk.model_dump(exclude_unset=True)


async def convert_anthropic_to_openai_responses_stream(
    stream: AsyncIterator[FormatDict],
) -> AsyncIterator[FormatDict]:
    """Convert Anthropic MessageStream to OpenAI Responses stream."""
    async for chunk_data in stream:
        # Convert dict to typed model with graceful fallback
        try:
            from pydantic import TypeAdapter

            from ccproxy.llms.models.anthropic import MessageStreamEvent

            adapter = TypeAdapter(MessageStreamEvent)
            chunk = adapter.validate_python(chunk_data)
        except Exception:
            # For unknown event types, create a simple object with the data
            # The downstream formatter will handle unknown events gracefully
            from types import SimpleNamespace

            chunk = SimpleNamespace(**chunk_data)

        # Create async iterator from single chunk
        async def single_chunk_stream(chunk_value=chunk):
            yield chunk_value

        # Use the proper anthropic -> openai.responses stream converter
        from ccproxy.llms.formatters.anthropic_to_openai import (
            helpers as anthropic_to_openai,
        )

        converted_chunks = (
            anthropic_to_openai.convert__anthropic_message_to_openai_responses__stream(
                single_chunk_stream()
            )
        )

        # Yield converted chunks as dicts
        async for converted_chunk in converted_chunks:
            yield converted_chunk.model_dump(exclude_unset=True)


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
    return result.model_dump(exclude_unset=True)


# Additional error conversion functions for completeness
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
