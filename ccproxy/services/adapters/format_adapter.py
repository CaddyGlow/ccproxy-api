"""Format adapter interfaces and helpers for dict-based conversions."""

from __future__ import annotations

import inspect
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any, Protocol, runtime_checkable


FormatDict = dict[str, Any]


async def _maybe_await(value: Any) -> Any:
    """Await coroutine-like values produced by adapter callables."""

    if inspect.isawaitable(value):
        return await value
    return value


@runtime_checkable
class FormatAdapterProtocol(Protocol):
    """Protocol for format adapters operating on plain dictionaries."""

    async def convert_request(self, data: FormatDict) -> FormatDict:
        """Convert an outgoing request payload."""

    async def convert_response(self, data: FormatDict) -> FormatDict:
        """Convert a non-streaming response payload."""

    async def convert_error(self, data: FormatDict) -> FormatDict:
        """Convert an error payload."""

    def convert_stream(
        self, stream: AsyncIterator[FormatDict]
    ) -> AsyncIterator[FormatDict]:
        """Convert a streaming response represented as an async iterator."""


class DictFormatAdapter(FormatAdapterProtocol):
    """Adapter built from per-stage callables with strict dict IO."""

    def __init__(
        self,
        *,
        request: Callable[[FormatDict], Awaitable[FormatDict]]
        | Callable[[FormatDict], FormatDict]
        | None = None,
        response: Callable[[FormatDict], Awaitable[FormatDict]]
        | Callable[[FormatDict], FormatDict]
        | None = None,
        error: Callable[[FormatDict], Awaitable[FormatDict]]
        | Callable[[FormatDict], FormatDict]
        | None = None,
        stream: Callable[[AsyncIterator[FormatDict]], AsyncIterator[FormatDict]]
        | Callable[[AsyncIterator[FormatDict]], Awaitable[AsyncIterator[FormatDict]]]
        | Callable[[AsyncIterator[FormatDict]], Awaitable[Any]]
        | None = None,
        name: str | None = None,
    ) -> None:
        self._request = request
        self._response = response
        self._error = error
        self._stream = stream
        self.name = name or self.__class__.__name__

    async def convert_request(self, data: FormatDict) -> FormatDict:
        return await self._run_stage(self._request, data, stage="request")

    async def convert_response(self, data: FormatDict) -> FormatDict:
        return await self._run_stage(self._response, data, stage="response")

    async def convert_error(self, data: FormatDict) -> FormatDict:
        return await self._run_stage(self._error, data, stage="error")

    def convert_stream(
        self, stream: AsyncIterator[FormatDict]
    ) -> AsyncIterator[FormatDict]:
        if self._stream is None:
            raise NotImplementedError(
                f"{self.name} does not implement stream conversion"
            )

        return self._create_stream_iterator(stream)

    async def _create_stream_iterator(
        self, stream: AsyncIterator[FormatDict]
    ) -> AsyncIterator[FormatDict]:
        """Helper method to create the actual async iterator."""
        if self._stream is None:
            raise NotImplementedError(
                f"{self.name} does not implement stream conversion"
            )

        handler = self._stream(stream)
        handler = await _maybe_await(handler)

        if not hasattr(handler, "__aiter__"):
            raise TypeError(
                f"{self.name}.stream must return an async iterator, got {type(handler).__name__}"
            )

        async for item in handler:
            if not isinstance(item, dict):
                raise TypeError(
                    f"{self.name}.stream yielded non-dict item: {type(item).__name__}"
                )
            yield item

    async def _run_stage(
        self,
        func: Callable[[FormatDict], Awaitable[FormatDict]]
        | Callable[[FormatDict], FormatDict]
        | None,
        data: FormatDict,
        *,
        stage: str,
    ) -> FormatDict:
        if func is None:
            raise NotImplementedError(
                f"{self.name} does not implement {stage} conversion"
            )

        result = await _maybe_await(func(data))
        if not isinstance(result, dict):
            raise TypeError(
                f"{self.name}.{stage} must return dict, got {type(result).__name__}"
            )
        return result


__all__ = [
    "FormatAdapterProtocol",
    "FormatDict",
    "DictFormatAdapter",
]
