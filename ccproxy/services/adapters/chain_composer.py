from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator
from typing import Any, Literal

from .format_adapter import DictFormatAdapter, FormatAdapterProtocol
from .format_registry import FormatRegistry


class ComposedAdapter(DictFormatAdapter):
    """A DictFormatAdapter composed from multiple pairwise adapters."""

    pass


def _pairs_from_chain(
    chain: list[str], stage: Literal["request", "response", "error", "stream"]
) -> list[tuple[str, str]]:
    if len(chain) < 2:
        return []
    # For responses and streaming, convert from provider format (tail) back to client format (head)
    if stage in ("response", "error", "stream"):
        pairs = [(chain[i + 1], chain[i]) for i in range(len(chain) - 1)]
        pairs.reverse()
        return pairs
    # Requests go forward (client -> provider)
    return [(chain[i], chain[i + 1]) for i in range(len(chain) - 1)]


def compose_from_chain(
    *,
    registry: FormatRegistry,
    chain: list[str],
    name: str | None = None,
) -> FormatAdapterProtocol:
    """Compose a FormatAdapter from a format_chain using the registry.

    The composed adapter sequentially applies the per‑pair adapters for request,
    response, error, and stream stages.
    """

    async def _compose_stage(
        data: dict[str, Any], stage: Literal["request", "response", "error"]
    ) -> dict[str, Any]:
        current = data
        for src, dst in _pairs_from_chain(chain, stage):
            adapter = registry.get(src, dst)
            if stage == "request":
                current = await adapter.convert_request(current)
            elif stage == "response":
                current = await adapter.convert_response(current)
            else:
                # Default error passthrough if adapter lacks explicit error handling
                with contextlib.suppress(NotImplementedError):
                    current = await adapter.convert_error(current)
        return current

    async def _request(data: dict[str, Any]) -> dict[str, Any]:
        return await _compose_stage(data, "request")

    async def _response(data: dict[str, Any]) -> dict[str, Any]:
        return await _compose_stage(data, "response")

    async def _error(data: dict[str, Any]) -> dict[str, Any]:
        return await _compose_stage(data, "error")

    async def _stream(
        stream: AsyncIterator[dict[str, Any]],
    ) -> AsyncIterator[dict[str, Any]]:
        # Pipe the stream through each pairwise adapter's convert_stream
        current_stream = stream
        for src, dst in _pairs_from_chain(chain, "stream"):
            adapter = registry.get(src, dst)
            current_stream = adapter.convert_stream(current_stream)
        async for item in current_stream:
            yield item

    return ComposedAdapter(
        request=_request,
        response=_response,
        error=_error,
        stream=_stream,
        name=name or f"ComposedAdapter({' -> '.join(chain)})",
    )


__all__ = ["compose_from_chain", "ComposedAdapter"]
