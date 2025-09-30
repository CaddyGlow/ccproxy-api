"""Unit tests for :mod:`ccproxy.services.handler_config`."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from ccproxy.services.adapters.format_adapter import DictFormatAdapter
from ccproxy.services.adapters.format_context import FormatContext
from ccproxy.services.handler_config import HandlerConfig


class HeaderTransformer:
    """Simple transformer that records header transformations."""

    def __init__(self) -> None:
        self.calls: list[dict[str, str]] = []

    def transform_headers(self, headers: dict[str, str], **_: object) -> dict[str, str]:
        updated = headers | {"x-transformed": "true"}
        self.calls.append(updated)
        return updated

    def transform_body(self, body: object) -> object:
        return body


class DummySSEParser:
    """Non-network SSE parser used to satisfy the protocol."""

    def __call__(self, raw: str) -> dict[str, str] | None:  # pragma: no cover - trivial
        return {"raw": raw}

    def transform_body(self, body: object) -> object:  # pragma: no cover - trivial
        return body


def test_handler_config_defaults() -> None:
    """HandlerConfig exposes consistent defaults for optional components."""
    config = HandlerConfig()

    assert config.request_adapter is None
    assert config.response_adapter is None
    assert config.request_transformer is None
    assert config.response_transformer is None
    assert config.supports_streaming is True
    assert config.preserve_header_case is False
    assert config.sse_parser is None
    assert config.format_context is None


@pytest.mark.asyncio
async def test_handler_config_custom_components() -> None:
    """Custom adapters, transformers, and parsers are preserved on the config."""
    adapter = DictFormatAdapter(
        request=lambda data: data | {"stage": "request"},
        response=lambda data: data | {"stage": "response"},
        error=lambda data: data | {"stage": "error"},
    )
    transformer = HeaderTransformer()
    sse_parser = DummySSEParser()
    format_context = FormatContext(
        source_format="openai",
        target_format="anthropic",
        conversion_needed=True,
        streaming_mode="auto",
    )

    config = HandlerConfig(
        request_adapter=adapter,
        response_adapter=adapter,
        request_transformer=transformer,
        response_transformer=transformer,
        supports_streaming=False,
        preserve_header_case=True,
        sse_parser=sse_parser,
        format_context=format_context,
    )

    assert config.request_adapter is adapter
    assert config.response_adapter is adapter
    assert config.request_transformer is transformer
    assert config.response_transformer is transformer
    assert config.supports_streaming is False
    assert config.preserve_header_case is True
    assert config.sse_parser is sse_parser
    assert config.format_context == format_context

    converted = await config.request_adapter.convert_request({"value": 1})
    assert converted == {"value": 1, "stage": "request"}

    headers = config.request_transformer.transform_headers({"x-original": "true"})
    assert headers["x-transformed"] == "true"
    assert transformer.calls[-1] == headers

    assert config.sse_parser("final-event") == {"raw": "final-event"}


def test_handler_config_is_immutable() -> None:
    """The dataclass is frozen to prevent mutation after creation."""
    config = HandlerConfig()

    with pytest.raises(FrozenInstanceError):
        config.supports_streaming = False  # type: ignore[misc]

    with pytest.raises(FrozenInstanceError):
        config.request_adapter = DictFormatAdapter()  # type: ignore[assignment]
