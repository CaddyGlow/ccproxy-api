from __future__ import annotations

from collections.abc import AsyncIterator
from unittest.mock import MagicMock

from fastapi import FastAPI

from ccproxy.api.format_validation import validate_route_format_chains
from ccproxy.services.adapters.format_adapter import DictFormatAdapter, FormatDict
from ccproxy.services.adapters.format_registry import FormatRegistry


def _make_app_with_chain(chain: list[str]) -> FastAPI:
    app = FastAPI()

    @app.get("/test")
    async def handler() -> dict[str, str]:
        return {"ok": "true"}

    handler.__format_chain__ = chain  # type: ignore[attr-defined]
    return app


async def _identity_stream(
    stream: AsyncIterator[FormatDict],
) -> AsyncIterator[FormatDict]:
    async for item in stream:
        yield item


def _register_bidirectional_adapter(
    registry: FormatRegistry, source: str, target: str
) -> None:
    adapter = DictFormatAdapter(
        request=lambda data: data,
        response=lambda data: data,
        error=lambda data: data,
        stream=_identity_stream,
        name="identity",
    )
    registry.register(
        from_format=source,
        to_format=target,
        adapter=adapter,
        plugin_name="test",
    )


def test_validate_route_format_chains_reports_missing() -> None:
    app = _make_app_with_chain(["one", "two"])
    registry = FormatRegistry()
    logger = MagicMock()

    validate_route_format_chains(app=app, registry=registry, logger=logger)

    logger.error.assert_called_once()
    error_kwargs = logger.error.call_args.kwargs
    assert error_kwargs["missing_adapters"] == ["Missing format adapter: one -> two"]
    assert error_kwargs["missing_stream_adapters"] == [
        "Missing streaming adapter: two -> one"
    ]


def test_validate_route_format_chains_passes_when_adapters_registered() -> None:
    app = _make_app_with_chain(["one", "two"])
    registry = FormatRegistry()
    logger = MagicMock()

    _register_bidirectional_adapter(registry, "one", "two")
    _register_bidirectional_adapter(registry, "two", "one")

    validate_route_format_chains(app=app, registry=registry, logger=logger)

    logger.error.assert_not_called()
    logger.warning.assert_not_called()


def test_validate_route_format_chains_logs_warning_on_exception() -> None:
    app = _make_app_with_chain(["one", "two"])
    registry = FormatRegistry()
    logger = MagicMock()

    # Force registry lookup to raise to exercise defensive warning path
    registry.get_if_exists = MagicMock(side_effect=RuntimeError("boom"))  # type: ignore[method-assign]

    validate_route_format_chains(app=app, registry=registry, logger=logger)

    logger.warning.assert_called_once()
    warning_kwargs = logger.warning.call_args.kwargs
    assert warning_kwargs["error"] == "boom"
