from __future__ import annotations

import inspect
from typing import TYPE_CHECKING

import structlog

from ccproxy.services.adapters.format_adapter import FormatAdapterProtocol


if TYPE_CHECKING:
    from ccproxy.core.plugins import (
        PluginManifest,
    )

logger = structlog.get_logger(__name__)


class FormatRegistry:
    """Registry mapping format pairs to concrete adapters."""

    def __init__(self) -> None:
        self._adapters: dict[tuple[str, str], FormatAdapterProtocol] = {}
        self._registered_plugins: dict[tuple[str, str], str] = {}

    def register(
        self,
        *,
        from_format: str,
        to_format: str,
        adapter: FormatAdapterProtocol,
        plugin_name: str = "unknown",
    ) -> None:
        key = (from_format, to_format)
        if key in self._adapters:
            existing = self._registered_plugins[key]
            logger.warning(
                "format_adapter_duplicate_ignored",
                from_format=from_format,
                to_format=to_format,
                existing_plugin=existing,
                attempted_plugin=plugin_name,
                category="format",
            )
            return

        self._adapters[key] = adapter
        self._registered_plugins[key] = plugin_name

        logger.info(
            "format_adapter_registered",
            from_format=from_format,
            to_format=to_format,
            adapter_type=type(adapter).__name__,
            plugin=plugin_name,
            category="format",
        )

    def get(self, from_format: str, to_format: str) -> FormatAdapterProtocol:
        if not from_format or not to_format:
            raise ValueError("Format names cannot be empty")

        key = (from_format, to_format)
        adapter = self._adapters.get(key)
        if adapter is None:
            available = ", ".join(
                f"{src}->{dst}" for src, dst in sorted(self._adapters)
            )
            raise ValueError(
                f"No adapter registered for {from_format}->{to_format}. Available: {available}"
            )
        return adapter

    def get_if_exists(
        self, from_format: str, to_format: str
    ) -> FormatAdapterProtocol | None:
        if not from_format or not to_format:
            raise ValueError("Format names cannot be empty")
        return self._adapters.get((from_format, to_format))

    def list_pairs(self) -> list[str]:
        return [f"{src}->{dst}" for src, dst in sorted(self._adapters)]

    def get_registered_plugins(self) -> set[str]:
        return set(self._registered_plugins.values())

    def clear(self) -> None:
        self._adapters.clear()
        self._registered_plugins.clear()

    async def register_from_manifest(
        self, manifest: PluginManifest, plugin_name: str
    ) -> None:
        for spec in manifest.format_adapters:
            adapter = spec.adapter_factory()
            if inspect.isawaitable(adapter):
                adapter = await adapter
            if not isinstance(adapter, FormatAdapterProtocol):
                raise TypeError(
                    f"Adapter factory for {spec.from_format}->{spec.to_format} returned invalid type {adapter!r}"
                )

            self.register(
                from_format=spec.from_format,
                to_format=spec.to_format,
                adapter=adapter,
                plugin_name=plugin_name,
            )

    def validate_requirements(
        self, manifests: dict[str, PluginManifest]
    ) -> dict[str, list[tuple[str, str]]]:
        available = set(self._adapters.keys())
        missing: dict[str, list[tuple[str, str]]] = {}
        for name, manifest in manifests.items():
            required = manifest.requires_format_adapters
            unresolved = [pair for pair in required if pair not in available]
            if unresolved:
                missing[name] = unresolved
        return missing


__all__ = ["FormatRegistry"]
