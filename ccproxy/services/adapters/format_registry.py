from __future__ import annotations

import builtins
from typing import TYPE_CHECKING, Literal

import structlog

from ccproxy.adapters.base import APIAdapter
from ccproxy.llms.adapters.base import BaseAPIAdapter
from ccproxy.llms.adapters.shim import AdapterShim


if TYPE_CHECKING:
    from ccproxy.core.plugins import (
        FormatAdapterSpec,
        PluginManifest,
    )

logger = structlog.get_logger(__name__)


class FormatAdapterRegistry:
    """Registry for managing format adapters."""

    def __init__(
        self, conflict_mode: Literal["fail_fast", "priority"] = "fail_fast"
    ) -> None:
        self._adapters: dict[tuple[str, str], APIAdapter] = {}
        self._registered_plugins: set[str] = set()
        self._adapter_specs: dict[tuple[str, str], FormatAdapterSpec] = {}
        self._conflicts: dict[tuple[str, str], list[FormatAdapterSpec]] = {}
        self._finalized: bool = False
        self._conflict_mode = conflict_mode

    def register(
        self,
        from_format: str,
        to_format: str,
        adapter: APIAdapter,
        plugin_name: str = "unknown",
    ) -> None:
        """Register an adapter for format conversion with fail-fast validation."""
        if not from_format or not to_format:
            raise ValueError("Format names cannot be empty")
        if not adapter:
            raise ValueError("Adapter cannot be None")

        # Auto-wrap typed adapters with compatibility shim
        if isinstance(adapter, BaseAPIAdapter):
            logger.debug(
                "wrapping_typed_adapter_with_shim",
                adapter_type=type(adapter).__name__,
                from_format=from_format,
                to_format=to_format,
                plugin=plugin_name,
                category="format",
            )
            adapter = AdapterShim(adapter)

        key = (from_format, to_format)

        # Fail fast on duplicate registration attempts
        if key in self._adapters:
            existing_adapter = type(self._adapters[key]).__name__
            new_adapter = type(adapter).__name__
            raise ValueError(
                f"Adapter already registered for {from_format}->{to_format}: "
                f"existing={existing_adapter}, attempted={new_adapter}"
            )

        self._adapters[key] = adapter
        self._registered_plugins.add(plugin_name)

        logger.info(
            "format_adapter_registered",
            from_format=from_format,
            to_format=to_format,
            adapter_type=type(adapter).__name__,
            plugin=plugin_name,
            category="format",
        )

    def get(self, from_format: str, to_format: str) -> APIAdapter | None:
        """Get adapter for format conversion with fail-fast validation."""
        if not from_format or not to_format:
            raise ValueError("Format names cannot be empty")

        key = (from_format, to_format)
        adapter = self._adapters.get(key)

        logger.info(
            "format_adapter_requested",
            from_format=from_format,
            to_format=to_format,
            adapter_found=adapter is not None,
            adapter_type=type(adapter).__name__ if adapter else None,
            category="format",
        )

        if not adapter:
            available = list(self._adapters.keys())
            raise ValueError(
                f"No adapter found for {from_format}->{to_format}. "
                f"Available: {available}"
            )

        return adapter

    def get_if_exists(self, from_format: str, to_format: str) -> APIAdapter | None:
        """Get adapter if it exists, return None if not found (no exception).

        This method allows plugins to check for existing adapters without failing
        when the adapter doesn't exist, enabling adapter reuse patterns.

        Args:
            from_format: Source format name
            to_format: Target format name

        Returns:
            APIAdapter instance if found, None otherwise
        """
        if not from_format or not to_format:
            raise ValueError("Format names cannot be empty")

        key = (from_format, to_format)
        return self._adapters.get(key)

    def list(self) -> list[str]:
        """List all supported format combinations."""
        return [f"{from_fmt}->{to_fmt}" for from_fmt, to_fmt in self._adapters]

    def get_registered_plugins(self) -> set[str]:
        """Get list of plugins that have registered adapters."""
        return self._registered_plugins.copy()

    def clear(self) -> None:
        """Clear registry - for testing only."""
        self._adapters.clear()
        self._registered_plugins.clear()
        self._adapter_specs.clear()
        self._conflicts.clear()
        self._finalized = False

    async def register_from_manifest(
        self, manifest: PluginManifest, plugin_name: str
    ) -> None:
        """Register format adapters from plugin manifest."""
        if self._finalized:
            logger.warning(
                "format_adapter_registry_finalized_registration_ignored",
                plugin=plugin_name,
                category="format",
            )
            return

        for spec in manifest.format_adapters:
            format_pair = spec.format_pair

            # Track potential conflicts
            if format_pair in self._adapter_specs:
                if format_pair not in self._conflicts:
                    self._conflicts[format_pair] = [self._adapter_specs[format_pair]]
                self._conflicts[format_pair].append(spec)

                logger.debug(
                    "format_adapter_conflict_detected",
                    from_format=spec.from_format,
                    to_format=spec.to_format,
                    plugin=plugin_name,
                    existing_plugins=[
                        s.adapter_factory.__module__
                        for s in self._conflicts[format_pair]
                    ],
                    category="format",
                )
            else:
                self._adapter_specs[format_pair] = spec

        self._registered_plugins.add(plugin_name)

    def validate_requirements(
        self, manifests: dict[str, PluginManifest]
    ) -> dict[str, builtins.list[tuple[str, str]]]:
        """Validate format adapter requirements across all manifests."""
        # Get all available adapters (including core pre-registered ones)
        available = set(self._adapters.keys()) | set(self._adapter_specs.keys())

        missing = {}
        for name, manifest in manifests.items():
            missing_reqs = manifest.validate_format_adapter_requirements(available)
            if missing_reqs:
                missing[name] = missing_reqs

        return missing

    async def resolve_conflicts_and_finalize(
        self, enable_priority_mode: bool = False
    ) -> None:
        """Resolve conflicts and finalize registry."""
        if self._finalized:
            return

        # Handle conflicts based on mode
        if self._conflict_mode == "priority" or enable_priority_mode:
            await self._resolve_conflicts_by_priority()
        else:
            await self._fail_fast_on_conflicts()

        # Instantiate all registered adapters asynchronously
        for format_pair, spec in self._adapter_specs.items():
            if (
                format_pair not in self._adapters
            ):  # Don't override pre-registered core adapters
                try:
                    # All adapter factories are now standardized as Callable[[], APIAdapter]
                    adapter = spec.adapter_factory()
                    # Handle async factories if needed
                    if hasattr(adapter, "__await__"):
                        adapter = await adapter

                    # Wrap typed adapters with shim to accept dict IO
                    if isinstance(adapter, BaseAPIAdapter):
                        logger.debug(
                            "wrapping_typed_adapter_with_shim_during_finalization",
                            adapter_type=type(adapter).__name__,
                            from_format=spec.from_format,
                            to_format=spec.to_format,
                            category="format",
                        )
                        adapter = AdapterShim(adapter)

                    self._adapters[format_pair] = adapter
                except Exception as e:
                    logger.error(
                        "format_adapter_instantiation_failed",
                        from_format=spec.from_format,
                        to_format=spec.to_format,
                        factory=spec.adapter_factory.__name__,
                        error=str(e),
                        category="format",
                    )
                    raise ValueError(
                        f"Failed to instantiate adapter {spec.from_format} -> {spec.to_format}"
                    ) from e

        self._finalized = True
        logger.info(
            "format_adapter_registry_finalized",
            total_adapters=len(self._adapters),
            registered_plugins=list(self._registered_plugins),
            conflict_mode=self._conflict_mode,
            category="format",
        )

    async def _resolve_conflicts_by_priority(self) -> None:
        """Resolve conflicts using priority (lower = higher priority)."""
        for format_pair, conflicting_specs in self._conflicts.items():
            # Sort by priority (lower = higher priority)
            winner = min(conflicting_specs, key=lambda s: s.priority)
            self._adapter_specs[format_pair] = winner

            logger.warning(
                "format_adapter_conflict_resolved_by_priority",
                from_format=format_pair[0],
                to_format=format_pair[1],
                winner=winner.adapter_factory.__name__,
                winner_priority=winner.priority,
                conflicting_specs=[
                    {"name": s.adapter_factory.__name__, "priority": s.priority}
                    for s in conflicting_specs
                ],
                category="format",
            )

    async def _fail_fast_on_conflicts(self) -> None:
        """Fail fast when conflicts are detected (current behavior)."""
        if self._conflicts:
            conflict_details = []
            for format_pair, specs in self._conflicts.items():
                conflict_details.append(
                    {
                        "format_pair": format_pair,
                        "conflicting_adapters": [
                            s.adapter_factory.__name__ for s in specs
                        ],
                    }
                )

            logger.error(
                "format_adapter_conflicts_detected_failing_fast",
                conflicts=conflict_details,
                category="format",
            )

            raise ValueError(
                f"Format adapter conflicts detected: {conflict_details}. "
                "Enable priority mode to resolve automatically."
            )
