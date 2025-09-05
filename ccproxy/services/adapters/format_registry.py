import structlog

from ccproxy.adapters.base import APIAdapter


logger = structlog.get_logger(__name__)


class FormatAdapterRegistry:
    """Registry for managing format adapters and their relationships."""

    def __init__(self) -> None:
        self._adapters: dict[tuple[str, str], APIAdapter] = {}
        self._registered_plugins: set[str] = set()

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
        )

    def get_adapter(self, from_format: str, to_format: str) -> APIAdapter | None:
        """Get adapter for format conversion with fail-fast validation."""
        if not from_format or not to_format:
            raise ValueError("Format names cannot be empty")

        key = (from_format, to_format)
        adapter = self._adapters.get(key)

        if not adapter:
            available = list(self._adapters.keys())
            raise ValueError(
                f"No adapter found for {from_format}->{to_format}. "
                f"Available: {available}"
            )

        return adapter

    def get_adapter_if_exists(
        self, from_format: str, to_format: str
    ) -> APIAdapter | None:
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

    def list_formats(self) -> list[str]:
        """List all supported format combinations."""
        return [f"{from_fmt}->{to_fmt}" for from_fmt, to_fmt in self._adapters]

    def get_registered_plugins(self) -> set[str]:
        """Get list of plugins that have registered adapters."""
        return self._registered_plugins.copy()

    def clear_registry(self) -> None:
        """Clear registry - for testing only."""
        self._adapters.clear()
        self._registered_plugins.clear()
