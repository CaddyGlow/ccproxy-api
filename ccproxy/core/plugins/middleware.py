"""Middleware management and ordering for the plugin system.

This module provides utilities for managing middleware registration
and ensuring proper ordering across core and plugin middleware.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI

from ccproxy.core.logging import TraceBoundLogger, get_logger

from .declaration import MiddlewareLayer, MiddlewareSpec


if TYPE_CHECKING:
    from starlette.middleware.base import BaseHTTPMiddleware
else:
    from starlette.middleware.base import BaseHTTPMiddleware


logger: TraceBoundLogger = get_logger()


@dataclass
class CoreMiddlewareSpec(MiddlewareSpec):
    """Specification for core application middleware.

    Extends MiddlewareSpec with a source field to distinguish
    between core and plugin middleware.
    """

    source: str = "core"  # "core" or plugin name


class MiddlewareManager:
    """Manages middleware registration and ordering."""

    def __init__(self) -> None:
        """Initialize middleware manager."""
        self.middleware_specs: list[CoreMiddlewareSpec] = []

    def add_core_middleware(
        self,
        middleware_class: type[BaseHTTPMiddleware],
        priority: int = MiddlewareLayer.APPLICATION,
        **kwargs: Any,
    ) -> None:
        """Add core application middleware.

        Args:
            middleware_class: Middleware class
            priority: Priority for ordering
            **kwargs: Additional middleware arguments
        """
        spec = CoreMiddlewareSpec(
            middleware_class=middleware_class,
            priority=priority,
            kwargs=kwargs,
            source="core",
        )
        self.middleware_specs.append(spec)

    def add_plugin_middleware(
        self, plugin_name: str, specs: list[MiddlewareSpec]
    ) -> None:
        """Add middleware from a plugin.

        Args:
            plugin_name: Name of the plugin
            specs: List of middleware specifications
        """
        for spec in specs:
            core_spec = CoreMiddlewareSpec(
                middleware_class=spec.middleware_class,
                priority=spec.priority,
                kwargs=spec.kwargs,
                source=plugin_name,
            )
            self.middleware_specs.append(core_spec)
            logger.trace(
                "plugin_middleware_added",
                plugin=plugin_name,
                middleware=spec.middleware_class.__name__,
                priority=spec.priority,
                category="middleware",
            )

    def get_ordered_middleware(self) -> list[CoreMiddlewareSpec]:
        """Get all middleware sorted by priority.

        Returns:
            List of middleware specs sorted by priority (lower first)
        """
        # Sort by priority (lower values first)
        # Secondary sort by source (core before plugins) for same priority
        return sorted(
            self.middleware_specs,
            key=lambda x: (x.priority, x.source != "core", x.source),
        )

    def apply_to_app(self, app: FastAPI) -> None:
        """Apply all middleware to the FastAPI app in correct order.

        Note: Middleware in FastAPI/Starlette is applied in reverse order
        (last added runs first), so we add them in reverse priority order.

        Args:
            app: FastAPI application
        """
        ordered = self.get_ordered_middleware()
        applied_middleware = []
        failed_middleware = []

        # Apply in reverse order (highest priority last so it runs first)
        for spec in reversed(ordered):
            try:
                app.add_middleware(spec.middleware_class, **spec.kwargs)  # type: ignore[arg-type]
                applied_middleware.append(
                    {
                        "name": spec.middleware_class.__name__,
                        "priority": spec.priority,
                        "source": spec.source,
                    }
                )
            except Exception as e:
                failed_middleware.append(
                    {
                        "name": spec.middleware_class.__name__,
                        "source": spec.source,
                        "error": str(e),
                    }
                )
                logger.error(
                    "middleware_application_failed",
                    middleware=spec.middleware_class.__name__,
                    source=spec.source,
                    error=str(e),
                    exc_info=e,
                    category="middleware",
                )

        # Log aggregated success
        if applied_middleware:
            logger.info(
                "middleware_stack_configured",
                applied=len(applied_middleware),
                failed=len(failed_middleware),
                middleware=[m["name"] for m in applied_middleware],
                category="middleware",
            )

    def get_middleware_summary(self) -> dict[str, Any]:
        """Get a summary of registered middleware.

        Returns:
            Dictionary with middleware statistics and order
        """
        ordered = self.get_ordered_middleware()

        summary = {
            "total": len(ordered),
            "core": len([m for m in ordered if m.source == "core"]),
            "plugins": len([m for m in ordered if m.source != "core"]),
            "order": [
                {
                    "name": spec.middleware_class.__name__,
                    "priority": spec.priority,
                    "layer": self._get_layer_name(spec.priority),
                    "source": spec.source,
                }
                for spec in ordered
            ],
        }

        return summary

    def _get_layer_name(self, priority: int) -> str:
        """Get the layer name for a priority value.

        Args:
            priority: Priority value

        Returns:
            Layer name
        """
        # Find the closest layer
        for layer in MiddlewareLayer:
            if priority < layer:
                return f"before_{layer.name.lower()}"
            elif priority == layer:
                return layer.name.lower()

        # If higher than all layers
        return "after_application"


def setup_default_middleware(manager: MiddlewareManager) -> None:
    """Setup default core middleware.

    Args:
        manager: Middleware manager
    """
    from ccproxy.api.middleware.hooks import HooksMiddleware
    from ccproxy.api.middleware.normalize_headers import NormalizeHeadersMiddleware
    from ccproxy.api.middleware.request_id import RequestIDMiddleware

    # Request ID should be first (lowest priority) to set context for all others
    manager.add_core_middleware(
        RequestIDMiddleware,
        priority=MiddlewareLayer.SECURITY - 50,  # Before security layer
    )

    # Hooks middleware should be early to capture all requests
    manager.add_core_middleware(
        HooksMiddleware,
        priority=MiddlewareLayer.SECURITY
        - 40,  # After request ID, before other middleware
    )

    # # Access logging in observability layer
    # manager.add_core_middleware(
    #     AccessLogMiddleware, priority=MiddlewareLayer.OBSERVABILITY
    # )
    #
    # Normalize headers: strip unsafe and ensure server header
    manager.add_core_middleware(
        NormalizeHeadersMiddleware,  # type: ignore[arg-type]
        priority=MiddlewareLayer.ROUTING,  # after routing layer
    )

    logger.debug("default_middleware_configured", category="middleware")
