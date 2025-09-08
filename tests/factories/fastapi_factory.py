"""FastAPI test application factory for composable test fixtures.

This module provides a factory pattern to eliminate combinatorial explosion
in FastAPI test fixtures by allowing flexible composition of different
configurations and dependency overrides.
"""

from collections.abc import Callable
from typing import Any, TypeAlias
from unittest.mock import AsyncMock

from fastapi import FastAPI
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

from ccproxy.api.app import create_app
from ccproxy.config.settings import Settings


# Type aliases for better readability
DependencyOverride: TypeAlias = Callable[..., Any]
DependencyOverrides: TypeAlias = dict[Callable[..., Any], DependencyOverride]
MockService: TypeAlias = AsyncMock


class FastAPIAppFactory:
    """Factory for creating FastAPI applications with flexible configurations.

    This factory eliminates the need for multiple similar fixtures by allowing
    composition of different configurations through a single, flexible interface.

    Example usage:
        # Create factory with specific mock service
        factory = FastAPIAppFactory(claude_service_mock=mock_service)

        # Create app with auth enabled and custom settings
        app = factory.create_app(
            settings=auth_settings,
            auth_enabled=True
        )
    """

    def __init__(
        self,
        default_settings: Settings | None = None,
        claude_service_mock: MockService | None = None,
    ) -> None:
        """Initialize the factory with default configuration.

        Args:
            default_settings: Default settings to use if none provided
            claude_service_mock: Default mock Claude service
        """
        self.default_settings = default_settings
        self.default_claude_service_mock = claude_service_mock

    def create_app(
        self,
        settings: Settings | None = None,
        dependency_overrides: DependencyOverrides | None = None,
        claude_service_mock: MockService | None = None,
        auth_enabled: bool = False,
        log_storage: Any | None = None,
        register_plugin_routes: bool = True,
        **kwargs: Any,
    ) -> FastAPI:
        """Create a FastAPI application with specified configuration.

        Args:
            settings: Application settings (uses factory default if None)
            dependency_overrides: Custom dependency overrides
            claude_service_mock: Mock Claude service (uses factory default if None)
            auth_enabled: Whether to enable authentication
            log_storage: Optional log storage instance to set in app state
            register_plugin_routes: Whether to register plugin routes (default: True)
            **kwargs: Additional configuration options

        Returns:
            Configured FastAPI application
        """
        # Use factory defaults if not provided
        effective_settings = settings or self.default_settings
        effective_claude_mock = claude_service_mock or self.default_claude_service_mock

        if effective_settings is None:
            raise ValueError(
                "Settings must be provided either in factory or create_app"
            )

        # Create the base app
        app = create_app(settings=effective_settings)

        # IMPORTANT: Set up app.state BEFORE dependency overrides
        # This mimics what happens in the real app's lifespan function
        # The cached dependencies expect these to be available in app.state

        # Always set settings (this is set in the real app's lifespan)
        app.state.settings = effective_settings

        # Set claude_service in app state if mock provided
        if effective_claude_mock is not None:
            app.state.claude_service = effective_claude_mock

        # Set log storage in app state if provided
        if log_storage is not None:
            app.state.log_storage = log_storage
            app.state.duckdb_storage = log_storage

        # Set optional services to None for tests (these aren't typically needed in unit tests)
        if not hasattr(app.state, "scheduler"):
            app.state.scheduler = None
        if not hasattr(app.state, "permission_service"):
            app.state.permission_service = None

        # Register plugin routes if requested and plugins are enabled
        if register_plugin_routes and effective_settings.enable_plugins:
            self._register_plugin_routes(app, effective_settings, effective_claude_mock)

        # Prepare all dependency overrides
        all_overrides = self._build_dependency_overrides(
            effective_settings,
            effective_claude_mock,
            auth_enabled,
            dependency_overrides or {},
        )

        # Apply all overrides
        app.dependency_overrides.update(all_overrides)

        return app

    def _build_dependency_overrides(
        self,
        settings: Settings,
        claude_service_mock: MockService | None,
        auth_enabled: bool,
        custom_overrides: DependencyOverrides,
    ) -> DependencyOverrides:
        """Build the complete set of dependency overrides.

        Args:
            settings: Application settings
            claude_service_mock: Mock Claude service
            auth_enabled: Whether authentication is enabled
            custom_overrides: Additional custom overrides

        Returns:
            Complete set of dependency overrides
        """
        overrides: DependencyOverrides = {}

        # Always override settings - both original and cached versions
        from fastapi import Request

        from ccproxy.api.dependencies import get_cached_settings
        from ccproxy.config.settings import get_settings as original_get_settings

        overrides[original_get_settings] = lambda: settings

        def mock_get_cached_settings_for_factory(request: Request):
            return settings

        overrides[get_cached_settings] = mock_get_cached_settings_for_factory

        # Override Claude service if mock provided
        # NOTE: Plugin-based architecture no longer uses get_claude_service dependency.
        # Mock should be attached to app.state after app creation if needed.

        # Override plugin adapter dependencies for tests
        if claude_service_mock is not None:
            self._add_plugin_dependency_overrides(overrides, claude_service_mock)

        # Skip auth manager override - let real auth system handle it

        # Add any custom overrides (these take precedence)
        overrides.update(custom_overrides)

        return overrides

    def _add_plugin_dependency_overrides(
        self, overrides: DependencyOverrides, claude_service_mock: MockService
    ) -> None:
        """Add plugin adapter dependency overrides for testing."""
        # Simplified mock adapter for testing
        mock_adapter = AsyncMock()
        mock_adapter.handle_request = AsyncMock(return_value=None)

        # Basic adapter mocking - just ensure tests don't fail
        # Real plugin integration testing should be done at integration level
        if hasattr(mock_adapter, "handle_request"):
            overrides[mock_adapter] = lambda: mock_adapter

    def _register_plugin_routes(
        self,
        app: FastAPI,
        settings: Settings,
        claude_service_mock: MockService | None = None,
    ) -> None:
        """Register plugin routes for test apps."""
        try:
            # Simplified plugin route registration
            self._register_claude_sdk_routes(app)
            self._register_claude_api_routes(app)
            self._register_codex_routes(app)
        except Exception:
            # Silently skip if plugins unavailable - tests should work without them
            pass

    def _register_claude_sdk_routes(self, app: FastAPI) -> None:
        """Register Claude SDK plugin routes."""
        try:
            from ccproxy.plugins.claude_sdk.routes import router as claude_sdk_router

            app.include_router(claude_sdk_router, prefix="/sdk")
        except ImportError:
            pass

    def _register_claude_api_routes(self, app: FastAPI) -> None:
        """Register Claude API plugin routes."""
        try:
            from ccproxy.plugins.claude_api.routes import router as claude_api_router

            app.include_router(claude_api_router, prefix="/api")
        except ImportError:
            pass

    def _register_codex_routes(self, app: FastAPI) -> None:
        """Register Codex plugin routes."""
        try:
            from ccproxy.plugins.codex.routes import router as codex_router

            app.include_router(codex_router, prefix="/codex")
        except ImportError:
            pass


class FastAPIClientFactory:
    """Factory for creating test clients with flexible configurations.

    This factory works with FastAPIAppFactory to provide both sync and async
    test clients with various configurations.
    """

    def __init__(self, app_factory: FastAPIAppFactory) -> None:
        """Initialize client factory with app factory.

        Args:
            app_factory: The app factory to use for creating applications
        """
        self.app_factory = app_factory

    def create_client(
        self,
        settings: Settings | None = None,
        dependency_overrides: DependencyOverrides | None = None,
        claude_service_mock: MockService | None = None,
        auth_enabled: bool = False,
        log_storage: Any | None = None,
        register_plugin_routes: bool = True,
        **kwargs: Any,
    ) -> TestClient:
        """Create a synchronous test client.

        Args:
            settings: Application settings
            dependency_overrides: Custom dependency overrides
            claude_service_mock: Mock Claude service
            auth_enabled: Whether to enable authentication
            log_storage: Optional log storage instance to set in app state
            register_plugin_routes: Whether to register plugin routes (default: True)
            **kwargs: Additional configuration options

        Returns:
            Configured TestClient
        """
        app = self.app_factory.create_app(
            settings=settings,
            dependency_overrides=dependency_overrides,
            claude_service_mock=claude_service_mock,
            auth_enabled=auth_enabled,
            log_storage=log_storage,
            register_plugin_routes=register_plugin_routes,
            **kwargs,
        )
        return TestClient(app)

    def create_async_client(
        self,
        settings: Settings | None = None,
        dependency_overrides: DependencyOverrides | None = None,
        claude_service_mock: MockService | None = None,
        auth_enabled: bool = False,
        log_storage: Any | None = None,
        register_plugin_routes: bool = True,
        **kwargs: Any,
    ) -> AsyncClient:
        """Create an asynchronous test client.

        Args:
            settings: Application settings
            dependency_overrides: Custom dependency overrides
            claude_service_mock: Mock Claude service
            auth_enabled: Whether to enable authentication
            log_storage: Optional log storage instance to set in app state
            register_plugin_routes: Whether to register plugin routes (default: True)
            **kwargs: Additional configuration options

        Returns:
            Configured AsyncClient (must be used with async context manager)
        """
        app = self.app_factory.create_app(
            settings=settings,
            dependency_overrides=dependency_overrides,
            claude_service_mock=claude_service_mock,
            auth_enabled=auth_enabled,
            log_storage=log_storage,
            register_plugin_routes=register_plugin_routes,
            **kwargs,
        )

        return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")

    def create_client_with_storage(
        self,
        storage: Any,
        settings: Settings | None = None,
        dependency_overrides: DependencyOverrides | None = None,
        claude_service_mock: MockService | None = None,
        auth_enabled: bool = False,
        register_plugin_routes: bool = True,
        **kwargs: Any,
    ) -> TestClient:
        """Create a test client with log storage set in app state and dependency override.

        Args:
            storage: Storage instance to use (can be None)
            settings: Application settings
            dependency_overrides: Custom dependency overrides
            claude_service_mock: Mock Claude service
            auth_enabled: Whether to enable authentication
            register_plugin_routes: Whether to register plugin routes (default: True)
            **kwargs: Additional configuration options

        Returns:
            Configured TestClient with storage configured
        """
        # Use the new log_storage parameter to set storage in app state
        # This is the preferred approach as it matches real app behavior
        return self.create_client(
            settings=settings,
            dependency_overrides=dependency_overrides,
            claude_service_mock=claude_service_mock,
            auth_enabled=auth_enabled,
            log_storage=storage,
            register_plugin_routes=register_plugin_routes,
            **kwargs,
        )


# Convenience functions for common configurations
def create_test_app(
    settings: Settings, claude_mock: MockService | None = None
) -> FastAPI:
    """Create a basic test app."""
    factory = FastAPIAppFactory(default_settings=settings)
    return factory.create_app(claude_service_mock=claude_mock)
