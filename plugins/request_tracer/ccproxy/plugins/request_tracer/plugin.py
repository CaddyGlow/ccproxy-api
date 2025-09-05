"""Request Tracer plugin implementation."""

import os
from typing import Any

from ccproxy.core.logging import get_plugin_logger
from ccproxy.core.plugins import (
    MiddlewareLayer,
    MiddlewareSpec,
    PluginContext,
    PluginManifest,
    SystemPluginFactory,
    SystemPluginRuntime,
)
from ccproxy.hooks import HookRegistry

from .config import RequestTracerConfig
from .hook import RequestTracerHook
from .hooks.http import HTTPTracerHook
from .middleware import RequestTracingMiddleware
from .tracer import RequestTracerImpl


logger = get_plugin_logger()


class RequestTracerRuntime(SystemPluginRuntime):
    """Runtime for request tracer plugin."""

    def __init__(self, manifest: PluginManifest):
        """Initialize runtime."""
        super().__init__(manifest)
        self.config: RequestTracerConfig | None = None
        self.tracer_instance: RequestTracerImpl | None = None
        self.hook: RequestTracerHook | None = None
        self.http_hook: HTTPTracerHook | None = None
        # Feature flag to control hook mode only
        self.use_hooks = os.getenv("HOOKS_ENABLED", "true").lower() == "true"

    async def _on_initialize(self) -> None:
        """Initialize the request tracer."""
        if not self.context:
            raise RuntimeError("Context not set")

        # Get configuration
        config = self.context.get("config")
        if not isinstance(config, RequestTracerConfig):
            logger.warning("plugin_no_config")
            # Use default config if none provided
            config = RequestTracerConfig()
            logger.info("plugin_using_default_config")
        self.config = config

        # Debug log the actual configuration being used
        logger.info(
            "plugin_configuration_loaded",
            enabled=config.enabled,
            json_logs_enabled=config.json_logs_enabled,
            raw_http_enabled=config.raw_http_enabled,
            verbose_api=config.verbose_api,
            log_dir=config.log_dir,
            exclude_paths=config.exclude_paths,
            log_client_request=config.log_client_request,
            log_client_response=config.log_client_response,
        )

        # Validate configuration
        validation_errors = self._validate_config(config)
        if validation_errors:
            logger.error(
                "plugin_config_validation_failed",
                errors=validation_errors,
                config=config.model_dump() if hasattr(config, 'model_dump') else str(config),
            )
            # Don't fail initialization, but log warnings
            for error in validation_errors:
                logger.warning("config_validation_warning", issue=error)

        # Create or reuse tracer instance
        # If factory created one (for middleware/raw HTTP), reuse it from context
        existing_tracer = self.context.get("request_tracer")
        if isinstance(existing_tracer, RequestTracerImpl):
            self.tracer_instance = existing_tracer
        else:
            self.tracer_instance = RequestTracerImpl(self.config)

        if self.config.enabled:
            if self.use_hooks:
                # Hook-based mode
                # Always register RequestTracerHook for JSON logging and client request/response events
                self.hook = RequestTracerHook(self.config, exclude_http_events=False)

                # Try to get hook registry from context
                hook_registry = None

                # Try direct from context first (provided by CoreServicesAdapter)
                hook_registry = self.context.get("hook_registry")
                logger.debug(
                    "hook_registry_from_context",
                    found=hook_registry is not None,
                    context_keys=list(self.context.keys()) if self.context else [],
                )

                # If not found, try app state
                if not hook_registry:
                    app = self.context.get("app")
                    if app and hasattr(app.state, "hook_registry"):
                        hook_registry = app.state.hook_registry
                        logger.debug("hook_registry_from_app_state", found=True)

                if hook_registry and isinstance(hook_registry, HookRegistry):
                    hook_registry.register(self.hook)

                    # Register HTTPTracerHook ONLY if middleware is not handling raw HTTP logging
                    # This prevents duplicate .http file writes
                    # Check if middleware was created by factory (via manifest.middleware)
                    middleware_handles_raw = (
                        self.config.raw_http_enabled and
                        hasattr(self.manifest, 'middleware') and
                        self.manifest.middleware is not None and
                        len(self.manifest.middleware) > 0
                    )

                    if self.config.raw_http_enabled and not middleware_handles_raw:
                        self.http_hook = HTTPTracerHook(self.config)
                        hook_registry.register(self.http_hook)
                        logger.info(
                            "http_tracer_hook_registered",
                            raw_http_enabled=True,
                            note="HTTPTracerHook registered for .http file generation (middleware not active)",
                        )

                    logger.info(
                        "request_tracer_hook_registered",
                        mode="hooks",
                        verbose_api=self.config.verbose_api,
                        raw_http=self.config.raw_http_enabled,
                        middleware_handles_raw=middleware_handles_raw,
                        http_hook_registered=self.http_hook is not None,
                        middleware_count=len(self.manifest.middleware) if hasattr(self.manifest, 'middleware') and self.manifest.middleware else 0,
                        manifest_has_middleware=hasattr(self.manifest, 'middleware') and self.manifest.middleware is not None,
                    )
                else:
                    logger.warning(
                        "hook_registry_not_available",
                        mode="hooks",
                        fallback=None,
                    )
                    # If hook registry not available, do not register
                    self.use_hooks = False

            # Register tracer with service container (for backward compatibility)
            service_container = self.context.get("service_container")
            if service_container and hasattr(service_container, "set_request_tracer"):
                # Use the public method to set the tracer
                service_container.set_request_tracer(self.tracer_instance)
                logger.info(
                    "request_tracer_registered",
                    verbose_api=self.config.verbose_api,
                    raw_http=self.config.raw_http_enabled,
                )

            # Note: Transport wrapping is not used with hooks mode
            # Raw HTTP logging is handled by the hooks/observer pattern

            logger.info(
                "request_tracer_enabled",
                log_dir=self.config.log_dir,
                verbose_api=self.config.verbose_api,
                json_logs=self.config.json_logs_enabled,
                raw_http=self.config.raw_http_enabled,
                log_client_request=self.config.log_client_request,
                log_client_response=self.config.log_client_response,
                log_provider_request=self.config.log_provider_request,
                log_provider_response=self.config.log_provider_response,
                max_body_size=self.config.max_body_size,
                exclude_paths=self.config.exclude_paths,
                exclude_headers=self.config.exclude_headers,
            )
        else:
            logger.info("request_tracer_disabled")

    def _validate_config(self, config: RequestTracerConfig) -> list[str]:
        """Validate plugin configuration.

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        if not config.enabled:
            return errors  # No validation needed if disabled

        # Validate directories
        from pathlib import Path

        try:
            log_dir = Path(config.log_dir)
            if config.json_logs_enabled:
                json_dir = Path(config.get_json_log_dir())
                if not json_dir.exists():
                    try:
                        json_dir.mkdir(parents=True, exist_ok=True)
                        logger.debug("created_json_log_directory", path=str(json_dir))
                    except OSError as e:
                        errors.append(f"Cannot create JSON log directory {json_dir}: {e}")

            if config.raw_http_enabled:
                raw_dir = Path(config.get_raw_log_dir())
                if not raw_dir.exists():
                    try:
                        raw_dir.mkdir(parents=True, exist_ok=True)
                        logger.debug("created_raw_log_directory", path=str(raw_dir))
                    except OSError as e:
                        errors.append(f"Cannot create raw HTTP log directory {raw_dir}: {e}")

        except Exception as e:
            errors.append(f"Directory validation failed: {e}")

        # Validate configuration consistency
        if not config.json_logs_enabled and not config.raw_http_enabled:
            errors.append("Both JSON logs and raw HTTP logging are disabled - no output will be generated")

        if config.max_body_size <= 0:
            errors.append("max_body_size must be positive")

        if config.truncate_body_preview <= 0:
            errors.append("truncate_body_preview must be positive")

        # Validate path filters
        if config.include_paths and config.exclude_paths:
            overlapping = set(config.include_paths) & set(config.exclude_paths)
            if overlapping:
                errors.append(f"Paths appear in both include and exclude lists: {overlapping}")

        return errors

    async def _on_shutdown(self) -> None:
        """Cleanup on shutdown."""
        # Unregister hook from registry
        if self.use_hooks and self.hook:
            # Try to get hook registry
            hook_registry = None
            if self.context:
                app = self.context.get("app")
                if app and hasattr(app.state, "hook_registry"):
                    hook_registry = app.state.hook_registry
                if not hook_registry:
                    hook_registry = self.context.get("hook_registry")

            if hook_registry and isinstance(hook_registry, HookRegistry):
                hook_registry.unregister(self.hook)
                if self.http_hook:
                    hook_registry.unregister(self.http_hook)
                    logger.debug("http_tracer_hook_unregistered", category="middleware")
                logger.debug("tracer_hook_unregistered", category="middleware")

        # Restore null tracer on shutdown
        if self.context:
            service_container = self.context.get("service_container")
            if service_container and hasattr(service_container, "set_request_tracer"):
                from ccproxy.services.tracing import NullRequestTracer

                service_container.set_request_tracer(NullRequestTracer())
                logger.debug("restored_null_tracer", category="middleware")

    async def _get_health_details(self) -> dict[str, Any]:
        """Get health check details."""
        details = {
            "type": "system",
            "initialized": self.initialized,
            "enabled": self.config.enabled if self.config else False,
        }

        if self.config and self.config.enabled:
            from pathlib import Path

            log_dir = Path(self.config.log_dir)
            details.update(
                {
                    "log_dir": str(log_dir),
                    "log_dir_exists": log_dir.exists(),
                    "verbose_api": self.config.verbose_api,
                    "json_logs": self.config.json_logs_enabled,
                    "raw_http": self.config.raw_http_enabled,
                }
            )

        return details


class RequestTracerFactory(SystemPluginFactory):
    """Factory for request tracer plugin."""

    cli_safe = True  # Safe for CLI - lightweight tracing only

    def __init__(self) -> None:
        """Initialize factory with manifest."""
        # Create manifest with static declarations
        manifest = PluginManifest(
            name="request_tracer",
            version="2.0.0",
            description="Unified request tracer with structured JSON and raw HTTP logging",
            is_provider=False,
            config_class=RequestTracerConfig,
        )

        # Initialize with manifest and runtime class
        super().__init__(manifest)

        # Store reference to tracer instance for middleware creation
        self._tracer_instance: RequestTracerImpl | None = None

    def create_runtime(self) -> RequestTracerRuntime:
        """Create runtime instance."""
        return RequestTracerRuntime(self.manifest)

    def create_context(self, core_services: Any) -> PluginContext:
        """Create context and update manifest with middleware if enabled."""
        # Get base context
        context = super().create_context(core_services)

        # Check if plugin is enabled and raw HTTP logging is enabled
        config = context.get("config")

        logger.debug(
            "factory_checking_middleware_creation_conditions",
            config_available=config is not None,
            config_type=type(config).__name__ if config else None,
            enabled=config.enabled if hasattr(config, 'enabled') else None,
            raw_http_enabled=config.raw_http_enabled if hasattr(config, 'raw_http_enabled') else None,
        )

        if (
            isinstance(config, RequestTracerConfig)
            and config.enabled
            and config.raw_http_enabled
        ):
            logger.info(
                "factory_creating_middleware_for_raw_http_logging",
                enabled=config.enabled,
                raw_http_enabled=config.raw_http_enabled,
                log_dir=config.log_dir,
            )
            # Disable HTTP compression for readable traces
            settings = getattr(core_services, "settings", None)
            if settings and hasattr(settings, "http"):
                settings.http.compression_enabled = False
                logger.info(
                    "request_tracer_disabled_compression",
                    message="Disabled HTTP compression for raw HTTP tracing",
                    reason="Ensures captured response bodies are human-readable in trace logs",
                    raw_http_enabled=config.raw_http_enabled,
                )

            # Create tracer instance for middleware
            self._tracer_instance = RequestTracerImpl(config)
            # Make it available to runtime via context to avoid double initialization
            # Override any placeholder/null tracer from core services
            context["request_tracer"] = self._tracer_instance

            # Add middleware to manifest
            # This is safe because it happens during app creation phase
            if not self.manifest.middleware:
                self.manifest.middleware = []

            # Create middleware spec with proper configuration
            # Note: We can't access hook_manager during factory creation,
            # so we'll pass it via a lazy loader pattern
            def get_hook_manager_from_app(app):
                return getattr(app.state, 'hook_manager', None)

            middleware_spec = MiddlewareSpec(
                middleware_class=RequestTracingMiddleware,  # type: ignore[arg-type]
                priority=MiddlewareLayer.OBSERVABILITY
                - 10,  # Early in observability layer
                kwargs={
                    "tracer": self._tracer_instance,
                    "hook_manager_factory": get_hook_manager_from_app,
                },
            )

            self.manifest.middleware.append(middleware_spec)
        else:
            logger.debug(
                "factory_skipping_middleware_creation",
                reason="conditions_not_met",
                config_available=config is not None,
                config_type=type(config).__name__ if config else None,
                enabled=config.enabled if hasattr(config, 'enabled') else None,
                raw_http_enabled=config.raw_http_enabled if hasattr(config, 'raw_http_enabled') else None,
            )

        return context


# Export the factory instance
factory = RequestTracerFactory()
