"""Request Tracer plugin implementation -  after refactoring."""

from typing import Any

from ccproxy.core.logging import get_plugin_logger
from ccproxy.core.plugins import (
    PluginManifest,
    SystemPluginFactory,
    SystemPluginRuntime,
)
from ccproxy.core.plugins.hooks import HookRegistry

from .config import RequestTracerConfig
from .hook import RequestTracerHook


logger = get_plugin_logger()


class RequestTracerRuntime(SystemPluginRuntime):
    """Runtime for the request tracer plugin.

    Handles only REQUEST_* events via a  hook.
    HTTP events are managed by the core HTTPTracerHook.
    """

    def __init__(self, manifest: PluginManifest):
        """Initialize runtime."""
        super().__init__(manifest)
        self.config: RequestTracerConfig | None = None
        self.hook: RequestTracerHook | None = None

    async def _on_initialize(self) -> None:
        """Initialize the  request tracer."""
        if not self.context:
            raise RuntimeError("Context not set")

        # Get configuration
        config = self.context.get("config")
        if not isinstance(config, RequestTracerConfig):
            logger.info("plugin_no_config")
            config = RequestTracerConfig()
            logger.debug("plugin_using_default_config")
        self.config = config

        # Debug log the actual configuration being used
        info_summaries_only = False
        try:
            app = self.context.get("app") if self.context else None
            info_summaries_only = (
                bool(getattr(app.state, "info_summaries_only", False)) if app else False
            )
        except Exception:
            info_summaries_only = False

        (logger.debug if info_summaries_only else logger.info)(
            "plugin_configuration_loaded",
            enabled=config.enabled,
            json_logs_enabled=config.json_logs_enabled,
            verbose_api=config.verbose_api,
            log_dir=config.log_dir,
            exclude_paths=config.exclude_paths,
            log_client_request=config.log_client_request,
            log_client_response=config.log_client_response,
            note="HTTP events handled by core HTTPTracerHook",
        )

        # Validate configuration
        validation_errors = self._validate_config(config)
        if validation_errors:
            logger.error(
                "plugin_config_validation_failed",
                errors=validation_errors,
                config=config.model_dump()
                if hasattr(config, "model_dump")
                else str(config),
            )
            for error in validation_errors:
                logger.warning("config_validation_warning", issue=error)

        if self.config.enabled:
            # Register  hook for REQUEST_* events only
            self.hook = RequestTracerHook(self.config)

            # Try to get hook registry from context
            hook_registry = self.context.get("hook_registry")

            # If not found, try app state
            if not hook_registry:
                app = self.context.get("app")
                if app and hasattr(app.state, "hook_registry"):
                    hook_registry = app.state.hook_registry

            if hook_registry and isinstance(hook_registry, HookRegistry):
                hook_registry.register(self.hook)
                (logger.debug if info_summaries_only else logger.info)(
                    "request_tracer_hook_registered",
                    mode="hooks",
                    json_logs=self.config.json_logs_enabled,
                    verbose_api=self.config.verbose_api,
                    note="HTTP events handled by core HTTPTracerHook",
                )
            else:
                logger.warning(
                    "hook_registry_not_available",
                    mode="hooks",
                    fallback="disabled",
                )

            (logger.debug if info_summaries_only else logger.info)(
                "request_tracer_enabled",
                log_dir=self.config.log_dir,
                json_logs=self.config.json_logs_enabled,
                exclude_paths=self.config.exclude_paths,
                architecture="hooks_only",
            )
        else:
            (logger.debug if info_summaries_only else logger.info)(
                "request_tracer_disabled"
            )

    def _validate_config(self, config: RequestTracerConfig) -> list[str]:
        """Validate plugin configuration.

        Returns:
            List of validation error messages (empty if valid)
        """
        errors: list[str] = []

        if not config.enabled:
            return errors  # No validation needed if disabled

        # Basic path validation
        try:
            from pathlib import Path

            log_path = Path(config.log_dir)
            if not log_path.parent.exists():
                errors.append(
                    f"Parent directory of log_dir does not exist: {log_path.parent}"
                )
        except Exception as e:
            errors.append(f"Invalid log_dir path: {e}")

        # Configuration consistency checks
        if not config.json_logs_enabled and not config.verbose_api:
            errors.append(
                "No logging output enabled (json_logs_enabled=False, verbose_api=False)"
            )

        if config.max_body_size < 0:
            errors.append("max_body_size cannot be negative")

        return errors

    async def _on_shutdown(self) -> None:
        """Cleanup resources."""
        if self.hook:
            logger.debug("shutting_down_request_tracer_hook")
            self.hook = None
        logger.debug("request_tracer_plugin_shutdown_complete")


class RequestTracerFactory(SystemPluginFactory):
    """factory for request tracer plugin."""

    def __init__(self) -> None:
        """Initialize factory with manifest."""
        # Create manifest with static declarations ( from original)
        manifest = PluginManifest(
            name="request_tracer",
            version="2.0.0",  # Version bump to reflect refactoring
            description=" request tracing for REQUEST_* events only",
            is_provider=False,
            config_class=RequestTracerConfig,
        )

        # Initialize with manifest
        super().__init__(manifest)

        logger.info(
            "request_tracer_manifest_created",
            version="2.0.0",
            architecture="hooks_only",
            note="HTTP events handled by core HTTPTracerHook",
        )

    def create_runtime(self) -> RequestTracerRuntime:
        """Create runtime instance."""
        return RequestTracerRuntime(self.manifest)

    def create_context(self, core_services: Any) -> Any:
        """Create context for the plugin."""
        # Get base context from parent
        context = super().create_context(core_services)

        return context


# Export the factory instance for entry points
factory = RequestTracerFactory()
