"""Request Tracer plugin implementation - after refactoring."""

from pathlib import Path
from typing import Any

from ccproxy.core.logging import get_plugin_logger
from ccproxy.core.plugins import (
    PluginManifest,
    SystemPluginFactory,
    SystemPluginRuntime,
)
from ccproxy.core.plugins.hooks import HookRegistry
from ccproxy.core.plugins.hooks.implementations import HTTPTracerHook
from ccproxy.core.plugins.hooks.implementations.formatters import (
    JSONFormatter,
    RawHTTPFormatter,
)

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
        self.http_tracer_hook: HTTPTracerHook | None = None

    async def _on_initialize(self) -> None:
        """Initialize the  request tracer."""
        if not self.context:
            raise RuntimeError("Context not set")

        # Get configuration
        config = self.context.get("config")
        if not isinstance(config, RequestTracerConfig):
            config = RequestTracerConfig()

        self.config = config

        # Debug log the actual configuration being used
        logger.debug(
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

        # Try to get hook registry from context
        hook_registry: HookRegistry | None = self.context.get("hook_registry")

        # If not found, try app state
        if not hook_registry:
            app = self.context.get("app")
            if app and hasattr(app.state, "hook_registry"):
                hook_registry = app.state.hook_registry

        if not hook_registry or not isinstance(hook_registry, HookRegistry):
            logger.warning(
                "hook_registry_not_available",
                mode="hooks",
                fallback="disabled",
            )
            return

        settings = self.context.get("settings") if self.context else None

        json_formatter, raw_formatter = self._build_formatters(settings)

        self.http_tracer_hook = HTTPTracerHook(
            json_formatter=json_formatter,
            raw_formatter=raw_formatter,
            enabled=True,
        )
        hook_registry.register(self.http_tracer_hook)
        logger.debug(
            "core_http_tracer_registered",
            mode="hooks",
            json_logs_enabled=self.config.json_logs_enabled,
            raw_http_enabled=(raw_formatter is not None),
            log_dir=json_formatter.log_dir if json_formatter else None,
        )

        # Register REQUEST_* hook for contextual logging
        self.hook = RequestTracerHook(self.config)
        hook_registry.register(self.hook)
        logger.debug(
            "request_tracer_hook_registered",
            mode="hooks",
            json_logs=self.config.json_logs_enabled,
            verbose_api=self.config.verbose_api,
            note="HTTP events handled by core HTTPTracerHook",
        )

        logger.debug(
            "request_tracer_enabled",
            log_dir=self.config.log_dir,
            json_logs=self.config.json_logs_enabled,
            exclude_paths=self.config.exclude_paths,
            architecture="hooks_only",
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
        self.http_tracer_hook = None
        logger.debug("request_tracer_plugin_shutdown_complete")

    def _build_formatters(
        self,
        settings: Any,
    ) -> tuple[JSONFormatter | None, RawHTTPFormatter | None]:
        """Construct formatters based on plugin and global logging settings."""

        if not self.config:
            return (None, None)

        # Determine logging base directory overrides from global settings
        json_log_dir = Path(self.config.get_json_log_dir())
        raw_log_dir = Path(self.config.get_raw_log_dir())

        override_base: Path | None = None
        if settings and getattr(settings, "logging", None):
            plugin_base = getattr(settings.logging, "plugin_log_base_dir", None)
            if plugin_base:
                override_base = Path(plugin_base) / "tracer"

        fields_set: set[str] = getattr(self.config, "model_fields_set", set())
        if override_base:
            if "request_log_dir" not in fields_set and "log_dir" not in fields_set:
                json_log_dir = override_base
            if "raw_log_dir" not in fields_set and "log_dir" not in fields_set:
                raw_log_dir = override_base

        json_formatter: JSONFormatter | None = None
        if self.config.json_logs_enabled:
            json_formatter = JSONFormatter(
                log_dir=str(json_log_dir),
                verbose_api=self.config.verbose_api,
                json_logs_enabled=self.config.json_logs_enabled,
                redact_sensitive=self.config.redact_sensitive,
                truncate_body_preview=self.config.truncate_body_preview,
            )

        raw_formatter: RawHTTPFormatter | None = None
        raw_logging_enabled = self.config.raw_http_enabled
        if raw_logging_enabled:
            raw_formatter = RawHTTPFormatter(
                log_dir=str(raw_log_dir),
                enabled=True,
                log_client_request=self.config.log_client_request,
                log_client_response=self.config.log_client_response,
                log_provider_request=self.config.log_provider_request,
                log_provider_response=self.config.log_provider_response,
                max_body_size=self.config.max_body_size,
                exclude_headers=self.config.exclude_headers,
            )

        return (json_formatter, raw_formatter)


class RequestTracerFactory(SystemPluginFactory):
    """factory for request tracer plugin."""

    def __init__(self) -> None:
        """Initialize factory with manifest."""
        # Create manifest with static declarations ( from original)
        manifest = PluginManifest(
            name="request_tracer",
            version="0.1.0",  # Standardized initial plugin version
            description=" request tracing for REQUEST_* events only",
            is_provider=False,
            config_class=RequestTracerConfig,
        )

        # Initialize with manifest
        super().__init__(manifest)

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
