"""Max tokens plugin implementation."""

from typing import Any

from ccproxy.core.logging import get_plugin_logger
from ccproxy.core.plugins import (
    PluginManifest,
    SystemPluginFactory,
    SystemPluginRuntime,
)
from ccproxy.core.plugins.hooks import HookRegistry

from .adapter import MaxTokensHook
from .config import MaxTokensConfig
from .service import TokenLimitsService


logger = get_plugin_logger(__name__)


class MaxTokensRuntime(SystemPluginRuntime):
    """Runtime for max_tokens plugin."""

    def __init__(self, manifest: PluginManifest):
        """Initialize runtime."""
        super().__init__(manifest)
        self.config: MaxTokensConfig | None = None
        self.service: TokenLimitsService | None = None
        self.hook: MaxTokensHook | None = None
        self.hook_registered = False

    async def _on_initialize(self) -> None:
        """Initialize the max tokens plugin."""
        if not self.context:
            raise RuntimeError("Context not set")

        # Get configuration
        config = self.context.get("config")
        if not isinstance(config, MaxTokensConfig):
            logger.debug("plugin_no_config_using_defaults", category="plugin")
            # Use default config if none provided
            self.config = MaxTokensConfig()
        else:
            self.config = config

        logger.debug("initializing_max_tokens_plugin", enabled=self.config.enabled)

        if not self.config.enabled:
            logger.debug("max_tokens_plugin_disabled")
            return

        # Create and initialize service
        self.service = TokenLimitsService(self.config)
        self.service.initialize()

        # Create hook instance
        self.hook = MaxTokensHook(self.config, self.service)

        # Attempt to register hook with available registry sources
        hook_registry: HookRegistry | None = None

        # 1. Directly from context (preferred when ServiceContainer wires it)
        hook_registry = self.context.get("hook_registry")

        # 2. Fallback to service container if available
        if not hook_registry:
            container = self.context.get("service_container")
            if container:
                try:
                    hook_registry = container.get_hook_registry()
                except Exception as container_error:  # pragma: no cover - defensive
                    logger.debug(
                        "max_tokens_hook_registry_from_container_failed",
                        error=str(container_error),
                    )

        # 3. Fallback to app.state when running inside FastAPI app
        if not hook_registry:
            app = self.context.get("app")
            if app and hasattr(app.state, "hook_registry"):
                hook_registry = app.state.hook_registry

        if hook_registry and isinstance(hook_registry, HookRegistry):
            hook_registry.register(self.hook)
            self.hook_registered = True
            logger.debug(
                "max_tokens_hook_registered",
                providers="*"
                if self.config.apply_to_all_providers
                else self.config.target_providers,
                fallback=self.config.fallback_max_tokens,
            )
        else:
            logger.warning(
                "max_tokens_hook_registry_unavailable",
                message="max_tokens adjustments disabled",
            )

        logger.info(
            "max_tokens_plugin_initialized",
            target_providers=self.config.target_providers,
            fallback_max_tokens=self.config.fallback_max_tokens,
        )

    async def _on_shutdown(self) -> None:
        """Shutdown the plugin and cleanup resources."""
        logger.debug("shutting_down_max_tokens_plugin")

        # Unregister hook if we registered one
        if self.hook:
            hook_registry: HookRegistry | None = None
            if self.context:
                hook_registry = self.context.get("hook_registry")
                if not hook_registry:
                    container = self.context.get("service_container")
                    if container:
                        try:
                            hook_registry = container.get_hook_registry()
                        except Exception as container_error:  # pragma: no cover
                            logger.debug(
                                "max_tokens_hook_registry_from_container_failed_shutdown",
                                error=str(container_error),
                            )
                if not hook_registry:
                    app = self.context.get("app")
                    if app and hasattr(app.state, "hook_registry"):
                        hook_registry = app.state.hook_registry

            if hook_registry and isinstance(hook_registry, HookRegistry):
                hook_registry.unregister(self.hook)
                self.hook_registered = False
                logger.debug("max_tokens_hook_unregistered")

            self.hook = None

        if self.service:
            self.service = None

        logger.debug("max_tokens_plugin_shutdown_complete")

    async def _get_health_details(self) -> dict[str, Any]:
        """Get health check details."""
        try:
            base_health = {
                "type": "system",
                "initialized": self.initialized,
                "enabled": self.config.enabled if self.config else False,
                "hook_registered": self.hook_registered,
            }

            if not self.config or not self.config.enabled:
                return base_health

            # Add service health info
            health_details = base_health.copy()

            if self.service:
                health_details["models_count"] = len(
                    self.service.token_limits_data.models
                )
                health_details["fallback_max_tokens"] = self.config.fallback_max_tokens

            return health_details

        except Exception as e:
            logger.error("health_check_failed", error=str(e))
            return {
                "type": "system",
                "initialized": self.initialized,
                "enabled": self.config.enabled if self.config else False,
                "error": str(e),
                "hook_registered": self.hook_registered,
            }


class MaxTokensFactory(SystemPluginFactory):
    """Factory for max_tokens plugin."""

    def __init__(self) -> None:
        """Initialize factory with manifest."""
        # Create manifest - max_tokens logic is now integrated into HTTP adapter
        manifest = PluginManifest(
            name="max_tokens",
            version="0.1.0",
            description="Automatically sets max_tokens based on model limits when missing or invalid",
            is_provider=False,
            config_class=MaxTokensConfig,
            provides=["max_tokens"],  # This plugin provides the max_tokens service
        )

        # Initialize with manifest
        super().__init__(manifest)

    def create_runtime(self) -> MaxTokensRuntime:
        """Create runtime instance."""
        return MaxTokensRuntime(self.manifest)


# Export the factory instance
factory = MaxTokensFactory()
