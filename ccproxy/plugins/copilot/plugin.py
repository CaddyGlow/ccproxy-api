"""GitHub Copilot plugin factory and runtime implementation."""

from typing import Any

from ccproxy.core.logging import get_plugin_logger
from ccproxy.core.plugins import (
    AuthProviderPluginFactory,
    AuthProviderPluginRuntime,
    BaseProviderPluginFactory,
    PluginContext,
    PluginManifest,
    ProviderPluginRuntime,
)
from ccproxy.core.plugins.declaration import FormatAdapterSpec, RouterSpec
from ccproxy.services.adapters.base import BaseAdapter

from .adapter import CopilotAdapter
from .config import CopilotConfig
from .detection_service import CopilotDetectionService
from .oauth.provider import CopilotOAuthProvider
from .routes import router_github, router_v1


logger = get_plugin_logger()


class CopilotPluginRuntime(ProviderPluginRuntime, AuthProviderPluginRuntime):
    """Runtime for GitHub Copilot plugin."""

    def __init__(self, manifest: PluginManifest):
        """Initialize runtime."""
        super().__init__(manifest)
        self.config: CopilotConfig | None = None
        self.adapter: CopilotAdapter | None = None
        self.oauth_provider: CopilotOAuthProvider | None = None
        self.detection_service: CopilotDetectionService | None = None

    async def _on_initialize(self) -> None:
        """Initialize the Copilot plugin."""
        logger.debug(
            "copilot_initializing",
            context_keys=list(self.context.keys()) if self.context else [],
        )

        # Get configuration
        if self.context:
            config = self.context.get("config")
            if not isinstance(config, CopilotConfig):
                config = CopilotConfig()
                logger.info("copilot_using_default_config")
            self.config = config

            # Get services from context
            self.oauth_provider = self.context.get("oauth_provider")
            self.detection_service = self.context.get("detection_service")
            self.adapter = self.context.get("adapter")

        # Call parent initialization - explicitly call both parent classes
        await ProviderPluginRuntime._on_initialize(self)
        await AuthProviderPluginRuntime._on_initialize(self)

        # Note: BaseHTTPAdapter doesn't have an initialize() method
        # Initialization is handled through dependency injection

        logger.debug(
            "copilot_plugin_initialized",
            status="initialized",
            has_oauth=bool(self.oauth_provider),
            has_detection=bool(self.detection_service),
            has_adapter=bool(self.adapter),
            category="plugin",
        )

    async def _setup_format_registry(self) -> None:
        """Format registry setup - using core Anthropic ↔ OpenAI adapters."""
        logger.debug(
            "copilot_using_core_format_adapters",
            required_adapters=["anthropic->openai"],
        )

    async def cleanup(self) -> None:
        """Cleanup plugin resources."""
        errors = []

        # Cleanup adapter
        if self.adapter:
            try:
                await self.adapter.cleanup()
            except Exception as e:
                errors.append(f"Adapter cleanup failed: {e}")
            finally:
                self.adapter = None

        # Cleanup OAuth provider
        if self.oauth_provider:
            try:
                await self.oauth_provider.cleanup()
            except Exception as e:
                errors.append(f"OAuth provider cleanup failed: {e}")
            finally:
                self.oauth_provider = None

        if errors:
            logger.error(
                "copilot_plugin_cleanup_failed",
                errors=errors,
            )
        else:
            logger.debug("copilot_plugin_cleanup_completed")


class CopilotPluginFactory(BaseProviderPluginFactory, AuthProviderPluginFactory):
    """Factory for GitHub Copilot plugin."""

    cli_safe = False  # Heavy provider - not for CLI use

    # Plugin configuration via class attributes
    plugin_name = "copilot"
    plugin_description = "GitHub Copilot provider plugin with OAuth authentication"
    runtime_class = CopilotPluginRuntime
    adapter_class = CopilotAdapter
    detection_service_class = CopilotDetectionService
    config_class = CopilotConfig
    routers = [
        RouterSpec(router=router_v1, prefix="/copilot/v1", tags=["copilot-api-v1"]),
        RouterSpec(router=router_github, prefix="/copilot", tags=["copilot-github"]),
    ]
    dependencies = []
    optional_requires = []

    # # Define format adapter dependencies (Anthropic ↔ OpenAI provided by core)
    # requires_format_adapters: list[FormatPair] = [
    #     (
    #         "anthropic",
    #         "openai",
    #     ),  # Provided by core OpenAI adapter for /v1/messages endpoint
    # ]

    format_adapters = [
        FormatAdapterSpec(
            from_format="anthropic",
            to_format="openai",
            adapter_factory=lambda: __import__(
                "ccproxy.adapters.openai.adapter",
                fromlist=["OpenAIAdapter"],
            ).OpenAIAdapter(),
            priority=100,  # Lower priority than SDK plugin
            description="Anthropic to OpenAI",
        ),
        FormatAdapterSpec(
            from_format="response_api",
            to_format="anthropic",
            adapter_factory=lambda: __import__(
                "ccproxy.adapters.anthropic.response_adapter",
                fromlist=["AnthropicResponseAPIAdapter"],
            ).AnthropicResponseAPIAdapter(),
            priority=50,  # Medium priority
            description="Response API to Anthropic format conversion for Claude API",
        ),
    ]

    def create_context(self, core_services: Any) -> PluginContext:
        """Create context with all plugin components.

        Args:
            core_services: Core services container

        Returns:
            Plugin context with all components
        """
        # Start with base context
        context = super().create_context(core_services)

        # Get or create configuration
        config = context.get("config")
        if not isinstance(config, CopilotConfig):
            config = CopilotConfig()
            context["config"] = config

        # Create OAuth provider
        oauth_provider = self.create_oauth_provider(context)
        context["oauth_provider"] = oauth_provider
        # Also set as auth_provider for AuthProviderPluginRuntime compatibility
        context["auth_provider"] = oauth_provider

        # Create detection service
        detection_service = self.create_detection_service(context)
        context["detection_service"] = detection_service

        # Note: adapter creation is handled asynchronously by create_runtime
        # in factories.py, so we don't create it here in the synchronous context creation

        return context

    def create_runtime(self) -> CopilotPluginRuntime:
        """Create runtime instance."""
        return CopilotPluginRuntime(self.manifest)

    def create_oauth_provider(
        self, context: PluginContext | None = None
    ) -> CopilotOAuthProvider:
        """Create OAuth provider instance.

        Args:
            context: Plugin context containing shared resources

        Returns:
            CopilotOAuthProvider instance
        """
        from typing import cast as _cast

        if context and isinstance(context.get("config"), CopilotConfig):
            cfg = _cast(CopilotConfig, context.get("config"))
        else:
            cfg = CopilotConfig()

        config: CopilotConfig = cfg
        http_client = context.get("http_client") if context else None
        hook_manager = context.get("hook_manager") if context else None
        cli_detection_service = (
            context.get("cli_detection_service") if context else None
        )

        return CopilotOAuthProvider(
            config.oauth,
            http_client=http_client,
            hook_manager=hook_manager,
            detection_service=cli_detection_service,
        )

    def create_detection_service(
        self, context: PluginContext | None = None
    ) -> CopilotDetectionService:
        """Create detection service instance.

        Args:
            context: Plugin context

        Returns:
            CopilotDetectionService instance
        """
        if not context:
            raise ValueError("Context required for detection service")

        settings = context.get("settings")
        cli_service = context.get("cli_detection_service")

        if not settings or not cli_service:
            raise ValueError("Settings and CLI detection service required")

        return CopilotDetectionService(settings, cli_service)

    async def create_adapter(self, context: PluginContext) -> BaseAdapter:
        """Create main adapter instance.

        Args:
            context: Plugin context

        Returns:
            CopilotAdapter instance
        """
        if not context:
            raise ValueError("Context required for adapter")

        config = context.get("config")
        if not isinstance(config, CopilotConfig):
            config = CopilotConfig()

        # Get required dependencies following BaseHTTPAdapter pattern
        oauth_provider = context.get("oauth_provider")
        detection_service = context.get("detection_service")
        http_pool_manager = context.get("http_pool_manager")

        # For Copilot, the oauth_provider serves as the auth_manager
        # since it has the required methods (ensure_copilot_token, etc.)
        auth_manager = oauth_provider

        # Optional dependencies
        request_tracer = context.get("request_tracer")
        metrics = context.get("metrics")
        streaming_handler = context.get("streaming_handler")
        hook_manager = context.get("hook_manager")

        # Get format_registry from service container
        service_container = context.get("service_container")
        format_registry = None
        if service_container:
            format_registry = service_container.get_format_registry()

        # Debug: Log what we actually have in the context
        logger.debug(
            "copilot_adapter_dependencies_debug",
            context_keys=list(context.keys()) if context else [],
            has_auth_manager=bool(auth_manager),
            has_detection_service=bool(detection_service),
            has_http_pool_manager=bool(http_pool_manager),
            has_oauth_provider=bool(oauth_provider),
            has_format_registry=bool(format_registry),
        )

        if not all(
            [auth_manager, detection_service, http_pool_manager, oauth_provider]
        ):
            missing = []
            if not auth_manager:
                missing.append("auth_manager")
            if not detection_service:
                missing.append("detection_service")
            if not http_pool_manager:
                missing.append("http_pool_manager")
            if not oauth_provider:
                missing.append("oauth_provider")

            raise ValueError(
                f"Required dependencies missing for CopilotAdapter: {missing}"
            )

        adapter = CopilotAdapter(
            auth_manager=auth_manager,
            detection_service=detection_service,
            http_pool_manager=http_pool_manager,
            oauth_provider=oauth_provider,
            config=config,
            request_tracer=request_tracer,
            metrics=metrics,
            streaming_handler=streaming_handler,
            hook_manager=hook_manager,
            format_registry=format_registry,
            context=context,
        )
        return adapter

    def create_auth_provider(
        self, context: PluginContext | None = None
    ) -> CopilotOAuthProvider:
        """Create OAuth provider instance for AuthProviderPluginFactory interface.

        Args:
            context: Plugin context containing shared resources

        Returns:
            CopilotOAuthProvider instance
        """
        return self.create_oauth_provider(context)


# Export the factory instance
factory = CopilotPluginFactory()
