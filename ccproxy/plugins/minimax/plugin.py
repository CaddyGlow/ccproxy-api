"""MiniMax provider plugin factory and runtime implementation."""

from __future__ import annotations

from typing import Any

from ccproxy.core.constants import (
    FORMAT_ANTHROPIC_MESSAGES,
    FORMAT_OPENAI_CHAT,
)
from ccproxy.core.logging import get_plugin_logger
from ccproxy.core.plugins import (
    BaseProviderPluginFactory,
    FormatAdapterSpec,
    FormatPair,
    PluginManifest,
    ProviderPluginRuntime,
)
from ccproxy.core.plugins.declaration import RouterSpec
from ccproxy.llms.streaming.accumulators import OpenAIAccumulator

from .adapter import MiniMaxAdapter
from .config import MiniMaxConfig
from .routes import router as minimax_router


logger = get_plugin_logger()


class MiniMaxRuntime(ProviderPluginRuntime):
    """Runtime for the MiniMax provider plugin."""

    def __init__(self, manifest: PluginManifest):
        """Initialize runtime."""
        super().__init__(manifest)
        self.config: MiniMaxConfig | None = None

    async def _on_initialize(self) -> None:
        """Initialize the MiniMax provider plugin."""
        if not self.context:
            raise RuntimeError("Context not set")

        try:
            config = self.context.get(MiniMaxConfig)
        except ValueError:
            config = MiniMaxConfig()
            logger.debug("minimax_using_default_config")
        self.config = config

        # Base runtime wires up the adapter from the factory.
        await super()._on_initialize()

        logger.debug(
            "minimax_plugin_initialized",
            plugin="minimax",
            version=self.manifest.version,
            region=self.config.region if self.config else None,
            has_adapter=self.adapter is not None,
        )

    async def _get_health_details(self) -> dict[str, Any]:
        """Get health check details."""
        details = await super()._get_health_details()
        if self.config:
            details.update(
                {
                    "region": self.config.region,
                    "base_url": self.config.base_url,
                    "supports_streaming": self.config.supports_streaming,
                    "api_key_configured": bool(self.config.api_key),
                    "models": [card.id for card in self.config.models_endpoint],
                }
            )
        return details


class MiniMaxFactory(BaseProviderPluginFactory):
    """Factory for the MiniMax provider plugin."""

    cli_safe = False  # Provider plugin - not safe for CLI use

    plugin_name = "minimax"
    plugin_description = (
        "MiniMax provider plugin with API-key authentication and "
        "OpenAI/Anthropic format conversion"
    )
    runtime_class = MiniMaxRuntime
    adapter_class = MiniMaxAdapter
    config_class = MiniMaxConfig

    # Static API-key auth: no OAuth credential manager or detection service.
    auth_manager_name = None
    routers = [
        RouterSpec(router=minimax_router, prefix="/minimax", tags=["minimax-api"]),
    ]
    dependencies: list[str] = []
    optional_requires = ["pricing"]

    # No plugin-provided format adapters - core supplies the conversions used
    # by the MiniMax endpoints.
    format_adapters: list[FormatAdapterSpec] = []
    requires_format_adapters: list[FormatPair] = [
        (FORMAT_ANTHROPIC_MESSAGES, FORMAT_OPENAI_CHAT),
        (FORMAT_OPENAI_CHAT, FORMAT_ANTHROPIC_MESSAGES),
    ]
    tool_accumulator_class = OpenAIAccumulator


# Export the factory instance
factory = MiniMaxFactory()
