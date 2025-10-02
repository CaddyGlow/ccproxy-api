"""Max tokens adapter implementation using hook system."""

from __future__ import annotations

import json
from typing import Any

from ccproxy.core.logging import get_plugin_logger
from ccproxy.core.plugins.hooks import Hook, HookContext, HookEvent
from ccproxy.core.plugins.hooks.layers import HookLayer

from .config import MaxTokensConfig
from .service import TokenLimitsService


logger = get_plugin_logger(__name__)


class MaxTokensHook(Hook):
    """Hook that enforces `max_tokens` limits before provider dispatch."""

    name = "max_tokens"
    events = [HookEvent.PROVIDER_REQUEST_PREPARED]
    priority = HookLayer.PROCESSING  # Run before observation hooks

    def __init__(self, config: MaxTokensConfig, service: TokenLimitsService):
        """Initialize max tokens hook."""
        self.config = config
        self.service = service

    async def __call__(self, context: HookContext) -> None:
        """Adjust token limits in provider request payload when required."""
        if not self.config.enabled:
            return

        if context.event is not HookEvent.PROVIDER_REQUEST_PREPARED:
            return

        provider = context.provider
        if provider and not self.config.should_process_provider(provider):
            return

        payload = context.data.get("body")
        if not isinstance(payload, dict):
            return

        model = payload.get("model")
        if not isinstance(model, str):
            return

        provider_model = context.metadata.get("provider_model")
        if not isinstance(provider_model, str):
            provider_model = model

        alias_map = context.metadata.get("_model_alias_map")
        client_model = context.metadata.get("client_model")
        if not client_model and isinstance(alias_map, dict):
            client_model = alias_map.get(provider_model) or alias_map.get(model)

        modifiers = context.data.setdefault("modifiers", {})
        modified = False

        # Ensure we work with the latest payload returned from the service
        modified_payload, modification = self.service.modify_max_tokens(
            payload, provider_model, provider
        )

        if modification and modification.was_modified():
            payload = modified_payload
            context.data["body"] = payload
            modifiers["max_tokens"] = {
                "original": modification.original_max_tokens,
                "new": modification.new_max_tokens,
                "reason": modification.reason,
            }
            modified = True
        else:
            payload = modified_payload
            context.data["body"] = payload

        current_max_output = payload.get("max_output_tokens")
        provider_limit = self.service.get_max_output_tokens(provider_model)
        if provider_limit is None and self.config.fallback_max_tokens:
            provider_limit = self.config.fallback_max_tokens

        new_max_output: int | None = None
        output_reason: str | None = None

        if isinstance(current_max_output, int) and current_max_output > 0:
            original_limit = (
                self.service.get_max_output_tokens(client_model)
                if client_model
                else None
            )

            if (
                provider_limit
                and original_limit
                and client_model
                and client_model != provider_model
                and current_max_output == original_limit
                and provider_limit != original_limit
            ):
                new_max_output = provider_limit
                output_reason = "max_output_tokens_aligned_with_mapped_model"
            elif provider_limit and current_max_output > provider_limit:
                new_max_output = provider_limit
                output_reason = "max_output_tokens_capped_to_provider_limit"

        if new_max_output is not None and new_max_output != current_max_output:
            payload["max_output_tokens"] = new_max_output
            modifiers["max_output_tokens"] = {
                "original": current_max_output,
                "new": new_max_output,
                "reason": output_reason,
            }
            modified = True

            if self.config.log_modifications:
                logger.info(
                    "max_output_tokens_adjusted",
                    provider=provider,
                    provider_model=provider_model,
                    client_model=client_model,
                    original=current_max_output,
                    new=new_max_output,
                    reason=output_reason,
                )

        if modified:
            context.data["body"] = payload
            context.data["body_kind"] = "json"
            context.data["body_raw"] = json.dumps(payload).encode("utf-8")


class MaxTokensAdapter:
    """Max tokens adapter using hook-based request interception."""

    def __init__(self, config: MaxTokensConfig):
        """Initialize max tokens adapter."""
        self.config = config
        self.service = TokenLimitsService(config)
        self.hook = MaxTokensHook(config, self.service)
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the adapter and register hooks."""
        if self._initialized:
            return

        if not self.config.enabled:
            logger.debug("max_tokens_adapter_disabled")
            return

        # Initialize the service
        self.service.initialize()

        # Register hook with hook manager
        try:
            from ccproxy.services.container import ServiceContainer

            container = ServiceContainer.get_current(strict=False)
            if container:
                try:
                    hook_registry = container.get_hook_registry()
                    if hook_registry:
                        hook_registry.register(self.hook)
                        logger.debug(
                            "max_tokens_hook_registered",
                            hook_name=self.hook.name,
                            events=[event.value for event in self.hook.events],
                            priority=self.hook.priority,
                        )
                    else:
                        logger.warning("no_hook_registry_available")
                except Exception as service_error:
                    logger.warning(
                        "hook_registry_service_not_available",
                        error=str(service_error),
                    )
            else:
                logger.warning("no_service_container_available")

        except Exception as e:
            logger.error(
                "failed_to_register_max_tokens_hook",
                error=str(e),
                exc_info=e,
            )
            # Continue without hook registration - plugin will be effectively disabled

        self._initialized = True
        logger.info("max_tokens_adapter_initialized")

    async def cleanup(self) -> None:
        """Cleanup resources."""
        if not self._initialized:
            return

        try:
            # Unregister hook
            from ccproxy.services.container import ServiceContainer

            container = ServiceContainer.get_current(strict=False)
            if container:
                try:
                    hook_registry = container.get_hook_registry()
                    if hook_registry:
                        hook_registry.unregister(self.hook)
                        logger.debug("max_tokens_hook_unregistered")
                except Exception as service_error:
                    logger.debug(
                        "hook_registry_service_not_available_during_cleanup",
                        error=str(service_error),
                    )

        except Exception as e:
            logger.error(
                "failed_to_unregister_max_tokens_hook",
                error=str(e),
                exc_info=e,
            )

        self._initialized = False
        logger.debug("max_tokens_adapter_cleanup_completed")

    def get_modification_stats(self) -> dict[str, Any]:
        """Get statistics about max_tokens modifications."""
        # This could be enhanced to track modification statistics
        return {
            "adapter_initialized": self._initialized,
            "config_enabled": self.config.enabled,
            "target_providers": self.config.target_providers,
            "fallback_max_tokens": self.config.fallback_max_tokens,
        }
