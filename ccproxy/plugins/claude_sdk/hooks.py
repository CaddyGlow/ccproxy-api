"""Hook integration for Claude SDK plugin to emit streaming metrics."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from ccproxy.core.logging import get_plugin_logger
from ccproxy.core.plugins.hooks import Hook, HookContext, HookEvent, HookManager


logger = get_plugin_logger(__name__)


class ClaudeSDKStreamingHook(Hook):
    """Hook for emitting Claude SDK streaming metrics.

    This hook handles streaming completion events from claude_sdk and emits
    PROVIDER_STREAM_END events with usage metrics for access logging.
    """

    name = "claude_sdk_streaming_metrics"
    events = []  # We'll emit events directly, not listen to them
    priority = 700  # HookLayer.METRICS

    def __init__(self, hook_manager: HookManager | None = None) -> None:
        """Initialize the Claude SDK streaming hook.

        Args:
            hook_manager: Hook manager for emitting events
        """
        self.hook_manager = hook_manager

    async def emit_stream_end(
        self,
        request_id: str,
        usage_metrics: dict[str, Any],
        provider: str = "claude_sdk",
        url: str = "claude-sdk://direct",
        method: str = "POST",
        total_chunks: int = 0,
        total_bytes: int = 0,
    ) -> None:
        """Emit PROVIDER_STREAM_END event with usage metrics.

        Args:
            request_id: Request ID for correlation
            usage_metrics: Dictionary containing token counts and costs
            provider: Provider name (default: claude_sdk)
            url: URL or endpoint identifier
            method: HTTP method
            total_chunks: Number of chunks streamed
            total_bytes: Total bytes streamed
        """
        if not self.hook_manager:
            logger.debug(
                "no_hook_manager_for_stream_end",
                request_id=request_id,
                provider=provider,
            )
            return

        try:
            # Normalize usage metrics to standard format
            normalized_metrics = {
                "input_tokens": usage_metrics.get("tokens_input", 0),
                "output_tokens": usage_metrics.get("tokens_output", 0),
                "cache_read_input_tokens": usage_metrics.get("cache_read_tokens", 0),
                "cache_creation_input_tokens": usage_metrics.get(
                    "cache_write_tokens", 0
                ),
                "cost_usd": usage_metrics.get("cost_usd", 0.0),
                "model": usage_metrics.get("model", ""),
            }

            stream_end_context = HookContext(
                event=HookEvent.PROVIDER_STREAM_END,
                timestamp=datetime.now(),
                provider=provider,
                data={
                    "url": url,
                    "method": method,
                    "request_id": request_id,
                    "total_chunks": total_chunks,
                    "total_bytes": total_bytes,
                    "usage_metrics": normalized_metrics,
                },
                metadata={
                    "request_id": request_id,
                },
            )

            await self.hook_manager.emit_with_context(stream_end_context)

            logger.info(
                "claude_sdk_stream_end_emitted",
                request_id=request_id,
                tokens_input=normalized_metrics["input_tokens"],
                tokens_output=normalized_metrics["output_tokens"],
                cost_usd=normalized_metrics["cost_usd"],
                model=normalized_metrics["model"],
            )

        except Exception as e:
            logger.error(
                "claude_sdk_hook_emission_failed",
                event="PROVIDER_STREAM_END",
                error=str(e),
                request_id=request_id,
                exc_info=e,
            )

    async def __call__(self, context: HookContext) -> None:
        """Handle hook events (not used for this hook as we emit directly)."""
        pass
