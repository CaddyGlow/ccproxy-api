"""Claude API plugin hooks for streaming metrics extraction."""

import json
from typing import Any

from ccproxy.core.logging import get_plugin_logger
from ccproxy.core.plugins.hooks import Hook, HookContext, HookEvent

from .streaming_metrics import extract_usage_from_streaming_chunk


logger = get_plugin_logger()


class ClaudeAPIStreamingMetricsHook(Hook):
    """Hook to extract and accumulate metrics from Claude API streaming responses."""

    name = "claude_api_streaming_metrics"
    events = [HookEvent.PROVIDER_STREAM_CHUNK, HookEvent.PROVIDER_STREAM_END]
    priority = 700  # HookLayer.OBSERVATION - Metrics collection layer

    def __init__(
        self, pricing_service: Any = None, plugin_registry: Any = None
    ) -> None:
        """Initialize with optional pricing service for cost calculation.

        Args:
            pricing_service: Direct pricing service instance (if available at init)
            plugin_registry: Plugin registry to get pricing service lazily
        """
        self.pricing_service = pricing_service
        self.plugin_registry = plugin_registry
        # Store metrics per request_id
        self._metrics_cache: dict[str, dict[str, Any]] = {}

    def _get_pricing_service(self) -> Any:
        """Get pricing service, trying lazy loading if not already available."""
        if self.pricing_service:
            return self.pricing_service

        if self.plugin_registry:
            try:
                from ccproxy.plugins.pricing.service import PricingService

                self.pricing_service = self.plugin_registry.get_service(
                    "pricing", PricingService
                )
                if self.pricing_service:
                    logger.debug(
                        "pricing_service_obtained_lazily",
                        plugin="claude_api",
                    )
            except Exception as e:
                logger.debug(
                    "lazy_pricing_service_failed",
                    plugin="claude_api",
                    error=str(e),
                )

        return self.pricing_service

    async def __call__(self, context: HookContext) -> None:
        """Extract metrics from streaming chunks and add to stream end events."""
        # Only process claude_api provider events
        if context.provider != "claude_api":
            return

        request_id = context.metadata.get("request_id")
        if not request_id:
            return

        if context.event == HookEvent.PROVIDER_STREAM_CHUNK:
            await self._process_chunk(context, request_id)
        elif context.event == HookEvent.PROVIDER_STREAM_END:
            await self._finalize_metrics(context, request_id)

    async def _process_chunk(self, context: HookContext, request_id: str) -> None:
        """Process a streaming chunk to extract metrics."""
        chunk_data = context.data.get("chunk")
        if not chunk_data:
            return

        # Debug: Log chunk type and sample
        logger.debug(
            "chunk_received",
            plugin="claude_api",
            request_id=request_id,
            chunk_type=type(chunk_data).__name__,
            chunk_sample=str(chunk_data)[:200] if chunk_data else None,
        )

        # Initialize metrics cache for this request if needed
        if request_id not in self._metrics_cache:
            self._metrics_cache[request_id] = {
                "tokens_input": None,
                "tokens_output": None,
                "cache_read_tokens": None,
                "cache_write_tokens": None,
                "cost_usd": None,
                "model": None,
            }

        try:
            # Handle bytes data
            if isinstance(chunk_data, bytes):
                chunk_data = chunk_data.decode("utf-8")

            # Parse SSE data if it's a string
            if isinstance(chunk_data, str):
                # Look for data lines in SSE format
                for line in chunk_data.split("\n"):
                    if line.startswith("data: "):
                        data_str = line[6:].strip()
                        if data_str and data_str != "[DONE]":
                            event_data = json.loads(data_str)
                            self._extract_and_accumulate(event_data, request_id)
                            break
            elif isinstance(chunk_data, dict):
                # Direct dict chunk
                self._extract_and_accumulate(chunk_data, request_id)

        except (json.JSONDecodeError, KeyError) as e:
            logger.debug(
                "chunk_metrics_parse_failed",
                plugin="claude_api",
                error=str(e),
                request_id=request_id,
            )

    def _extract_and_accumulate(
        self, event_data: dict[str, Any], request_id: str
    ) -> None:
        """Extract metrics from parsed event data and accumulate."""
        usage_data = extract_usage_from_streaming_chunk(event_data)

        if not usage_data:
            return

        cache = self._metrics_cache[request_id]
        event_type = usage_data.get("event_type")

        # Handle message_start: get input tokens and initial cache tokens
        if event_type == "message_start":
            cache["tokens_input"] = usage_data.get("input_tokens")
            cache["cache_read_tokens"] = (
                usage_data.get("cache_read_input_tokens") or cache["cache_read_tokens"]
            )
            cache["cache_write_tokens"] = (
                usage_data.get("cache_creation_input_tokens")
                or cache["cache_write_tokens"]
            )

            # Extract model from the message_start event
            if not cache["model"] and usage_data.get("model"):
                cache["model"] = usage_data.get("model")

            logger.debug(
                "hook_metrics_extracted",
                plugin="claude_api",
                event_type="message_start",
                tokens_input=cache["tokens_input"],
                cache_read_tokens=cache["cache_read_tokens"],
                cache_write_tokens=cache["cache_write_tokens"],
                model=cache["model"],
                request_id=request_id,
            )

        # Handle message_delta: get final output tokens
        elif event_type == "message_delta":
            cache["tokens_output"] = usage_data.get("output_tokens")

            # Calculate cost if we have all required data
            pricing_service = self._get_pricing_service()
            logger.debug(
                "hook_calculating_cost",
                plugin="claude_api",
                request_id=request_id,
                pricing_service=bool(pricing_service is not None),
                model=cache["model"],
            )
            if pricing_service and cache["model"]:
                try:
                    from ccproxy.plugins.pricing.exceptions import (
                        ModelPricingNotFoundError,
                        PricingDataNotLoadedError,
                        PricingServiceDisabledError,
                    )

                    cost_decimal = pricing_service.calculate_cost_sync(
                        model_name=cache["model"],
                        input_tokens=cache["tokens_input"] or 0,
                        output_tokens=cache["tokens_output"] or 0,
                        cache_read_tokens=cache["cache_read_tokens"] or 0,
                        cache_write_tokens=cache["cache_write_tokens"] or 0,
                    )
                    cache["cost_usd"] = float(cost_decimal)

                    logger.debug(
                        "hook_cost_calculated",
                        plugin="claude_api",
                        model=cache["model"],
                        cost_usd=cache["cost_usd"],
                        request_id=request_id,
                    )
                except (
                    ModelPricingNotFoundError,
                    PricingDataNotLoadedError,
                    PricingServiceDisabledError,
                ) as e:
                    logger.debug(
                        "hook_cost_calculation_skipped",
                        plugin="claude_api",
                        reason=str(e),
                        request_id=request_id,
                    )
                except Exception as e:
                    logger.debug(
                        "hook_cost_calculation_failed",
                        plugin="claude_api",
                        error=str(e),
                        request_id=request_id,
                    )

            logger.debug(
                "hook_metrics_extracted",
                plugin="claude_api",
                event_type="message_delta",
                tokens_output=cache["tokens_output"],
                cost_usd=cache.get("cost_usd"),
                request_id=request_id,
            )

    async def _finalize_metrics(self, context: HookContext, request_id: str) -> None:
        """Add accumulated metrics to the PROVIDER_STREAM_END event."""
        if request_id not in self._metrics_cache:
            return

        metrics = self._metrics_cache.pop(request_id, {})

        # Add metrics to the event's usage_metrics field
        if not context.data.get("usage_metrics"):
            context.data["usage_metrics"] = {}

        # Update with our collected metrics
        if metrics["tokens_input"] is not None:
            context.data["usage_metrics"]["input_tokens"] = metrics["tokens_input"]
        if metrics["tokens_output"] is not None:
            context.data["usage_metrics"]["output_tokens"] = metrics["tokens_output"]
        if metrics["cache_read_tokens"] is not None:
            context.data["usage_metrics"]["cache_read_input_tokens"] = metrics[
                "cache_read_tokens"
            ]
        if metrics["cache_write_tokens"] is not None:
            context.data["usage_metrics"]["cache_creation_input_tokens"] = metrics[
                "cache_write_tokens"
            ]
        if metrics["cost_usd"] is not None:
            context.data["usage_metrics"]["cost_usd"] = metrics["cost_usd"]
        if metrics["model"]:
            context.data["model"] = metrics["model"]

        logger.info(
            "streaming_metrics_finalized",
            plugin="claude_api",
            request_id=request_id,
            usage_metrics=context.data.get("usage_metrics", {}),
            context_data_keys=list(context.data.keys()) if context.data else [],
        )
