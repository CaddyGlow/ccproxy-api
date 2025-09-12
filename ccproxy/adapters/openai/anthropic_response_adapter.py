"""Direct Anthropic Messages ↔ OpenAI Response API conversion adapter.

This adapter provides direct conversion between Anthropic Messages format and OpenAI Response API format
without chaining through OpenAI Chat Completions format. The formats are structurally very similar,
making direct conversion much more efficient and reliable.

Format Comparison:
- Anthropic: messages, max_tokens, system, tools
- Response API: input, max_completion_tokens, instructions, tools
- Both use same content block structure: {"type": "text", "text": "..."}
"""

import json
from collections.abc import AsyncGenerator, AsyncIterator
from typing import Any

from ccproxy.adapters.base import APIAdapter
from ccproxy.core.logging import get_logger


logger = get_logger()


class AnthropicResponseAPIAdapter(APIAdapter):
    """Direct adapter for Anthropic Messages ↔ OpenAI Response API conversion."""

    async def adapt_request(self, request: dict[str, Any]) -> dict[str, Any]:
        """Convert Anthropic Messages request to OpenAI Response API format.

        Args:
            request: Anthropic Messages API request

        Returns:
            OpenAI Response API formatted request
        """
        try:
            response_api_request = {}

            # Direct field mappings
            if "model" in request:
                response_api_request["model"] = request["model"]

            # messages → input (same structure, same content blocks)
            if "messages" in request:
                response_api_request["input"] = self._convert_messages_to_input(
                    request["messages"]
                )

            # system → instructions
            if "system" in request:
                response_api_request["instructions"] = request["system"]

            # Direct copy supported fields
            for field in ["stream", "tools", "tool_choice", "parallel_tool_calls"]:
                if field in request:
                    response_api_request[field] = request[field]

            # Handle temperature - Response API might not support it, but pass through
            if "temperature" in request:
                response_api_request["temperature"] = request["temperature"]

            if "max_completion_tokens" in response_api_request:
                response_api_request.pop("max_completion_tokens")
            # Mandatory fixed field for codex
            response_api_request["model"] = "gpt-5"
            response_api_request["store"] = False

            logger.debug(
                "anthropic_to_response_api_conversion",
                original_keys=list(request.keys()),
                converted_keys=list(response_api_request.keys()),
                input_messages=len(response_api_request.get("input", [])),
            )

            return response_api_request

        except Exception as e:
            logger.error(
                "anthropic_to_response_api_conversion_failed",
                error=str(e),
                request_keys=list(request.keys())
                if isinstance(request, dict)
                else "not_dict",
                exc_info=e,
            )
            raise

    async def adapt_response(self, response: dict[str, Any]) -> dict[str, Any]:
        """Convert OpenAI Response API response to Anthropic Messages format.

        Args:
            response: OpenAI Response API response

        Returns:
            Anthropic Messages formatted response
        """
        logger.info(
            "anthropic_response_adapter_received",
            response_keys=list(response.keys()) if response else [],
            response_preview=str(response)[:500] if response else "empty",
            response_type=type(response).__name__,
        )

        # Check if this is an error response - convert to Anthropic format
        if "error" in response:
            logger.info(
                "anthropic_adapter_error_response_detected",
                error_type=response.get("error", {}).get("type"),
                error_message=response.get("error", {}).get("message", ""),
            )
            return await self.adapt_error(response)

        try:
            # Extract content from Response API format
            content_blocks = []
            stop_reason = "end_turn"

            # Handle SSE event wrapper format
            actual_response = response
            if "response" in response and isinstance(response["response"], dict):
                logger.info(
                    "anthropic_adapter_using_nested_response",
                    nested_keys=list(response["response"].keys()),
                )
                actual_response = response["response"]

            # Response API has nested structure: output -> message -> content
            if "output" in actual_response:
                output = actual_response["output"]
                logger.info(
                    "anthropic_adapter_found_output",
                    output_type=type(output).__name__,
                    output_length=len(output)
                    if isinstance(output, list)
                    else "not_list",
                )

                if isinstance(output, list):
                    for i, item in enumerate(output):
                        logger.info(
                            f"anthropic_adapter_output_item_{i}",
                            item_type=type(item).__name__,
                            item_keys=list(item.keys())
                            if isinstance(item, dict)
                            else "not_dict",
                        )
                        if isinstance(item, dict):
                            # Handle direct content blocks (legacy format)
                            if item.get("type") == "text" and "text" in item:
                                content_blocks.append(
                                    {"type": "text", "text": item["text"]}
                                )
                            elif item.get("type") == "tool_use":
                                # Convert tool use blocks
                                content_blocks.append(
                                    {
                                        "type": "tool_use",
                                        "id": item.get("id", ""),
                                        "name": item.get("name", ""),
                                        "input": item.get("input", {}),
                                    }
                                )
                            # Handle nested message format (current Codex format)
                            elif item.get("type") == "message":
                                message_content = item.get("content", [])
                                for content_block in message_content:
                                    if isinstance(content_block, dict):
                                        if (
                                            content_block.get("type")
                                            in ["text", "output_text"]
                                            and "text" in content_block
                                        ):
                                            extracted_text = content_block["text"]
                                            logger.info(
                                                "anthropic_adapter_extracted_text",
                                                text_length=len(extracted_text),
                                                text_preview=extracted_text[:100],
                                            )
                                            content_blocks.append(
                                                {"type": "text", "text": extracted_text}
                                            )
                                        elif content_block.get("type") == "tool_use":
                                            content_blocks.append(
                                                {
                                                    "type": "tool_use",
                                                    "id": content_block.get("id", ""),
                                                    "name": content_block.get(
                                                        "name", ""
                                                    ),
                                                    "input": content_block.get(
                                                        "input", {}
                                                    ),
                                                }
                                            )
                elif isinstance(output, str):
                    # Simple string output
                    content_blocks.append({"type": "text", "text": output})

            # Handle choices format (similar to OpenAI)
            elif "choices" in response:
                choices = response["choices"]
                if choices and len(choices) > 0:
                    choice = choices[0]
                    message = choice.get("message", {})

                    # Extract text content
                    if "content" in message and message["content"]:
                        content_blocks.append(
                            {"type": "text", "text": message["content"]}
                        )

                    # Extract tool calls
                    if "tool_calls" in message:
                        for tool_call in message["tool_calls"]:
                            function = tool_call.get("function", {})
                            arguments = function.get("arguments", "{}")

                            # Parse arguments if they're a JSON string
                            if isinstance(arguments, str):
                                try:
                                    parsed_args = json.loads(arguments)
                                except json.JSONDecodeError:
                                    parsed_args = {}
                            else:
                                parsed_args = arguments

                            content_blocks.append(
                                {
                                    "type": "tool_use",
                                    "id": tool_call.get("id", ""),
                                    "name": function.get("name", ""),
                                    "input": parsed_args,
                                }
                            )

                    # Map finish reason
                    finish_reason = choice.get("finish_reason", "stop")
                    stop_reason_map = {
                        "stop": "end_turn",
                        "length": "max_tokens",
                        "tool_calls": "tool_use",
                        "content_filter": "stop_sequence",
                    }
                    stop_reason = stop_reason_map.get(finish_reason, "end_turn")

            # Default content if nothing found
            if not content_blocks:
                content_blocks = [{"type": "text", "text": ""}]

            # Build Anthropic response
            anthropic_response: dict[str, Any] = {
                "content": content_blocks,
                "stop_reason": stop_reason,
            }

            logger.info(
                "anthropic_adapter_final_result",
                content_blocks_count=len(content_blocks),
                has_content=bool(content_blocks),
                result_preview=str(anthropic_response)[:200],
            )

            # Add usage information
            if "usage" in actual_response:
                usage = actual_response["usage"]
                if isinstance(usage, dict):
                    anthropic_response["usage"] = {
                        "input_tokens": usage.get("prompt_tokens", 0),
                        "output_tokens": usage.get("completion_tokens", 0),
                    }

            # Add model and id if available
            if "model" in response:
                anthropic_response["model"] = response["model"]
            if "id" in response:
                anthropic_response["id"] = response["id"]

            logger.debug(
                "response_api_to_anthropic_conversion",
                original_keys=list(response.keys()),
                converted_keys=list(anthropic_response.keys()),
                content_blocks=len(content_blocks),
                stop_reason=stop_reason,
            )

            return anthropic_response

        except Exception as e:
            logger.error(
                "response_api_to_anthropic_conversion_failed",
                error=str(e),
                response_keys=list(response.keys())
                if isinstance(response, dict)
                else "not_dict",
                exc_info=e,
            )
            raise

    def adapt_stream(
        self, stream: AsyncIterator[dict[str, Any]]
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Convert streaming response between Response API and Anthropic formats.

        Args:
            stream: Streaming response data

        Yields:
            Converted streaming response chunks
        """
        logger.info("anthropic_response_api_adapter_stream_called")
        return self._adapt_stream_impl(stream)

    async def _adapt_stream_impl(
        self, stream: AsyncIterator[dict[str, Any]]
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Implementation of streaming conversion."""
        try:
            logger.debug("anthropic_response_api_stream_conversion_started")

            message_started = False
            content_block_index = 0

            # Track provider metadata to enrich Anthropic events
            response_id: str | None = None
            response_model: str | None = None
            usage_input_tokens: int | None = None
            usage_output_tokens: int | None = None
            usage_cache_read: int | None = None
            usage_cache_write: int | None = None

            # Lightweight heartbeat: emit ping every few deltas to satisfy clients
            delta_count = 0
            PING_INTERVAL = 3

            async for chunk in stream:
                # Handle Response API streaming events
                event_type = chunk.get("type")

                # Capture provider lifecycle/metadata events to enrich outgoing payloads
                if isinstance(event_type, str) and event_type.startswith("response."):
                    # Many providers wrap details under a "response" object
                    resp = chunk.get("response")
                    if isinstance(resp, dict):
                        # Response ID and model if present
                        response_id = resp.get("id") or response_id
                        response_model = resp.get("model") or response_model
                        # Usage details (map to Anthropic-style fields when finishing)
                        usage = resp.get("usage")
                        if isinstance(usage, dict):
                            usage_input_tokens = usage.get(
                                "prompt_tokens", usage_input_tokens
                            )
                            usage_output_tokens = usage.get(
                                "completion_tokens", usage_output_tokens
                            )
                            # Some providers expose cache stats under different keys; be defensive
                            usage_cache_read = (
                                usage.get("cache_read_input_tokens")
                                or usage.get("cache_read_tokens")
                                or usage_cache_read
                            )
                            usage_cache_write = (
                                usage.get("cache_creation_input_tokens")
                                or usage.get("cache_write_tokens")
                                or usage_cache_write
                            )

                    # Don't emit provider lifecycle events to clients
                    # Continue to next chunk to avoid falling through
                    if event_type not in (
                        "response.output_text.delta",
                        "response.done",
                    ):
                        # However, some providers use "response.completed" instead of "response.done".
                        if event_type == "response.completed":
                            # Normalize to done to trigger finish logic below
                            event_type = "response.done"
                        else:
                            continue

                if event_type == "response.output_text.delta":
                    # Text delta from Response API
                    if not message_started:
                        # Enrich message_start with id/model/usage if known
                        msg: dict[str, Any] = {
                            "id": response_id or "msg_generated",
                            "type": "message",
                            "role": "assistant",
                            "content": [],
                        }
                        if response_model:
                            msg["model"] = response_model
                        # Initial usage (input tokens if known)
                        usage_obj: dict[str, Any] | None = None
                        if usage_input_tokens is not None:
                            usage_obj = {
                                "input_tokens": usage_input_tokens,
                                "output_tokens": 0,
                            }

                        yield {
                            "type": "message_start",
                            "message": msg
                            | ({"usage": usage_obj} if usage_obj else {}),
                        }
                        yield {
                            "type": "content_block_start",
                            "index": content_block_index,
                            "content_block": {"type": "text", "text": ""},
                        }
                        message_started = True

                    delta_text = chunk.get("delta", "")
                    if delta_text:
                        yield {
                            "type": "content_block_delta",
                            "index": content_block_index,
                            "delta": {"type": "text_delta", "text": delta_text},
                        }
                        delta_count += 1
                        if delta_count % PING_INTERVAL == 0:
                            # Periodic ping to keep connections lively and match client expectations
                            yield {"type": "ping"}

                elif event_type == "response.done":
                    # End of streaming
                    if message_started:
                        yield {
                            "type": "content_block_stop",
                            "index": content_block_index,
                        }
                        # Include final usage if we captured any
                        msg_delta: dict[str, Any] = {
                            "type": "message_delta",
                            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                        }
                        usage_final: dict[str, Any] = {}
                        if usage_input_tokens is not None:
                            usage_final["input_tokens"] = usage_input_tokens
                        if usage_output_tokens is not None:
                            usage_final["output_tokens"] = usage_output_tokens
                        if usage_cache_write is not None:
                            usage_final["cache_creation_input_tokens"] = (
                                usage_cache_write
                            )
                        if usage_cache_read is not None:
                            usage_final["cache_read_input_tokens"] = usage_cache_read
                        if usage_final:
                            msg_delta["usage"] = usage_final
                        yield msg_delta
                        yield {"type": "message_stop"}

                # For other chunk types, pass through or convert as needed
                elif isinstance(chunk, dict):
                    # Generic streaming chunk conversion
                    yield chunk

            logger.debug("anthropic_response_api_stream_conversion_completed")

        except Exception as e:
            logger.error(
                "anthropic_response_api_stream_conversion_failed",
                error=str(e),
                exc_info=e,
            )
            raise

    def _convert_messages_to_input(
        self, messages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Convert Anthropic messages to Response API input format.

        The formats are nearly identical - both use the same content block structure.
        """
        input_messages = []

        for message in messages:
            # Direct copy - the structure is the same
            input_message = {
                "role": message.get("role", "user"),
                "content": message.get("content", []),
            }

            # Add type field that Response API expects
            input_message["type"] = "message"

            # Add optional id if present
            if "id" in message:
                input_message["id"] = message["id"]

            input_messages.append(input_message)

        return input_messages

    async def adapt_error(self, error_body: dict[str, Any]) -> dict[str, Any]:
        """Convert Response API error format to Anthropic error format.

        Args:
            error_body: Response API error response

        Returns:
            Anthropic-formatted error response
        """
        # Extract error details from Response API format
        response_api_error = error_body.get("error", {})
        error_type = response_api_error.get("type", "internal_server_error")
        error_message = response_api_error.get("message", "An error occurred")

        # Map Response API error types to Anthropic error types
        error_type_mapping = {
            "invalid_request_error": "invalid_request_error",
            "authentication_error": "authentication_error",
            "permission_error": "permission_error",
            "not_found_error": "not_found_error",
            "rate_limit_error": "rate_limit_error",
            "usage_limit_reached": "rate_limit_error",  # Map usage limit to rate limit
            "internal_server_error": "internal_server_error",
            "server_error": "overloaded_error",
        }

        anthropic_error_type = error_type_mapping.get(
            error_type, "internal_server_error"
        )

        # Return Anthropic-formatted error
        return {
            "error": {
                "type": anthropic_error_type,
                "message": error_message,
            }
        }
