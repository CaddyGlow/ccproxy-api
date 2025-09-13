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
        """Convert between Anthropic Messages and OpenAI Response API formats.

        Args:
            request: Anthropic Messages API request

        Returns:
            OpenAI Response API formatted request
        """
        try:
            # Dual-direction support: detect input shape by keys
            # If request has 'input' (Response API), convert to Anthropic
            if "input" in request and isinstance(request["input"], list):
                anthropic_request: dict[str, Any] = {}

                # Model
                if "model" in request:
                    anthropic_request["model"] = request["model"]

                # input -> messages (convert content blocks)
                anthropic_request["messages"] = self._convert_input_to_messages(
                    request["input"]
                )

                # instructions -> system
                if "instructions" in request:
                    anthropic_request["system"] = request["instructions"]

                # Tools passthrough
                for field in ["tools", "tool_choice", "parallel_tool_calls"]:
                    if field in request:
                        anthropic_request[field] = request[field]

                # max tokens mapping with sensible default (Anthropic requires it)
                max_tokens_val = (
                    request.get("max_tokens")
                    or request.get("max_output_tokens")
                    or request.get("max_completion_tokens")
                    or 4096
                )
                anthropic_request["max_tokens"] = max_tokens_val

                # stream flag passthrough
                if "stream" in request:
                    anthropic_request["stream"] = request["stream"]

                logger.debug(
                    "response_api_to_anthropic_request_conversion",
                    original_keys=list(request.keys()),
                    converted_keys=list(anthropic_request.keys()),
                    message_count=len(anthropic_request.get("messages", [])),
                )

                return anthropic_request

            # Otherwise treat request as Anthropic -> Response API
            response_api_request: dict[str, Any] = {}

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

        # Heuristic: if response looks like Anthropic (message with content blocks),
        # convert to OpenAI Response API shape instead of the legacy reverse path.
        try:
            if (
                isinstance(response, dict)
                and isinstance(response.get("content"), list)
                and (response.get("type") in (None, "message"))
            ):
                content_blocks = response.get("content", []) or []
                role = response.get("role", "assistant")
                model = response.get("model")
                resp_id = response.get("id")
                stop_reason = response.get("stop_reason")

                output_item = {
                    "type": "message",
                    "role": role,
                    "content": content_blocks,
                }
                result: dict[str, Any] = {
                    "id": resp_id or "resp_generated",
                    "model": model,
                    "output": [output_item],
                }
                if stop_reason is not None:
                    result["stop_reason"] = stop_reason

                usage = response.get("usage")
                if isinstance(usage, dict):
                    result["usage"] = {
                        "prompt_tokens": usage.get("input_tokens", 0),
                        "completion_tokens": usage.get("output_tokens", 0),
                    }

                logger.debug(
                    "anthropic_to_response_api_conversion",
                    converted_keys=list(result.keys()),
                    has_output=bool(result.get("output")),
                )
                return result
        except Exception:
            # Fall through to legacy path
            pass

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

            # Track Anthropic→Response API direction state
            anth_id: str | None = None
            anth_model: str | None = None
            done_emitted = False

            async for chunk in stream:
                # Handle Response API streaming events
                event_type = chunk.get("type")
                logger.trace(
                    "stream_chunk_received",
                    event_type=event_type,
                    keys=list(chunk.keys()),
                    category="streaming",
                )

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

                    msg: dict[str, Any] = {
                        "id": response_id or "msg_generated",
                        "type": "message",
                        "role": "assistant",
                        "content": [],
                    }
                    # Translate lifecycle events to early client-visible events
                    if event_type in ("response.created", "response.in_progress"):
                        if not message_started:
                            if response_model:
                                msg["model"] = response_model
                            initial_usage: dict[str, Any] | None = None
                            if usage_input_tokens is not None:
                                initial_usage = {
                                    "input_tokens": usage_input_tokens,
                                    "output_tokens": 0,
                                }
                            logger.trace(
                                "emit_message_start",
                                id=msg.get("id"),
                                model=msg.get("model"),
                                category="streaming",
                            )
                            yield {
                                "type": "message_start",
                                "message": msg
                                | ({"usage": initial_usage} if initial_usage else {}),
                            }
                            yield {
                                "type": "content_block_start",
                                "index": content_block_index,
                                "content_block": {"type": "text", "text": ""},
                            }
                            message_started = True
                        else:
                            # Keep-alive ping while waiting for first delta
                            logger.trace(
                                "emit_ping_waiting_for_delta", category="streaming"
                            )
                            yield {"type": "ping"}
                        continue

                    # Normalize alternate completion event naming
                    if event_type == "response.completed":
                        event_type = "response.done"
                    # Skip other provider lifecycle events
                    if event_type not in (
                        "response.output_text.delta",
                        "response.done",
                    ):
                        continue

                if event_type == "response.output_text.delta":
                    # Text delta from Response API
                    if not message_started:
                        # Enrich message_start with id/model/usage if known
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
                        logger.trace(
                            "emit_text_delta",
                            size=len(delta_text or ""),
                            category="streaming",
                        )
                        yield {
                            "type": "content_block_delta",
                            "index": content_block_index,
                            "delta": {"type": "text_delta", "text": delta_text},
                        }
                        delta_count += 1
                        if delta_count % PING_INTERVAL == 0:
                            # Periodic ping to keep connections lively and match client expectations
                            logger.trace("emit_ping_periodic", category="streaming")
                            yield {"type": "ping"}

                elif event_type == "response.done":
                    # End of streaming
                    if message_started:
                        logger.trace("emit_message_stop_sequence", category="streaming")
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
                elif event_type in {
                    "message_start",
                    "content_block_start",
                    "content_block_delta",
                    "content_block_stop",
                    "message_delta",
                    "message_stop",
                }:
                    # Convert Anthropic events to Response API events
                    if event_type == "message_start":
                        message_started = True
                        msg = (
                            chunk.get("message", {})
                            if isinstance(chunk.get("message"), dict)
                            else {}
                        )
                        anth_id = msg.get("id") or anth_id or "resp_generated"
                        anth_model = msg.get("model") or anth_model
                        resp_obj: dict[str, Any] = {"id": anth_id}
                        if anth_model:
                            resp_obj["model"] = anth_model
                        yield {"type": "response.created", "response": resp_obj}
                        done_emitted = False
                        continue

                    if event_type == "content_block_delta":
                        delta = (
                            chunk.get("delta", {})
                            if isinstance(chunk.get("delta"), dict)
                            else {}
                        )
                        if delta.get("type") == "text_delta" and "text" in delta:
                            yield {
                                "type": "response.output_text.delta",
                                "delta": delta.get("text", ""),
                            }
                            delta_count += 1
                            if delta_count % PING_INTERVAL == 0:
                                yield {
                                    "type": "response.in_progress",
                                    "response": {"id": anth_id or "resp_generated"},
                                }
                        continue

                    if event_type == "message_delta":
                        d = (
                            chunk.get("delta", {})
                            if isinstance(chunk.get("delta"), dict)
                            else {}
                        )
                        usage = d.get("usage") if isinstance(d, dict) else None
                        resp_done: dict[str, Any] = {"id": anth_id or "resp_generated"}
                        if anth_model:
                            resp_done["model"] = anth_model
                        if isinstance(usage, dict):
                            resp_done["usage"] = {
                                "prompt_tokens": usage.get("input_tokens", 0),
                                "completion_tokens": usage.get("output_tokens", 0),
                            }
                        yield {"type": "response.done", "response": resp_done}
                        done_emitted = True
                        continue

                    if event_type == "message_stop":
                        if not done_emitted:
                            yield {
                                "type": "response.done",
                                "response": {
                                    "id": anth_id or "resp_generated",
                                    **({"model": anth_model} if anth_model else {}),
                                },
                            }
                            done_emitted = True
                        continue

                    # content_block_start/stop have no direct mapping; skip
                    continue

                elif isinstance(chunk, dict):
                    # Generic passthrough for unknown chunks
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
            # Direct copy with content block type normalization for Response API
            input_message = {
                "role": message.get("role", "user"),
                "content": message.get("content", []),
            }

            # Add type field that Response API expects
            input_message["type"] = "message"

            # Normalize content block types to Response API expectations
            content = input_message.get("content") or []
            normalized_content: list[dict[str, Any]] = []
            for block in content:
                if not isinstance(block, dict):
                    continue
                btype = block.get("type")
                if btype == "text":
                    normalized_content.append(
                        {"type": "input_text", "text": block.get("text", "")}
                    )
                else:
                    # Preserve unknown blocks (tool_use, etc.) as-is
                    normalized_content.append(block)
            input_message["content"] = normalized_content

            # Do NOT carry over extraneous fields like 'id' to keep target schema clean

            input_messages.append(input_message)

        return input_messages

    def _convert_input_to_messages(
        self, input_messages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Convert Response API input list to Anthropic messages.

        Maps content types:
        - {type: "input_text", text: "..."} -> {type: "text", text: "..."}
        - {type: "text", text: "..."} -> {type: "text", text: "..."}
        Leaves unknown types as-is to avoid data loss.
        """
        messages: list[dict[str, Any]] = []

        for item in input_messages:
            role = item.get("role", "user")
            content_list = item.get("content", [])
            converted_content: list[dict[str, Any]] = []
            for block in content_list or []:
                btype = block.get("type")
                if btype == "input_text" or btype == "text":
                    converted_content.append(
                        {"type": "text", "text": block.get("text", "")}
                    )
                # Drop unknown block types to avoid extra fields in target schema

            # Anthropic request messages should only include role and content
            msg = {"role": role, "content": converted_content}
            messages.append(msg)

        return messages

    async def adapt_error(self, error: dict[str, Any]) -> dict[str, Any]:
        """Convert Response API error format to Anthropic error format.

        Args:
            error_body: Response API error response

        Returns:
            Anthropic-formatted error response
        """
        # Extract error details from Response API format
        response_api_error = error.get("error", {})
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
