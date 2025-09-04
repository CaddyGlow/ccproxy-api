"""Composite adapter for Anthropic Messages format using existing conversion infrastructure.

This adapter chains existing, well-tested adapters:
1. ResponseAdapter: Response API ↔ OpenAI Chat Completions
2. OpenAIAdapter: OpenAI Chat Completions ↔ Anthropic Messages

Flow:
- Request: Anthropic Messages → OpenAI Chat Completions → Response API
- Response: Response API → OpenAI Chat Completions → Anthropic Messages
- Stream: Response API streaming → OpenAI streaming → Anthropic streaming
"""

import json
from collections.abc import AsyncGenerator, AsyncIterator
from typing import Any

from ccproxy.adapters.base import APIAdapter
from ccproxy.adapters.openai.adapter import OpenAIAdapter
from ccproxy.adapters.openai.response_adapter import ResponseAdapter
from ccproxy.core.logging import get_plugin_logger


logger = get_plugin_logger()


class CompositeAnthropicAdapter(APIAdapter):
    """Composite adapter that chains ResponseAdapter and OpenAIAdapter for Anthropic Messages support.

    This adapter provides conversion between Anthropic Messages format and Response API format
    by chaining two existing, well-tested adapters:
    - ResponseAdapter: Handles Response API ↔ OpenAI Chat Completions conversion
    - OpenAIAdapter: Handles OpenAI Chat Completions ↔ Anthropic Messages conversion
    """

    def __init__(self) -> None:
        """Initialize the composite adapter with chained adapters."""
        # ResponseAdapter handles Response API ↔ OpenAI Chat Completions
        self.response_adapter = ResponseAdapter()

        # OpenAIAdapter handles OpenAI Chat Completions ↔ Anthropic Messages
        self.openai_adapter = OpenAIAdapter()

    async def adapt_request(self, request_data: dict[str, Any]) -> dict[str, Any]:
        """Convert Anthropic Messages request to Response API format.

        Flow: Anthropic Messages → OpenAI Chat Completions → Response API

        Args:
            request_data: Anthropic Messages API request

        Returns:
            Response API formatted request
        """
        try:
            # Step 1: Convert Anthropic Messages → OpenAI Chat Completions
            # We need to reverse the OpenAI adapter's adapt_request method
            openai_request = self._anthropic_to_openai_request(request_data)

            logger.debug(
                "composite_adapter_step1_completed",
                conversion="anthropic_to_openai",
                openai_keys=list(openai_request.keys()),
                has_messages=bool(openai_request.get("messages")),
                has_tools=bool(openai_request.get("tools")),
            )

            # Step 2: Convert OpenAI Chat Completions → Response API
            response_request = self.response_adapter.chat_to_response_request(
                openai_request
            )
            response_api_request = response_request.model_dump()

            # Ensure Codex-specific model mapping (always use gpt-5 for Codex)
            response_api_request["model"] = "gpt-5"

            # Remove parameters not supported by Codex Response API
            response_api_request.pop("max_tool_calls", None)

            # Clean Codex-specific unsupported fields
            self._clean_codex_request(response_api_request)

            logger.debug(
                "composite_adapter_step2_completed",
                conversion="openai_to_response_api",
                response_keys=list(response_api_request.keys()),
                model=response_api_request.get("model"),
                tools_count=len(response_api_request.get("tools") or []),
            )

            return response_api_request

        except Exception as e:
            logger.error(
                "composite_adapter_request_conversion_failed",
                error=str(e),
                request_keys=list(request_data.keys())
                if isinstance(request_data, dict)
                else "not_dict",
                exc_info=e,
            )
            raise

    async def adapt_response(self, response_data: dict[str, Any]) -> dict[str, Any]:
        """Convert Response API response to Anthropic Messages format.

        Flow: Response API → OpenAI Chat Completions → Anthropic Messages

        Args:
            response_data: Response API response

        Returns:
            Anthropic Messages formatted response
        """
        try:
            # Step 1: Convert Response API → OpenAI Chat Completions
            openai_response = self.response_adapter.response_to_chat_completion(
                response_data
            )
            openai_response_dict = openai_response.model_dump()

            logger.debug(
                "composite_adapter_response_step1_completed",
                conversion="response_api_to_openai",
                openai_keys=list(openai_response_dict.keys()),
                choices_count=len(openai_response_dict.get("choices", [])),
            )

            # Step 2: Convert OpenAI Chat Completions → Anthropic Messages
            # We need to reverse the OpenAI adapter's adapt_response method
            anthropic_response = self._openai_to_anthropic_response(
                openai_response_dict
            )

            logger.debug(
                "composite_adapter_response_step2_completed",
                conversion="openai_to_anthropic",
                anthropic_keys=list(anthropic_response.keys()),
                content_blocks=len(anthropic_response.get("content", [])),
                stop_reason=anthropic_response.get("stop_reason"),
            )

            return anthropic_response

        except Exception as e:
            logger.error(
                "composite_adapter_response_conversion_failed",
                error=str(e),
                response_keys=list(response_data.keys())
                if isinstance(response_data, dict)
                else "not_dict",
                exc_info=e,
            )
            raise

    def adapt_stream(
        self, stream: AsyncIterator[dict[str, Any]]
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Convert Response API streaming to Anthropic Messages streaming format.

        Flow: Response API streaming → OpenAI streaming → Anthropic Messages streaming

        Args:
            stream: Response API streaming events

        Yields:
            Anthropic Messages streaming events
        """
        logger.info("composite_anthropic_adapter_adapt_stream_called")
        return self._adapt_stream_impl(stream)

    async def _adapt_stream_impl(
        self, stream: AsyncIterator[dict[str, Any]]
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Implementation of streaming conversion."""
        try:
            logger.debug("composite_adapter_stream_conversion_started")

            # Step 1: Convert Response API events to OpenAI Chat Completions streaming
            # Use ResponseAdapter's existing streaming conversion
            openai_stream = self.response_adapter.stream_response_to_chat(
                self._dict_stream_to_bytes(stream)
            )

            # Step 2: Convert OpenAI streaming to Anthropic Messages streaming
            # We need to reverse the OpenAI adapter's adapt_stream method
            anthropic_stream = self._openai_stream_to_anthropic(openai_stream)

            event_count = 0
            async for event in anthropic_stream:
                event_count += 1
                logger.debug(
                    "composite_adapter_stream_event",
                    event_number=event_count,
                    event_type=event.get("type"),
                )
                yield event

            logger.debug(
                "composite_adapter_stream_conversion_completed",
                total_events=event_count,
            )

        except Exception as e:
            logger.error(
                "composite_adapter_stream_conversion_failed",
                error=str(e),
                exc_info=e,
            )
            raise

    def _anthropic_to_openai_request(
        self, anthropic_request: dict[str, Any]
    ) -> dict[str, Any]:
        """Convert Anthropic Messages request to OpenAI Chat Completions request.

        This reverses the logic in OpenAIAdapter.adapt_request.
        """
        # Extract basic fields
        openai_request = {
            "model": anthropic_request.get("model", "gpt-4"),
            "messages": self._anthropic_messages_to_openai(
                anthropic_request.get("messages", [])
            ),
        }

        # Handle system prompt
        if "system" in anthropic_request:
            # Add system message at the beginning
            system_message = {"role": "system", "content": anthropic_request["system"]}
            openai_request["messages"].insert(0, system_message)

        # Handle optional parameters
        if "max_tokens" in anthropic_request:
            openai_request["max_tokens"] = anthropic_request["max_tokens"]
        if "temperature" in anthropic_request:
            openai_request["temperature"] = anthropic_request["temperature"]
        if "top_p" in anthropic_request:
            openai_request["top_p"] = anthropic_request["top_p"]
        if "stream" in anthropic_request:
            openai_request["stream"] = anthropic_request["stream"]

        # Handle tools
        if "tools" in anthropic_request:
            openai_request["tools"] = self._anthropic_tools_to_openai(
                anthropic_request["tools"]
            )
        if "tool_choice" in anthropic_request:
            openai_request["tool_choice"] = self._anthropic_tool_choice_to_openai(
                anthropic_request["tool_choice"]
            )

        return openai_request

    def _anthropic_messages_to_openai(
        self, anthropic_messages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Convert Anthropic Messages format messages to OpenAI format."""
        openai_messages = []

        for msg in anthropic_messages:
            role = msg.get("role", "user")
            content = msg.get("content", [])

            # Handle different content formats
            if isinstance(content, str):
                # Simple string content
                openai_messages.append({"role": role, "content": content})
            elif isinstance(content, list):
                # Structured content blocks
                text_content = ""
                tool_calls = []

                for block in content:
                    if block.get("type") == "text":
                        text_content += block.get("text", "")
                    elif block.get("type") == "tool_use":
                        # Convert Anthropic tool_use to OpenAI tool_call
                        # Convert input dict to JSON string for OpenAI format
                        input_data = block.get("input", {})
                        arguments_str = json.dumps(input_data) if input_data else "{}"

                        tool_calls.append(
                            {
                                "id": block.get("id", ""),
                                "type": "function",
                                "function": {
                                    "name": block.get("name", ""),
                                    "arguments": arguments_str,
                                },
                            }
                        )

                # Create OpenAI message
                openai_msg = {"role": role}
                if text_content:
                    openai_msg["content"] = text_content
                if tool_calls:
                    openai_msg["tool_calls"] = tool_calls

                openai_messages.append(openai_msg)
            else:
                # Fallback for other content types
                openai_messages.append({"role": role, "content": str(content)})

        return openai_messages

    def _anthropic_tools_to_openai(
        self, anthropic_tools: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Convert Anthropic tools format to OpenAI tools format."""
        openai_tools = []

        for tool in anthropic_tools:
            if tool.get("type") == "function":
                # Extract function definition
                func = tool.get("function", {})
                openai_tools.append(
                    {
                        "type": "function",
                        "function": {
                            "name": func.get("name", ""),
                            "description": func.get("description"),
                            "parameters": func.get("parameters", {}),
                        },
                    }
                )

        return openai_tools

    def _anthropic_tool_choice_to_openai(
        self, anthropic_tool_choice: str | dict[str, Any]
    ) -> str | dict[str, Any]:
        """Convert Anthropic tool_choice to OpenAI tool_choice."""
        if isinstance(anthropic_tool_choice, str):
            return anthropic_tool_choice  # "auto", "none", "required"
        elif isinstance(anthropic_tool_choice, dict):
            # Handle specific tool choice
            if anthropic_tool_choice.get("type") == "function":
                return {
                    "type": "function",
                    "function": {"name": anthropic_tool_choice.get("name", "")},
                }
        return "auto"  # Fallback

    def _openai_to_anthropic_response(
        self, openai_response: dict[str, Any]
    ) -> dict[str, Any]:
        """Convert OpenAI Chat Completions response to Anthropic Messages response.

        This reverses the logic in OpenAIAdapter.adapt_response.
        """
        choices = openai_response.get("choices", [])
        if not choices:
            return {
                "content": [{"type": "text", "text": ""}],
                "stop_reason": "end_turn",
            }

        choice = choices[0]
        message = choice.get("message", {})
        finish_reason = choice.get("finish_reason", "stop")

        # Convert content blocks
        content = []

        # Add text content
        if message.get("content"):
            content.append({"type": "text", "text": message["content"]})

        # Add tool use blocks
        tool_calls = message.get("tool_calls", [])
        for tool_call in tool_calls:
            function = tool_call.get("function", {})
            # Parse arguments if they're a JSON string
            arguments = function.get("arguments", {})
            if isinstance(arguments, str):
                import json

                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {}

            content.append(
                {
                    "type": "tool_use",
                    "id": tool_call.get("id", ""),
                    "name": function.get("name", ""),
                    "input": arguments,
                }
            )

        # Map finish reason
        stop_reason_map = {
            "stop": "end_turn",
            "length": "max_tokens",
            "tool_calls": "tool_use",
            "content_filter": "stop_sequence",
        }
        stop_reason = stop_reason_map.get(finish_reason, "end_turn")

        # Build Anthropic response
        anthropic_response = {
            "content": content,
            "stop_reason": stop_reason,
        }

        # Add usage if available
        if "usage" in openai_response:
            usage = openai_response["usage"]
            anthropic_response["usage"] = {
                "input_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0),
            }

        # Add model and id if available
        if "model" in openai_response:
            anthropic_response["model"] = openai_response["model"]
        if "id" in openai_response:
            anthropic_response["id"] = openai_response["id"]

        return anthropic_response

    async def _openai_stream_to_anthropic(
        self, openai_stream: AsyncIterator[dict[str, Any]]
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Convert OpenAI streaming to Anthropic Messages streaming format.

        This reverses the logic in OpenAIAdapter._adapt_stream_impl.
        """
        content_block_index = 0
        accumulated_content = ""
        tool_calls_state = {}
        message_started = False

        async for chunk in openai_stream:
            choices = chunk.get("choices", [])
            if not choices:
                continue

            choice = choices[0]
            delta = choice.get("delta", {})
            finish_reason = choice.get("finish_reason")

            # Handle role
            if delta.get("role") == "assistant" and not message_started:
                yield {
                    "type": "message_start",
                    "message": {"role": "assistant", "content": []},
                }
                message_started = True

            # Handle text content
            if "content" in delta and delta["content"]:
                if not accumulated_content:
                    # First text content - start block
                    yield {
                        "type": "content_block_start",
                        "index": content_block_index,
                        "content_block": {"type": "text", "text": ""},
                    }

                accumulated_content += delta["content"]
                yield {
                    "type": "content_block_delta",
                    "index": content_block_index,
                    "delta": {"type": "text_delta", "text": delta["content"]},
                }

            # Handle tool calls
            if "tool_calls" in delta:
                for tool_call_delta in delta["tool_calls"]:
                    tool_index = tool_call_delta.get("index", 0)
                    tool_id = tool_call_delta.get("id")

                    if tool_id and tool_id not in tool_calls_state:
                        # New tool call
                        content_block_index += 1
                        tool_calls_state[tool_id] = {
                            "index": content_block_index,
                            "name": "",
                            "arguments": "",
                        }

                        function = tool_call_delta.get("function", {})
                        if function.get("name"):
                            tool_calls_state[tool_id]["name"] = function["name"]
                            yield {
                                "type": "content_block_start",
                                "index": content_block_index,
                                "content_block": {
                                    "type": "tool_use",
                                    "id": tool_id,
                                    "name": function["name"],
                                },
                            }

                    # Handle function arguments
                    function = tool_call_delta.get("function", {})
                    if function.get("arguments") and tool_id in tool_calls_state:
                        args_delta = function["arguments"]
                        tool_calls_state[tool_id]["arguments"] += args_delta
                        yield {
                            "type": "content_block_delta",
                            "index": tool_calls_state[tool_id]["index"],
                            "delta": {
                                "type": "input_json_delta",
                                "partial_json": args_delta,
                            },
                        }

            # Handle completion
            if finish_reason:
                # End any active content blocks
                if accumulated_content:
                    yield {"type": "content_block_stop", "index": 0}

                for tool_info in tool_calls_state.values():
                    yield {"type": "content_block_stop", "index": tool_info["index"]}

                # Map finish reason
                stop_reason_map = {
                    "stop": "end_turn",
                    "length": "max_tokens",
                    "tool_calls": "tool_use",
                    "content_filter": "stop_sequence",
                }
                stop_reason = stop_reason_map.get(finish_reason, "end_turn")

                yield {"type": "message_delta", "delta": {"stop_reason": stop_reason}}

                # Add usage if available
                if "usage" in chunk:
                    usage = chunk["usage"]
                    yield {
                        "type": "message_delta",
                        "usage": {
                            "input_tokens": usage.get("prompt_tokens", 0),
                            "output_tokens": usage.get("completion_tokens", 0),
                        },
                    }

                yield {"type": "message_stop"}

    def _clean_codex_request(self, codex_request: dict[str, Any]) -> None:
        """Remove fields from request that are not supported by Codex.

        Args:
            codex_request: The Codex request to clean (modified in place)
        """
        # Clean input messages
        if "input" in codex_request and isinstance(codex_request["input"], list):
            for message in codex_request["input"]:
                if isinstance(message, dict) and "content" in message:
                    if isinstance(message["content"], list):
                        for content_block in message["content"]:
                            if isinstance(content_block, dict):
                                # Remove id fields that Codex doesn't support
                                content_block.pop("id", None)
                                # Remove function field from non-tool_call blocks
                                if content_block.get("type") != "tool_call":
                                    content_block.pop("function", None)

    async def _dict_stream_to_bytes(
        self, dict_stream: AsyncIterator[dict[str, Any]]
    ) -> AsyncIterator[bytes]:
        """Convert dict stream to bytes stream for ResponseAdapter compatibility.

        ResponseAdapter expects SSE format with event type in separate line:
        event: response.output_text.delta
        data: {"delta": "Hello"}

        """
        async for event_dict in dict_stream:
            # Extract event type from the dict
            event_type = event_dict.get("type", "")

            # Create event data without the type field
            event_data = {k: v for k, v in event_dict.items() if k != "type"}

            # Convert to SSE format that ResponseAdapter expects
            import json

            json_str = json.dumps(event_data, ensure_ascii=False)

            # Format as proper SSE with separate event and data lines
            sse_event = f"event: {event_type}\ndata: {json_str}\n\n"
            sse_bytes = sse_event.encode("utf-8")
            yield sse_bytes
