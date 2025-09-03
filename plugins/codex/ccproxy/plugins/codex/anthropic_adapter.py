"""Anthropic Messages format adapter for Codex Response API conversions."""

import json
from collections.abc import AsyncIterator
from typing import Any, AsyncGenerator

from ccproxy.adapters.openai.response_adapter import ResponseAdapter
from ccproxy.adapters.openai.streaming import AnthropicSSEFormatter
from ccproxy.core.logging import get_plugin_logger

logger = get_plugin_logger()


class AnthropicMessagesAdapter:
    """Format adapter for Anthropic Messages API to Codex Response API conversions.

    Handles conversion between Anthropic Messages format and Codex Response API format,
    with full function calling support.
    """

    def __init__(self) -> None:
        """Initialize the Anthropic Messages adapter."""
        self._response_adapter = ResponseAdapter()

    async def adapt_request(self, request_data: dict[str, Any]) -> dict[str, Any]:
        """Convert Anthropic Messages request to Response API format.

        Args:
            request_data: Anthropic Messages API request

        Returns:
            Codex Response API formatted request
        """
        if "messages" in request_data:
            has_tools = bool(request_data.get("tools"))
            has_tool_choice = "tool_choice" in request_data
            logger.debug(
                "converting_anthropic_messages_to_response_api",
                has_tools=has_tools,
                has_tool_choice=has_tool_choice,
                max_tokens=request_data.get("max_tokens"),
            )
            
            # Use ResponseAdapter to convert Anthropic Messages → Response API
            response_request = self._response_adapter.chat_to_response_request(
                request_data
            )
            codex_request = response_request.model_dump()

            # Ensure Codex-specific defaults and filter unsupported parameters
            codex_request["model"] = "gpt-5"  # Always use gpt-5 for Codex
            
            # Remove parameters not supported by Codex
            codex_request.pop("max_tool_calls", None)
            
            # Remove unsupported fields from content blocks
            self._clean_codex_request(codex_request)

            logger.debug(
                "anthropic_codex_request_conversion",
                original_keys=list(request_data.keys()),
                converted_keys=list(codex_request.keys()),
                tools_count=len(codex_request.get("tools") or []),
                tool_choice=codex_request.get("tool_choice", "auto"),
            )
            return codex_request

        # Native Response API format - passthrough
        logger.debug(
            "anthropic_request_passthrough", request_keys=list(request_data.keys())
        )
        return request_data

    async def adapt_response(self, response_data: dict[str, Any]) -> dict[str, Any]:
        """Convert Response API response to Anthropic Messages format.

        Args:
            response_data: Codex Response API response

        Returns:
            Anthropic Messages formatted response
        """
        # Check if this is a Response API format response
        if self._is_response_api_format(response_data):
            has_tool_calls = self._has_tool_calls_in_response(response_data)
            logger.debug(
                "converting_response_api_to_anthropic_messages",
                has_tool_calls=has_tool_calls,
            )
            
            # First convert to OpenAI Chat Completions format
            chat_response = self._response_adapter.response_to_chat_completion(
                response_data
            )
            
            # Then convert to Anthropic Messages format
            anthropic_response = self._convert_to_anthropic_format(
                chat_response.model_dump()
            )

            # Log conversion details
            content = anthropic_response.get("content", [])
            tool_uses = [block for block in content if block.get("type") == "tool_use"]
            logger.debug(
                "anthropic_response_conversion_completed",
                stop_reason=anthropic_response.get("stop_reason"),
                tool_uses_count=len(tool_uses),
                content_blocks=len(content),
            )

            return anthropic_response

        return response_data

    def adapt_stream(
        self, stream: AsyncIterator[dict[str, Any]]
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Convert individual Response API events to Anthropic Messages format.

        Args:
            stream: Response API streaming events

        Yields:
            Anthropic Messages streaming events
        """
        # Convert Response API events directly to Anthropic format
        return self._convert_response_stream_to_anthropic(stream)

    async def _convert_stream_to_anthropic(
        self, openai_stream: AsyncGenerator[dict[str, Any], None]
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Convert OpenAI streaming format to Anthropic streaming format.
        
        Args:
            openai_stream: OpenAI Chat Completions streaming events
            
        Yields:
            Anthropic Messages streaming events
        """
        content_block_index = 0
        accumulated_content = ""
        tool_calls_state = {}
        
        async for chunk in openai_stream:
            choices = chunk.get("choices", [])
            if not choices:
                continue
                
            choice = choices[0]
            delta = choice.get("delta", {})
            finish_reason = choice.get("finish_reason")
            
            # Handle role
            if delta.get("role") == "assistant":
                yield {"type": "message_start", "message": {"role": "assistant", "content": []}}
            
            # Handle text content
            if "content" in delta and delta["content"]:
                if not accumulated_content:
                    # First text content - start block
                    yield {
                        "type": "content_block_start",
                        "index": content_block_index,
                        "content_block": {"type": "text", "text": ""}
                    }
                
                accumulated_content += delta["content"]
                yield {
                    "type": "content_block_delta", 
                    "index": content_block_index,
                    "delta": {"type": "text", "text": delta["content"]}
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
                            "arguments": ""
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
                                    "name": function["name"]
                                }
                            }
                    
                    # Handle function arguments
                    function = tool_call_delta.get("function", {})
                    if function.get("arguments") and tool_id in tool_calls_state:
                        args_delta = function["arguments"]
                        tool_calls_state[tool_id]["arguments"] += args_delta
                        yield {
                            "type": "content_block_delta",
                            "index": tool_calls_state[tool_id]["index"],
                            "delta": {"type": "input_json", "partial_json": args_delta}
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
                    "content_filter": "stop_sequence"
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
                            "output_tokens": usage.get("completion_tokens", 0)
                        }
                    }
                
                yield {"type": "message_stop"}

    def _is_response_api_format(self, response_data: dict[str, Any]) -> bool:
        """Check if response is in Response API format (used by Codex)."""
        # Response API responses have 'output' field or are wrapped in 'response'
        return "output" in response_data or "response" in response_data

    def _has_tool_calls_in_response(self, response_data: dict[str, Any]) -> bool:
        """Check if response contains tool calls."""
        # Check direct response format
        output = response_data.get("output", [])
        if output:
            for output_item in output:
                if output_item.get("type") == "message":
                    content = output_item.get("content", [])
                    for block in content:
                        if block.get("type") == "tool_call":
                            return True

        # Check wrapped response format
        response = response_data.get("response", {})
        if response:
            return self._has_tool_calls_in_response(response)

        return False

    def _convert_to_anthropic_format(self, openai_response: dict[str, Any]) -> dict[str, Any]:
        """Convert OpenAI Chat Completions response to Anthropic Messages format.

        Args:
            openai_response: OpenAI Chat Completions response

        Returns:
            Anthropic Messages format response
        """
        # Extract the main choice content
        choices = openai_response.get("choices", [])
        if not choices:
            return {
                "content": [{"type": "text", "text": ""}],
                "stop_reason": "end_turn",
                "usage": openai_response.get("usage", {})
            }

        choice = choices[0]
        message = choice.get("message", {})
        finish_reason = choice.get("finish_reason", "stop")

        # Convert content blocks
        content = []

        # Add text content if present
        if message.get("content"):
            content.append({"type": "text", "text": message["content"]})

        # Add tool use blocks if present
        tool_calls = message.get("tool_calls", [])
        for tool_call in tool_calls:
            if hasattr(tool_call, 'model_dump'):
                # Handle Pydantic model
                tool_dict = tool_call.model_dump()
            else:
                # Handle dict
                tool_dict = tool_call

            function = tool_dict.get("function", {})

            # Parse arguments JSON
            arguments_str = function.get("arguments", "{}")
            try:
                if isinstance(arguments_str, str):
                    arguments = json.loads(arguments_str)
                else:
                    arguments = arguments_str
            except json.JSONDecodeError:
                logger.warning(
                    "tool_arguments_parse_failed",
                    arguments=arguments_str[:200] + "..."
                    if len(str(arguments_str)) > 200
                    else str(arguments_str),
                    operation="convert_to_anthropic_format",
                )
                arguments = {}

            content.append({
                "type": "tool_use",
                "id": tool_dict.get("id", ""),
                "name": function.get("name", ""),
                "input": arguments
            })

        # Convert finish reason
        stop_reason_map = {
            "stop": "end_turn",
            "length": "max_tokens",
            "tool_calls": "tool_use",
            "content_filter": "stop_sequence"
        }
        stop_reason = stop_reason_map.get(finish_reason, "end_turn")

        # Build Anthropic response
        anthropic_response = {
            "content": content,
            "stop_reason": stop_reason
        }

        # Add usage if available
        if "usage" in openai_response:
            usage = openai_response["usage"]
            anthropic_response["usage"] = {
                "input_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0)
            }

        # Add model and id if available
        if "model" in openai_response:
            anthropic_response["model"] = openai_response["model"]
        if "id" in openai_response:
            anthropic_response["id"] = openai_response["id"]

        return anthropic_response

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

    async def _convert_response_stream_to_anthropic(
        self, stream: AsyncIterator[dict[str, Any]]
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Convert Response API streaming events directly to Anthropic Messages format.
        
        Args:
            stream: Response API streaming events
            
        Yields:
            Anthropic Messages streaming events
        """
        message_started = False
        content_blocks = []
        current_content_index = 0
        
        async for event in stream:
            event_type = event.get("type", "")
            
            # Handle response creation
            if event_type == "response.created":
                if not message_started:
                    yield {"type": "message_start", "message": {"role": "assistant", "content": []}}
                    message_started = True
            
            # Handle text output deltas
            elif event_type == "response.output_text.delta":
                delta_text = event.get("delta", "")
                if delta_text:
                    # Start content block if first text
                    if not content_blocks or content_blocks[-1].get("type") != "text":
                        yield {
                            "type": "content_block_start",
                            "index": current_content_index,
                            "content_block": {"type": "text", "text": ""}
                        }
                        content_blocks.append({"type": "text", "index": current_content_index})
                        current_content_index += 1
                    
                    # Send text delta
                    text_block_index = next(
                        (block["index"] for block in content_blocks if block["type"] == "text"),
                        0
                    )
                    yield {
                        "type": "content_block_delta",
                        "index": text_block_index,
                        "delta": {"type": "text_delta", "text": delta_text}
                    }
            
            # Handle completion
            elif event_type == "response.completed":
                # End all content blocks
                for block in content_blocks:
                    yield {"type": "content_block_stop", "index": block["index"]}
                
                # Send completion message
                response = event.get("response", {})
                yield {"type": "message_delta", "delta": {"stop_reason": "end_turn"}}
                
                # Add usage if available
                usage = response.get("usage")
                if usage:
                    yield {
                        "type": "message_delta",
                        "usage": {
                            "input_tokens": usage.get("input_tokens", 0),
                            "output_tokens": usage.get("output_tokens", 0)
                        }
                    }
                
                yield {"type": "message_stop"}
            
            # Handle errors
            elif event_type == "error":
                yield {"type": "error", "error": event.get("error", {})}
        
        # Ensure message_stop is sent if stream ends without completion
        if message_started:
            for block in content_blocks:
                yield {"type": "content_block_stop", "index": block["index"]}
            yield {"type": "message_stop"}