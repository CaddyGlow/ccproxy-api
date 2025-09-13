from __future__ import annotations

import json
from collections.abc import AsyncGenerator, AsyncIterator
from typing import Any

from ccproxy.llms.adapters.base import BaseAPIAdapter
from ccproxy.llms.adapters.mapping import DEFAULT_MAX_TOKENS
from ccproxy.llms.anthropic.models import CreateMessageRequest


class OpenAIChatToAnthropicMessagesAdapter(BaseAPIAdapter):
    """OpenAI Chat → Anthropic Messages request adapter.

    Implemented
    - model: passthrough
    - messages: extracts `system` and maps `user`/`assistant` to Anthropic `messages`
    - max_completion_tokens/max_tokens → `max_tokens` (with default fallback)
    - stream: passthrough
    - thinking: enables Anthropic `thinking` based on `reasoning_effort` (low/medium/high)
      and o1/o3 model families with budgets and safety (ensures `max_tokens > budget`),
      forces `temperature=1.0` when enabled
    - images: supports OpenAI `image_url` with data URL, mapped to Anthropic base64 `ImageBlock`
    - tools: maps OpenAI function tools → Anthropic custom tools
    - tool_choice: maps `none|auto|required|{type:function}` → Anthropic tool choice and
      sets `disable_parallel_tool_use` from `parallel_tool_calls`
    - response_format: injects system guidance for `json_object` and `json_schema`

    TODO
    - Full multimodal request support (audio/prompt images that aren't data URLs)
    - MCP servers, container fields, and advanced options not yet mapped
    - Stop sequences and more nuanced safety/service_tier mapping
    """

    def __init__(self) -> None:
        super().__init__(name="openai_chat_to_anthropic_messages")

    async def adapt_request(self, request: dict[str, Any]) -> dict[str, Any]:
        model = (request.get("model") or "").strip()

        # Determine max tokens
        max_tokens = request.get("max_completion_tokens") or request.get("max_tokens")
        if max_tokens is None:
            max_tokens = DEFAULT_MAX_TOKENS

        # Extract system message if present
        system_value: str | None = None
        out_messages: list[dict[str, Any]] = []
        for msg in request.get("messages", []) or []:
            role = msg.get("role")
            content = msg.get("content")
            if role == "system":
                # Only support simple string content here
                if isinstance(content, str):
                    system_value = content
                else:
                    # Best effort: join text parts if array
                    if isinstance(content, list):
                        texts = [
                            part.get("text")
                            for part in content
                            if isinstance(part, dict)
                        ]
                        system_value = " ".join([t for t in texts if t]) or None
            elif role in ("user", "assistant"):
                # Support multimodal: array content with text and (data URL) images
                if isinstance(content, list):
                    blocks: list[dict[str, Any]] = []
                    text_accum: list[str] = []
                    for part in content:
                        if not isinstance(part, dict):
                            continue
                        ptype = part.get("type")
                        if ptype == "text":
                            t = part.get("text")
                            if isinstance(t, str):
                                text_accum.append(t)
                        elif ptype == "image_url":
                            url = (part.get("image_url") or {}).get("url")
                            if isinstance(url, str) and url.startswith("data:"):
                                # data:[<mediatype>][;base64],<data>
                                try:
                                    header, b64data = url.split(",", 1)
                                    mediatype = header.split(";")[0].split(":", 1)[1]
                                    blocks.append(
                                        {
                                            "type": "image",
                                            "source": {
                                                "type": "base64",
                                                "media_type": mediatype,
                                                "data": b64data,
                                            },
                                        }
                                    )
                                except Exception:
                                    # Skip if malformed
                                    pass
                    # If we collected any structured blocks or multiple text segments, use list form
                    if blocks or len(text_accum) > 1:
                        if text_accum:
                            blocks.insert(
                                0, {"type": "text", "text": " ".join(text_accum)}
                            )
                        out_messages.append({"role": role, "content": blocks})
                    else:
                        # Fallback to simple string
                        content_str = text_accum[0] if text_accum else ""
                        out_messages.append({"role": role, "content": content_str})
                else:
                    out_messages.append({"role": role, "content": content})

        payload: dict[str, Any] = {
            "model": model,
            "messages": out_messages,
            "max_tokens": max_tokens,
        }
        # Inject system guidance for response_format JSON modes
        resp_fmt = request.get("response_format")
        if isinstance(resp_fmt, dict):
            rtype = (resp_fmt.get("type") or "").strip()
            inject: str | None = None
            if rtype == "json_object":
                inject = (
                    "Respond ONLY with a valid JSON object. "
                    "Do not include any additional text, markdown, or explanation."
                )
            elif rtype == "json_schema":
                schema = resp_fmt.get("json_schema")
                try:
                    schema_str = json.dumps(
                        schema, ensure_ascii=False, separators=(",", ":")
                    )
                except Exception:
                    schema_str = str(schema)
                inject = (
                    "Respond ONLY with a JSON object that strictly conforms to this JSON Schema:\n"
                    f"{schema_str}"
                )
            if inject:
                if system_value:
                    system_value = f"{system_value}\n\n{inject}"
                else:
                    system_value = inject

        if system_value is not None:
            payload["system"] = system_value
        if "stream" in request:
            payload["stream"] = bool(request.get("stream"))

        # Tools mapping (OpenAI function tools -> Anthropic custom tools)
        tools_in = request.get("tools") or []
        if isinstance(tools_in, list) and tools_in:
            anth_tools: list[dict[str, Any]] = []
            for t in tools_in:
                if not isinstance(t, dict):
                    continue
                if t.get("type") == "function" and isinstance(t.get("function"), dict):
                    fn = t["function"]
                    anth_tools.append(
                        {
                            "type": "custom",
                            "name": fn.get("name"),
                            "description": fn.get("description"),
                            "input_schema": fn.get("parameters") or {},
                        }
                    )
            if anth_tools:
                payload["tools"] = anth_tools

        # tool_choice mapping
        tool_choice = request.get("tool_choice")
        parallel_tool_calls = request.get("parallel_tool_calls")
        disable_parallel = None
        if isinstance(parallel_tool_calls, bool):
            disable_parallel = not parallel_tool_calls

        if tool_choice is not None:
            anth_choice: dict[str, Any] | None = None
            if isinstance(tool_choice, str):
                if tool_choice == "none":
                    anth_choice = {"type": "none"}
                elif tool_choice == "auto":
                    anth_choice = {"type": "auto"}
                elif tool_choice == "required":
                    anth_choice = {"type": "any"}
            elif isinstance(tool_choice, dict):
                # e.g., {"type":"function", "function":{"name":"my_fn"}}
                if tool_choice.get("type") == "function" and isinstance(
                    tool_choice.get("function"), dict
                ):
                    anth_choice = {
                        "type": "tool",
                        "name": tool_choice["function"].get("name"),
                    }
            if anth_choice is not None:
                if disable_parallel is not None and anth_choice["type"] in {
                    "auto",
                    "any",
                    "tool",
                }:
                    anth_choice["disable_parallel_tool_use"] = disable_parallel
                payload["tool_choice"] = anth_choice

        # Thinking configuration
        thinking_cfg = self._derive_thinking_config(model, request)
        if thinking_cfg is not None:
            payload["thinking"] = thinking_cfg
            # Ensure token budget fits under max_tokens
            budget = thinking_cfg.get("budget_tokens", 0)
            if isinstance(budget, int) and max_tokens <= budget:
                payload["max_tokens"] = budget + 64
            # Temperature constraint when thinking enabled
            payload["temperature"] = 1.0

        # Validate against Anthropic model to ensure shape
        return CreateMessageRequest.model_validate(payload).model_dump()

    def _derive_thinking_config(
        self, model: str, request: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Derive Anthropic thinking config from OpenAI fields and model name.

        Rules:
        - If model matches o1/o3 families, enable thinking by default with model-specific budget
        - Map reasoning_effort: low=1000, medium=5000, high=10000
        - o3*: 10000; o1-mini: 3000; other o1*: 5000
        - If thinking is enabled, return {"type":"enabled","budget_tokens":N}
        - Otherwise return None
        """
        # Explicit reasoning_effort mapping
        effort = (request.get("reasoning_effort") or "").strip().lower()
        effort_budgets = {"low": 1000, "medium": 5000, "high": 10000}

        budget: int | None = None
        if effort in effort_budgets:
            budget = effort_budgets[effort]

        m = model.lower()
        # Model defaults if budget not set by effort
        if budget is None:
            if m.startswith("o3"):
                budget = 10000
            elif m.startswith("o1-mini"):
                budget = 3000
            elif m.startswith("o1"):
                budget = 5000

        if budget is None:
            return None

        return {"type": "enabled", "budget_tokens": budget}

    async def adapt_response(self, response: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    def adapt_stream(
        self, stream: AsyncIterator[dict[str, Any]]
    ) -> AsyncGenerator[dict[str, Any], None]:
        raise NotImplementedError

    async def adapt_error(self, error: dict[str, Any]) -> dict[str, Any]:
        return error
