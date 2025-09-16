"""Response API → OpenAI adapter with streaming support.

Converts provider "Response API" format (Codex/ChatGPT web backend style)
to OpenAI Chat Completions format for both non‑streaming and streaming flows.
"""

from __future__ import annotations

import time
from collections.abc import AsyncGenerator, AsyncIterator
from typing import Any

from ccproxy.adapters.base import APIAdapter
from ccproxy.core.logging import get_logger


logger = get_logger(__name__)


class ResponseAPIToOpenAIAdapter(APIAdapter):
    """Adapter that converts Response API payloads to OpenAI format."""

    async def adapt_request(self, request: dict[str, Any]) -> dict[str, Any]:
        """Response API → OpenAI request conversion (not commonly used).

        Note: Typical pipelines use a dedicated OpenAI→Response API adapter for requests.
        This method provides a best‑effort mapping when invoked.
        """
        try:
            model = request.get("model")
            messages: list[dict[str, Any]] = []

            # Response API uses input: [ {type: 'message', role, content:[...]} ]
            for item in request.get("input", []) or []:
                if not isinstance(item, dict):
                    continue
                role = item.get("role", "user")
                # Collapse content blocks into plain text for OpenAI
                text_parts: list[str] = []
                for block in item.get("content", []) or []:
                    if isinstance(block, dict):
                        btype = block.get("type")
                        if btype in {
                            "text",
                            "input_text",
                            "output_text",
                            "summary_text",
                        }:
                            if "text" in block:
                                text_parts.append(str(block.get("text", "")))
                content_text = "".join(text_parts)
                messages.append({"role": role, "content": content_text})

            openai_req: dict[str, Any] = {"model": model, "messages": messages}

            # Pass through a few common fields when present
            for k in ("temperature", "top_p", "n", "stream"):
                if k in request:
                    openai_req[k] = request[k]

            return openai_req
        except Exception as e:
            logger.error(
                "response_to_openai_request_conversion_failed", error=str(e), exc_info=e
            )
            raise

    async def adapt_response(self, response: dict[str, Any]) -> dict[str, Any]:
        """Convert Response API response to OpenAI Chat Completion."""
        try:
            created = int(time.time())
            model = response.get("model") or response.get("response", {}).get("model")

            # Extract text content from Response API structure
            text_content = ""
            actual = (
                response.get("response")
                if isinstance(response.get("response"), dict)
                else response
            )
            output = actual.get("output") if isinstance(actual, dict) else None
            if isinstance(output, list):
                parts: list[str] = []
                for item in output:
                    if isinstance(item, dict):
                        if item.get("type") == "message":
                            for b in item.get("content", []) or []:
                                if (
                                    isinstance(b, dict)
                                    and b.get("type")
                                    in {"text", "output_text", "summary_text"}
                                    and "text" in b
                                ):
                                    parts.append(str(b.get("text", "")))
                        elif item.get("type") in {
                            "text",
                            "output_text",
                            "summary_text",
                        }:
                            if "text" in item:
                                parts.append(str(item.get("text", "")))
                text_content = "".join(parts)

            usage = actual.get("usage") if isinstance(actual, dict) else None
            prompt_tokens = (usage or {}).get("prompt_tokens", 0)
            completion_tokens = (usage or {}).get("completion_tokens", 0)

            openai_resp: dict[str, Any] = {
                "id": response.get("id", f"chatcmpl-{created}"),
                "object": "chat.completion",
                "created": created,
                "model": model or "gpt-4o-2024-11-20",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": text_content or ""},
                        "finish_reason": "stop",
                        "logprobs": None,
                    }
                ],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
            }
            return openai_resp
        except Exception as e:
            logger.error(
                "response_to_openai_responses_conversion_failed",
                error=str(e),
                exc_info=e,
            )
            raise

    async def adapt_error(self, error: dict[str, Any]) -> dict[str, Any]:
        """Map Response API error to OpenAI error format."""
        err = error.get("error", {}) if isinstance(error, dict) else {}
        return {
            "error": {
                "message": err.get("message", "An error occurred"),
                "type": err.get("type", "internal_server_error"),
                "param": err.get("param"),
                "code": err.get("code"),
            }
        }

    def adapt_stream(
        self, stream: AsyncIterator[dict[str, Any]]
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Convert Response API streaming events to OpenAI streaming chunks.

        Input events of interest:
        - response.created { response: { id, model } }
        - response.output_text.delta { delta: "..." }
        - response.done { response: { usage? } }
        """

        return self._adapt_stream_impl(stream)

    async def _adapt_stream_impl(
        self, stream: AsyncIterator[dict[str, Any]]
    ) -> AsyncGenerator[dict[str, Any], None]:
        logger.debug("response_to_openai_stream_start")
        created = int(time.time())
        message_id: str | None = None
        model: str | None = None
        # Optional usage to include in final chunk if present
        usage_prompt: int | None = None
        usage_completion: int | None = None

        try:
            async for chunk in stream:
                if not isinstance(chunk, dict):
                    continue
                et = chunk.get("type")
                if et == "response.created":
                    resp = chunk.get("response", {})
                    message_id = resp.get("id", message_id)
                    model = resp.get("model", model)
                    # Emit initial role delta chunk to match OpenAI semantics
                    yield {
                        "id": message_id or f"chatcmpl-{created}",
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model or "gpt-4o-2024-11-20",
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"role": "assistant"},
                                "logprobs": None,
                                "finish_reason": None,
                            }
                        ],
                    }
                elif et == "response.output_text.delta":
                    delta_text = chunk.get("delta", "")
                    if not isinstance(delta_text, str) or len(delta_text) == 0:
                        continue
                    yield {
                        "id": message_id or f"chatcmpl-{created}",
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model or "gpt-4o-2024-11-20",
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": delta_text},
                                "logprobs": None,
                                "finish_reason": None,
                            }
                        ],
                    }
                elif et == "response.done":
                    resp = chunk.get("response", {})
                    usage = resp.get("usage") if isinstance(resp, dict) else None
                    if isinstance(usage, dict):
                        usage_prompt = usage.get("prompt_tokens", usage_prompt)
                        usage_completion = usage.get(
                            "completion_tokens", usage_completion
                        )
                    # Emit final stop chunk (OpenAI convention)
                    final: dict[str, Any] = {
                        "id": message_id or f"chatcmpl-{created}",
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model or "gpt-4o-2024-11-20",
                        "choices": [
                            {
                                "index": 0,
                                "delta": {},
                                "logprobs": None,
                                "finish_reason": "stop",
                            }
                        ],
                    }
                    if usage_prompt is not None or usage_completion is not None:
                        final["usage"] = {
                            "prompt_tokens": usage_prompt or 0,
                            "completion_tokens": usage_completion or 0,
                            "total_tokens": (usage_prompt or 0)
                            + (usage_completion or 0),
                        }
                    yield final
                # Ignore other events

            logger.debug("response_to_openai_stream_complete")
        except Exception as e:
            logger.error("response_to_openai_stream_failed", error=str(e), exc_info=e)
            raise
