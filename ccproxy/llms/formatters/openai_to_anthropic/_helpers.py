"""Shared helper utilities for OpenAI→Anthropic formatters."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ccproxy.llms.formatters.utils import strict_parse_tool_arguments
from ccproxy.llms.models import openai as openai_models


def _to_mapping(value: Any) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        return value
    if hasattr(value, "model_dump"):
        dumped = value.model_dump()
        if isinstance(dumped, Mapping):
            return dumped
    return None


def _normalize_text_and_images(
    content: Any,
) -> tuple[list[str], list[dict[str, Any]]]:
    text_parts: list[str] = []
    image_blocks: list[dict[str, Any]] = []

    if not isinstance(content, list):
        return text_parts, image_blocks

    for part in content:
        mapping = _to_mapping(part)
        if not mapping:
            continue
        part_type = str(mapping.get("type", "")).lower()
        if part_type in {"text", "input_text"}:
            text_val = mapping.get("text")
            if isinstance(text_val, str) and text_val:
                text_parts.append(text_val)
        elif part_type == "image_url":
            image_info = mapping.get("image_url")
            image_map = _to_mapping(image_info)
            if not image_map:
                continue
            url = image_map.get("url")
            if isinstance(url, str) and url.startswith("data:"):
                try:
                    header, b64data = url.split(",", 1)
                    mediatype = header.split(";")[0].split(":", 1)[1]
                except (ValueError, IndexError):
                    continue
                image_blocks.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": mediatype,
                            "data": b64data,
                        },
                    }
                )

    return text_parts, image_blocks


def _stringify_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts, _ = _normalize_text_and_images(content)
        return " ".join(text_parts)
    return str(content)


def _coerce_system_content(content: Any) -> str | None:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts, _ = _normalize_text_and_images(content)
        return " ".join(text_parts) if text_parts else None
    return None


def _build_user_blocks(content: Any) -> str | list[dict[str, Any]] | None:
    if content is None:
        return None
    if isinstance(content, str):
        return content

    text_parts, image_blocks = _normalize_text_and_images(content)

    if not text_parts and not image_blocks:
        return ""

    blocks: list[dict[str, Any]] = []
    if text_parts:
        blocks.append({"type": "text", "text": " ".join(text_parts)})
    blocks.extend(image_blocks)

    if len(blocks) == 1 and blocks[0]["type"] == "text":
        return str(blocks[0]["text"])
    return blocks


def _build_assistant_blocks(
    content: Any, tool_calls: list[openai_models.ToolCall] | None
) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []

    normalized_content = _build_user_blocks(content)
    if isinstance(normalized_content, str):
        text = normalized_content.strip()
        if text:
            blocks.append({"type": "text", "text": text})
    elif isinstance(normalized_content, list):
        blocks.extend(normalized_content)

    for call in tool_calls or []:
        args_dict = strict_parse_tool_arguments(call.function.arguments)
        blocks.append(
            {
                "type": "tool_use",
                "id": call.id,
                "name": call.function.name,
                "input": args_dict,
            }
        )

    return blocks


__all__ = [
    "_to_mapping",
    "_normalize_text_and_images",
    "_stringify_content",
    "_coerce_system_content",
    "_build_user_blocks",
    "_build_assistant_blocks",
]
