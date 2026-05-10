"""Identifier helpers shared across formatter adapters."""

from __future__ import annotations

import json
import re
import uuid
from typing import Any, TypeVar


_SAFE_ID_CHARS = re.compile(r"[^A-Za-z0-9_-]+")
_FUNCTION_ARGUMENT_EVENT_TYPES = {
    "response.function_call_arguments.delta",
    "response.function_call_arguments.done",
}
_Payload = TypeVar("_Payload")


def normalize_suffix(identifier: str) -> str:
    """Return the suffix part of an identifier split on the first underscore."""

    if "_" in identifier:
        return identifier.split("_", 1)[1]
    return identifier


def ensure_identifier(prefix: str, existing: str | None = None) -> tuple[str, str]:
    """Return a stable identifier and suffix for the given prefix.

    If an existing identifier already matches the prefix we reuse its suffix.
    Existing identifiers that begin with ``resp_`` are also understood so both
    ``resp`` and alternate prefixes can build consistent derived identifiers.
    """

    if isinstance(existing, str) and existing.startswith(f"{prefix}_"):
        return existing, normalize_suffix(existing)

    if (
        isinstance(existing, str)
        and existing
        and prefix == "resp"
        and existing.startswith("resp_")
    ):
        return existing, normalize_suffix(existing)

    if (
        isinstance(existing, str)
        and existing
        and existing.startswith("resp_")
        and prefix != "resp"
    ):
        suffix = normalize_suffix(existing)
        return f"{prefix}_{suffix}", suffix

    suffix = uuid.uuid4().hex
    return f"{prefix}_{suffix}", suffix


def _safe_identifier_suffix(identifier: str | None, fallback: str) -> str:
    """Return a compact suffix suitable for generated Responses identifiers."""

    if isinstance(identifier, str) and identifier:
        suffix = normalize_suffix(identifier)
        suffix = _SAFE_ID_CHARS.sub("_", suffix).strip("_")
        if suffix:
            return suffix
    return fallback


def ensure_responses_function_call_identifiers(
    *,
    item_id: str | None,
    call_id: str | None,
    fallback_index: int | str = 0,
) -> tuple[str, str]:
    """Return OpenAI Responses-compatible function-call item and call IDs.

    Responses function-call output items use an ``fc_*`` item id. The model call
    correlation id remains a distinct ``call_*`` value and is reused by
    ``function_call_output`` input items on subsequent turns.
    """

    fallback = str(fallback_index)
    item_suffix = _safe_identifier_suffix(item_id or call_id, fallback)
    call_suffix = _safe_identifier_suffix(call_id or item_id, fallback)

    normalized_item_id = (
        item_id if isinstance(item_id, str) and item_id.startswith("fc_") else None
    )
    if normalized_item_id is None:
        normalized_item_id = f"fc_{item_suffix}"

    normalized_call_id = (
        call_id if isinstance(call_id, str) and call_id.startswith("call_") else None
    )
    if normalized_call_id is None:
        normalized_call_id = f"call_{call_suffix}"

    return normalized_item_id, normalized_call_id


def normalize_responses_function_call_ids(payload: _Payload) -> _Payload:
    """Normalize Responses function-call ids in a JSON-like payload.

    This intentionally skips ``function_call_output`` entries; they are input
    items carrying user code output and should keep their own item identifiers.
    """

    if isinstance(payload, list):
        for item in payload:
            normalize_responses_function_call_ids(item)
        return payload

    if not isinstance(payload, dict):
        return payload

    payload_type = payload.get("type")
    fallback_value = payload.get("output_index", payload.get("index", 0))
    fallback_index = fallback_value if isinstance(fallback_value, int | str) else 0

    if payload_type == "function_call":
        item_id, call_id = ensure_responses_function_call_identifiers(
            item_id=payload.get("id") if isinstance(payload.get("id"), str) else None,
            call_id=payload.get("call_id")
            if isinstance(payload.get("call_id"), str)
            else None,
            fallback_index=fallback_index,
        )
        payload["id"] = item_id
        payload["call_id"] = call_id

    elif payload_type in _FUNCTION_ARGUMENT_EVENT_TYPES:
        item_id, call_id = ensure_responses_function_call_identifiers(
            item_id=payload.get("item_id")
            if isinstance(payload.get("item_id"), str)
            else None,
            call_id=payload.get("call_id")
            if isinstance(payload.get("call_id"), str)
            else None,
            fallback_index=fallback_index,
        )
        payload["item_id"] = item_id
        if isinstance(payload.get("call_id"), str):
            payload["call_id"] = call_id

    for value in payload.values():
        normalize_responses_function_call_ids(value)

    return payload


def normalize_responses_sse_event_bytes(event_data: bytes) -> bytes:
    """Normalize a complete SSE event carrying a Responses JSON payload."""

    try:
        text = event_data.decode("utf-8")
    except UnicodeDecodeError:
        return event_data

    lines = text.splitlines()
    passthrough_lines: list[str] = []
    data_lines: list[str] = []
    for line in lines:
        if line.startswith("data:"):
            data_value = line[5:]
            if data_value.startswith(" "):
                data_value = data_value[1:]
            data_lines.append(data_value)
        elif line:
            passthrough_lines.append(line)

    if not data_lines:
        return event_data

    data_payload = "\n".join(data_lines)
    if data_payload.strip() == "[DONE]":
        return event_data

    try:
        parsed = json.loads(data_payload)
    except json.JSONDecodeError:
        return event_data

    normalized = normalize_responses_function_call_ids(parsed)
    compact = json.dumps(normalized, ensure_ascii=False, separators=(",", ":"))
    normalized_lines = [*passthrough_lines, f"data: {compact}", ""]
    return ("\n".join(normalized_lines) + "\n").encode("utf-8")


__all__ = [
    "ensure_identifier",
    "ensure_responses_function_call_identifiers",
    "normalize_responses_function_call_ids",
    "normalize_responses_sse_event_bytes",
    "normalize_suffix",
]
