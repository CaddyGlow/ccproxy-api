"""Error payload helpers for streaming provider paths."""

from __future__ import annotations

import json
import re
from contextlib import suppress
from typing import Any


_UNSUPPORTED_PARAMETER_RE = re.compile(r"Unsupported parameter:\s*([A-Za-z0-9_.-]+)")


def normalize_openai_error_payload(payload: Any) -> dict[str, Any]:
    """Return a minimal OpenAI-compatible error envelope."""

    data = _decode_error_payload(payload)
    if isinstance(data, dict) and isinstance(data.get("error"), dict):
        error = dict(data["error"])
        message = error.get("message")
        if not isinstance(message, str) or not message:
            error["message"] = "upstream error"
        error.setdefault("type", "invalid_request_error")
        error.setdefault("param", None)
        error.setdefault("code", None)
        return {"error": error}

    message = _extract_error_message(data)
    param = _extract_unsupported_parameter(message)
    return {
        "error": {
            "type": "invalid_request_error",
            "message": message or "upstream error",
            "param": param,
            "code": "unsupported_parameter" if param else None,
        }
    }


def _decode_error_payload(payload: Any) -> Any:
    if isinstance(payload, bytes | bytearray | memoryview):
        text = bytes(payload).decode("utf-8", errors="replace")
        with suppress(json.JSONDecodeError):
            return json.loads(text)
        return text
    return payload


def _extract_error_message(data: Any) -> str:
    if isinstance(data, dict):
        for key in ("detail", "message", "error_description"):
            value = data.get(key)
            if isinstance(value, str) and value:
                return value
        try:
            return json.dumps(data, ensure_ascii=False)
        except (TypeError, ValueError):
            return str(data)
    if isinstance(data, str):
        return data
    return str(data) if data is not None else ""


def _extract_unsupported_parameter(message: str) -> str | None:
    match = _UNSUPPORTED_PARAMETER_RE.search(message)
    if not match:
        return None
    return match.group(1)


__all__ = ["normalize_openai_error_payload"]
