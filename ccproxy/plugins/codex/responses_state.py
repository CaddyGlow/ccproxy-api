"""Local OpenAI Responses continuation state for the Codex backend."""

from __future__ import annotations

import copy
import hashlib
import json
import time
from collections import OrderedDict
from dataclasses import dataclass
from threading import RLock
from typing import Any

from ccproxy.llms.formatters.common import normalize_responses_function_call_ids


_FAILED_STATUSES = {"failed", "incomplete", "cancelled", "canceled"}


class ResponsesStateNotFoundError(ValueError):
    """Raised when a local ``previous_response_id`` cannot be resolved."""

    def __init__(self, response_id: str) -> None:
        self.response_id = response_id
        super().__init__(f"Unknown previous_response_id: {response_id}")

    def to_openai_error(self) -> dict[str, Any]:
        return {
            "error": {
                "type": "invalid_request_error",
                "message": str(self),
                "param": "previous_response_id",
                "code": "previous_response_not_found",
            }
        }


@dataclass
class ResponsesStateRecord:
    scope: str
    response_id: str
    context_items: list[Any]
    expires_at: float


class CodexResponsesStateStore:
    """Bounded per-client state used to emulate ``previous_response_id`` locally."""

    def __init__(self, *, max_entries: int = 1024, ttl_seconds: int = 3600) -> None:
        self.max_entries = _positive_int(max_entries, default=1024)
        self.ttl_seconds = _positive_int(ttl_seconds, default=3600)
        self._records: OrderedDict[tuple[str, str], ResponsesStateRecord] = (
            OrderedDict()
        )
        self._lock = RLock()

    def prepare_payload(
        self,
        payload: dict[str, Any],
        *,
        headers: dict[str, str],
    ) -> tuple[dict[str, Any], str, str | None]:
        """Return a provider payload with any local continuation expanded."""

        scope = self.scope_for_headers(headers)
        previous_response_id = payload.get("previous_response_id")
        prepared = copy.deepcopy(payload)

        if previous_response_id is None:
            return prepared, scope, None

        if not isinstance(previous_response_id, str) or not previous_response_id:
            return prepared, scope, None

        record = self.get(scope, previous_response_id)
        if record is None:
            raise ResponsesStateNotFoundError(previous_response_id)

        current_input = _normalize_input_items(prepared.get("input", []))
        prepared["input"] = copy.deepcopy(record.context_items) + current_input
        prepared.pop("previous_response_id", None)
        return prepared, scope, previous_response_id

    def store_response(
        self,
        *,
        scope: str,
        request_payload: dict[str, Any],
        response_payload: dict[str, Any],
    ) -> bool:
        """Store a completed or tool-pending Responses payload for continuation."""

        if not isinstance(response_payload, dict):
            return False

        response_id = response_payload.get("id")
        if not isinstance(response_id, str) or not response_id:
            return False

        status = response_payload.get("status")
        if isinstance(status, str) and status.lower() in _FAILED_STATUSES:
            return False

        output_items = response_payload.get("output")
        if not isinstance(output_items, list):
            output_items = []

        input_items = _normalize_input_items(request_payload.get("input", []))
        context_items = input_items + _normalize_output_items(output_items)
        if not context_items:
            return False

        expires_at = time.monotonic() + self.ttl_seconds
        record = ResponsesStateRecord(
            scope=scope,
            response_id=response_id,
            context_items=context_items,
            expires_at=expires_at,
        )
        key = (scope, response_id)
        with self._lock:
            self._prune_locked(now=time.monotonic())
            self._records[key] = record
            self._records.move_to_end(key)
            while len(self._records) > self.max_entries:
                self._records.popitem(last=False)
        return True

    def get(self, scope: str, response_id: str) -> ResponsesStateRecord | None:
        key = (scope, response_id)
        now = time.monotonic()
        with self._lock:
            self._prune_locked(now=now)
            record = self._records.get(key)
            if record is None:
                return None
            if record.expires_at <= now:
                self._records.pop(key, None)
                return None
            self._records.move_to_end(key)
            return copy.deepcopy(record)

    def scope_for_headers(self, headers: dict[str, str]) -> str:
        values: list[tuple[str, str]] = []
        for key in (
            "authorization",
            "x-api-key",
            "session_id",
            "session-id",
            "conversation_id",
            "conversation-id",
            "chatgpt-account-id",
            "cf-connecting-ip",
        ):
            value = headers.get(key)
            if isinstance(value, str) and value:
                values.append((key, value))

        if not values:
            values.append(("anonymous", "anonymous"))

        raw = json.dumps(values, separators=(",", ":"), sort_keys=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _prune_locked(self, *, now: float) -> None:
        expired = [
            key for key, record in self._records.items() if record.expires_at <= now
        ]
        for key in expired:
            self._records.pop(key, None)


def _normalize_input_items(value: Any) -> list[Any]:
    if isinstance(value, list):
        return copy.deepcopy(value)
    if isinstance(value, str):
        return [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": value}],
            }
        ]
    if value is None:
        return []
    return [copy.deepcopy(value)]


def _normalize_output_items(value: list[Any]) -> list[Any]:
    normalized = copy.deepcopy(value)
    normalize_responses_function_call_ids(normalized)
    return normalized


def _positive_int(value: Any, *, default: int) -> int:
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return default


__all__ = [
    "CodexResponsesStateStore",
    "ResponsesStateNotFoundError",
    "ResponsesStateRecord",
]
