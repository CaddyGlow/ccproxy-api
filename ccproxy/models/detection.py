"""Shared helper dataclasses for plugin detection caches."""

from __future__ import annotations

from collections.abc import ItemsView, Iterable
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class DetectedHeaders:
    """Normalized, lowercase HTTP headers captured during CLI detection."""

    values: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        normalized: dict[str, str] = {}
        for key, raw_value in (self.values or {}).items():
            if key is None:
                continue
            normalized_key = str(key).lower()
            normalized[normalized_key] = "" if raw_value is None else str(raw_value)
        self.values = normalized

    def as_dict(self) -> dict[str, str]:
        """Return a copy of the detected headers as a plain dict."""

        return dict(self.values)

    def filtered(
        self,
        ignores: Iterable[str] | None = None,
        redacted: Iterable[str] | None = None,
    ) -> dict[str, str]:
        """Return headers filtered for safe forwarding."""

        ignore_set = {item.lower() for item in ignores or ()}
        redacted_set = {item.lower() for item in redacted or ()}
        return {
            key: value
            for key, value in self.values.items()
            if value and key not in ignore_set and key not in redacted_set
        }

    def get(self, key: str, default: str | None = None) -> str | None:
        """Lookup a header by key (case-insensitive)."""

        return self.values.get(key.lower(), default)

    def items(self) -> ItemsView[str, str]:
        """Iterate over header key/value pairs."""

        return self.values.items()

    def __bool__(self) -> bool:
        return bool(self.values)


@dataclass(slots=True)
class DetectedPrompts:
    """Structured prompt metadata extracted from CLI detection payloads."""

    instructions: str | None = None
    system: Any | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_body(cls, body: Any | None) -> DetectedPrompts:
        """Build a DetectedPrompts instance from a captured request body."""

        if not isinstance(body, dict):
            return cls(raw={} if body is None else {"__raw__": body})

        body_copy = deepcopy(body)

        instructions = body_copy.get("instructions")
        if not isinstance(instructions, str) or not instructions.strip():
            instructions = None

        system_value = body_copy.get("system")

        return cls(instructions=instructions, system=system_value, raw=body_copy)

    def instructions_payload(self) -> dict[str, Any]:
        """Return a payload suitable for injecting Codex-style instructions."""

        if self.instructions:
            return {"instructions": self.instructions}
        return {}

    def system_payload(self, mode: str = "full") -> dict[str, Any]:
        """Return anthropic-style system data respecting the requested mode."""

        if self.system is None or mode == "none":
            return {}

        if mode == "minimal" and isinstance(self.system, list):
            return {"system": self.system[:1]} if self.system else {}

        return {"system": self.system}

    def has_system(self) -> bool:
        return bool(self.system)

    def has_instructions(self) -> bool:
        return bool(self.instructions)
