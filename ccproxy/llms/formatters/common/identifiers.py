"""Identifier helpers shared across formatter adapters."""

from __future__ import annotations

import uuid


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


__all__ = ["ensure_identifier", "normalize_suffix"]
