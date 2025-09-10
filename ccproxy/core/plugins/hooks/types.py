"""Shared hook typing for headers to support dict or dict-like inputs."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol


class HookHeaders(Protocol):
    """Protocol for header-like objects passed through hooks.

    Implementations must preserve order when iterated. Plain dicts and
    other dict-like objects can conform to this via duck typing.
    """

    def items(self) -> Iterable[tuple[str, str]]:
        """Return an iterable of (name, value) pairs in order."""
        ...

    def to_dict(self) -> dict[str, str]:  # pragma: no cover - protocol
        """Return a dict view (last occurrence wins per name)."""
        ...
