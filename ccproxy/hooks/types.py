"""Shared hook typing for headers to support HeaderBag or dict inputs."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol


class HookHeaders(Protocol):
    """Protocol for header-like objects passed through hooks.

    Implementations must preserve order when iterated. A minimal HeaderBag
    implementation and plain dicts can conform to this via duck typing.
    """

    def items(self) -> Iterable[tuple[str, str]]:
        """Return an iterable of (name, value) pairs in order."""
        ...

    def to_dict(self) -> dict[str, str]:  # pragma: no cover - protocol
        """Return a dict view (last occurrence wins per name)."""
        ...
