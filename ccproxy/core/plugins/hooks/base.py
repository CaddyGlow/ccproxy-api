"""Core interfaces for the hook system."""

from collections.abc import Awaitable
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol

from fastapi import Request, Response

from .events import HookEvent


@dataclass
class HookContext:
    """Context passed to all hooks"""

    event: HookEvent
    timestamp: datetime
    data: dict[str, Any]
    metadata: dict[str, Any]

    # Request-specific (optional)
    request: Request | None = None
    response: Response | None = None

    # Provider-specific (optional)
    provider: str | None = None
    plugin: str | None = None

    # Error context (optional)
    error: Exception | None = None


class Hook(Protocol):
    """Base hook protocol"""

    def __call__(self, context: HookContext) -> None | Awaitable[None]:
        """Execute hook with context (can be async or sync)"""
        ...

    @property
    def name(self) -> str:
        """Hook name for debugging"""
        ...

    @property
    def events(self) -> list[HookEvent]:
        """Events this hook listens to"""
        ...

    @property
    def priority(self) -> int:
        """Hook execution priority (0-1000, lower executes first).

        Default is 500 (middle priority) for backward compatibility.
        See HookLayer enum for standard priority values.
        """
        return 500
