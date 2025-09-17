"""Legacy base adapter stub for compatibility.

This module provides the minimal interface needed for legacy adapter compatibility.
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, AsyncIterator
from typing import Any


class LegacyBaseAPIAdapter(ABC):
    """Legacy base adapter interface for compatibility."""

    def __init__(self, name: str = "legacy_adapter"):
        """Initialize with optional name."""
        self.name = name

    @abstractmethod
    async def adapt_request(self, request: dict[str, Any]) -> dict[str, Any]:
        """Adapt request format."""
        ...

    @abstractmethod
    async def adapt_response(self, response: dict[str, Any]) -> dict[str, Any]:
        """Adapt response format."""
        ...

    @abstractmethod
    def adapt_stream(
        self, stream: AsyncIterator[dict[str, Any]]
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Adapt streaming response format."""
        ...

    async def adapt_error(self, error: dict[str, Any]) -> dict[str, Any]:
        """Adapt error format - optional method."""
        return error