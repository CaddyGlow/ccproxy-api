"""Protocol definitions for streaming adapters."""

from __future__ import annotations

from typing import Any, Protocol


class StreamAccumulatorProtocol(Protocol):
    """Protocol for stream accumulation strategies.

    Implementations should accumulate streaming chunks and return complete
    objects when they are ready for validation. This allows handling providers
    that send data in partial chunks (like tool calls across multiple chunks).
    """

    def accumulate_chunk(self, chunk: dict[str, Any]) -> dict[str, Any] | None:
        """Accumulate a streaming chunk and return complete object if ready.

        Args:
            chunk: The incoming stream chunk data

        Returns:
            None if accumulation is ongoing, or the complete object when ready
            for validation. The complete object should have all required fields
            properly populated.
        """
        ...

    def reset(self) -> None:
        """Reset accumulator state for next message.

        Called when starting a new message or when an error occurs to ensure
        clean state for the next accumulation cycle.
        """
        ...
