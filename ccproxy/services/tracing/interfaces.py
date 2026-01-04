"""Request tracing interfaces for monitoring and debugging proxy requests."""

from abc import ABC, abstractmethod


class RequestTracer(ABC):
    """Base interface for request tracing across all providers."""

    @abstractmethod
    async def trace_request(
        self,
        request_id: str,
        method: str,
        url: str,
        headers: dict[str, str],
        body: bytes | None,
    ) -> None:
        """Record request details for debugging/monitoring.

        - Logs to console with redacted sensitive headers
        - Writes complete request to file if verbose mode enabled
        - Tracks request timing and metadata
        """

    @abstractmethod
    async def trace_response(
        self, request_id: str, status: int, headers: dict[str, str], body: bytes
    ) -> None:
        """Record response details.

        - Logs response with body preview to console
        - Writes complete response to file for debugging
        - Handles JSON pretty-printing when applicable
        """


class StreamingTracer(ABC):
    """Interface for tracing streaming operations."""

    @abstractmethod
    async def trace_stream_start(
        self, request_id: str, headers: dict[str, str]
    ) -> None:
        """Mark beginning of stream with initial headers."""

    @abstractmethod
    async def trace_stream_chunk(
        self, request_id: str, chunk: bytes, chunk_number: int
    ) -> None:
        """Record individual stream chunk (optional, for deep debugging)."""

    @abstractmethod
    async def trace_stream_complete(
        self, request_id: str, total_chunks: int, total_bytes: int
    ) -> None:
        """Mark stream completion with statistics.

        - Total chunks processed
        - Total bytes transferred
        - Stream duration
        """
