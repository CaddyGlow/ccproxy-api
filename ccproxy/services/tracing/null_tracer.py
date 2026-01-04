"""Null implementation of request tracer for when tracing is disabled."""

from .interfaces import RequestTracer, StreamingTracer


class NullRequestTracer(RequestTracer, StreamingTracer):
    """No-op implementation of request tracer.

    Used as a fallback when the request_tracer plugin is disabled.
    """

    async def trace_request(
        self,
        request_id: str,
        method: str,
        url: str,
        headers: dict[str, str],
        body: bytes | None,
    ) -> None:
        """No-op request tracing."""
        pass

    async def trace_response(
        self,
        request_id: str,
        status: int,
        headers: dict[str, str],
        body: bytes,
    ) -> None:
        """No-op response tracing."""
        pass

    async def trace_stream_start(
        self,
        request_id: str,
        headers: dict[str, str],
    ) -> None:
        """No-op stream start tracing."""
        pass

    async def trace_stream_chunk(
        self,
        request_id: str,
        chunk: bytes,
        chunk_number: int,
    ) -> None:
        """No-op stream chunk tracing."""
        pass

    async def trace_stream_complete(
        self,
        request_id: str,
        total_chunks: int,
        total_bytes: int,
    ) -> None:
        """No-op stream complete tracing."""
        pass
