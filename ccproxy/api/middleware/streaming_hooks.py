"""Streaming response wrapper for hook emission.

This module provides a wrapper for streaming responses that emits
REQUEST_COMPLETED hook event when the stream actually completes.
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncGenerator, AsyncIterator
from datetime import datetime
from typing import TYPE_CHECKING, Any

from fastapi.responses import StreamingResponse

from ccproxy.core.plugins.hooks import HookContext, HookEvent
from ccproxy.utils.headers import (
    extract_response_headers,
)


if TYPE_CHECKING:
    from ccproxy.core.plugins.hooks import HookManager


class StreamingResponseWithHooks(StreamingResponse):
    """Streaming response wrapper that emits hooks on completion.

    This wrapper ensures REQUEST_COMPLETED is emitted when streaming
    actually finishes, not when the response is initially created.
    """

    def __init__(
        self,
        content: AsyncGenerator[bytes, None] | AsyncIterator[bytes],
        hook_manager: HookManager | None,
        request_id: str,
        request_data: dict[str, Any],
        start_time: float,
        status_code: int = 200,
        request_metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize streaming response with hook emission.

        Args:
            content: The async generator producing streaming content
            hook_manager: Hook manager for emitting events
            request_id: Request ID for correlation
            request_data: Original request data for context
            start_time: Request start timestamp
            status_code: HTTP status code for the response
            request_metadata: Metadata from RequestContext (includes tokens, cost, etc.)
            **kwargs: Additional arguments passed to StreamingResponse
        """
        self.hook_manager = hook_manager
        self.request_id = request_id
        self.request_data = request_data
        self.request_metadata = request_metadata or {}
        self.start_time = start_time

        # Wrap the content generator to add hook emission
        wrapped_content = self._wrap_with_hooks(content, status_code)

        super().__init__(wrapped_content, status_code=status_code, **kwargs)

    async def _wrap_with_hooks(
        self,
        content: AsyncGenerator[bytes, None] | AsyncIterator[bytes],
        status_code: int,
    ) -> AsyncGenerator[bytes, None]:
        """Wrap content generator with hook emission on completion.

        Args:
            content: The original content generator
            status_code: HTTP status code

        Yields:
            bytes: Content chunks from the original generator
        """
        error_occurred = None
        final_status = status_code
        # Collect chunks for HTTP_RESPONSE hook
        collected_chunks: list[bytes] = []

        try:
            # Stream all content from the original generator
            async for chunk in content:
                collected_chunks.append(chunk)  # Collect for HTTP hook
                yield chunk

        except GeneratorExit:
            # Client disconnected - still emit completion hook
            error_occurred = "client_disconnected"
            raise

        except Exception as e:
            # Error during streaming
            error_occurred = str(e)
            final_status = 500
            raise

        finally:
            # Emit HTTP_RESPONSE hook first with collected body, then REQUEST_COMPLETED
            if self.hook_manager:
                try:
                    end_time = time.time()
                    duration = end_time - self.start_time

                    # First emit HTTP_RESPONSE hook with collected streaming body
                    await self._emit_http_response_hook(
                        collected_chunks, final_status, end_time
                    )

                    # Then emit REQUEST_COMPLETED hook (existing behavior)
                    completion_data = {
                        "request_id": self.request_id,
                        "duration": duration,
                        "response_status": final_status,
                        "streaming_completed": True,
                    }

                    # Include original request data
                    if self.request_data:
                        completion_data.update(
                            {
                                "method": self.request_data.get("method"),
                                "url": self.request_data.get("url"),
                                "headers": self.request_data.get("headers"),
                            }
                        )

                    # Add error info if an error occurred
                    if error_occurred:
                        completion_data["error"] = error_occurred
                        event = HookEvent.REQUEST_FAILED
                    else:
                        event = HookEvent.REQUEST_COMPLETED

                    # Merge request metadata (tokens, cost, etc.) into hook metadata
                    hook_metadata = {"request_id": self.request_id}
                    hook_metadata.update(self.request_metadata)

                    hook_context = HookContext(
                        event=event,
                        timestamp=datetime.fromtimestamp(end_time),
                        data=completion_data,
                        metadata=hook_metadata,
                    )

                    await self.hook_manager.emit_with_context(hook_context)

                except Exception:
                    # Silently ignore hook emission errors to avoid breaking the stream
                    pass

    async def _emit_http_response_hook(
        self, collected_chunks: list[bytes], status_code: int, end_time: float
    ) -> None:
        """Emit HTTP_RESPONSE hook with collected streaming response body.

        Args:
            collected_chunks: All chunks collected from the stream
            status_code: Final HTTP status code
            end_time: Timestamp when streaming completed
        """
        try:
            # Combine all chunks to get full response body
            full_response_body = b"".join(collected_chunks)

            # Build HTTP response context
            http_response_context = {
                "request_id": self.request_id,
                "status_code": status_code,
                "is_client_response": True,  # Distinguish from provider responses
            }

            # Include request data for context
            if self.request_data:
                http_response_context.update(
                    {
                        "method": self.request_data.get("method"),
                        "url": self.request_data.get("url"),
                        "headers": self.request_data.get("headers"),
                    }
                )

            # Add response headers if available, preserving order and case
            try:
                http_response_context["response_headers"] = extract_response_headers(
                    self
                )
            except Exception:
                if hasattr(self, "headers"):
                    http_response_context["response_headers"] = dict(self.headers)

            # Parse response body
            if full_response_body:
                try:
                    # For streaming responses, try to parse as text first
                    response_text = full_response_body.decode("utf-8", errors="replace")

                    # Check if it looks like JSON
                    headers_obj = http_response_context.get("response_headers")
                    content_type = ""
                    if headers_obj is not None and isinstance(headers_obj, dict):
                        content_type = headers_obj.get("content-type", "")

                    if "application/json" in content_type:
                        try:
                            http_response_context["response_body"] = json.loads(
                                response_text
                            )
                        except json.JSONDecodeError:
                            http_response_context["response_body"] = response_text
                    else:
                        # For streaming responses (like SSE), include as text
                        http_response_context["response_body"] = response_text

                except UnicodeDecodeError:
                    # If decode fails, include as bytes
                    http_response_context["response_body"] = full_response_body

            # Emit HTTP_RESPONSE hook
            if self.hook_manager:
                await self.hook_manager.emit(
                    HookEvent.HTTP_RESPONSE, http_response_context
                )

        except Exception:
            # Silently ignore HTTP hook emission errors
            pass
