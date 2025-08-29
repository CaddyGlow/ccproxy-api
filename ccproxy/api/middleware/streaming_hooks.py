"""Streaming response wrapper for hook emission.

This module provides a wrapper for streaming responses that emits
REQUEST_COMPLETED hook event when the stream actually completes.
"""

from __future__ import annotations

import time
from collections.abc import AsyncGenerator, AsyncIterator
from datetime import datetime
from typing import TYPE_CHECKING, Any

from fastapi.responses import StreamingResponse

from ccproxy.hooks import HookContext, HookEvent


if TYPE_CHECKING:
    from ccproxy.hooks import HookManager


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
            **kwargs: Additional arguments passed to StreamingResponse
        """
        self.hook_manager = hook_manager
        self.request_id = request_id
        self.request_data = request_data
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

        try:
            # Stream all content from the original generator
            async for chunk in content:
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
            # Emit REQUEST_COMPLETED hook when streaming actually completes
            if self.hook_manager:
                try:
                    end_time = time.time()
                    duration = end_time - self.start_time

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

                    hook_context = HookContext(
                        event=event,
                        timestamp=datetime.fromtimestamp(end_time),
                        data=completion_data,
                        metadata={"request_id": self.request_id},
                    )

                    await self.hook_manager.emit_with_context(hook_context)

                except Exception:
                    # Silently ignore hook emission errors to avoid breaking the stream
                    pass

