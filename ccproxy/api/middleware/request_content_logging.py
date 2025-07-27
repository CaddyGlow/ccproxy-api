"""Request content logging middleware for capturing full HTTP request/response data."""

import json
from collections.abc import AsyncGenerator
from typing import Any

import structlog
from fastapi import Request, Response
from fastapi.responses import StreamingResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from ccproxy.utils.simple_request_logger import (
    append_streaming_log,
    write_request_log,
)


logger = structlog.get_logger(__name__)


class RequestContentLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for logging full HTTP request and response content."""

    def __init__(self, app: ASGIApp):
        """Initialize the request content logging middleware.

        Args:
            app: The ASGI application
        """
        super().__init__(app)

    async def dispatch(self, request: Request, call_next: Any) -> Any:
        """Process the request and log content.

        Args:
            request: The incoming HTTP request
            call_next: The next middleware/handler in the chain

        Returns:
            The HTTP response
        """
        # Get request ID and timestamp from context if available
        request_id = self._get_request_id(request)
        timestamp = self._get_timestamp_prefix(request)

        # Log incoming request
        await self._log_request(request, request_id, timestamp)

        # Process the request
        response = await call_next(request)

        # Log outgoing response
        await self._log_response(response, request_id, timestamp)

        return response

    def _get_request_id(self, request: Request) -> str:
        """Extract request ID from request state or context.

        Args:
            request: The HTTP request

        Returns:
            Request ID string or 'unknown' if not found
        """
        try:
            # Try to get from request state
            if hasattr(request.state, "request_id"):
                return str(request.state.request_id)

            # Try to get from request context
            if hasattr(request.state, "context"):
                context = request.state.context
                if hasattr(context, "request_id"):
                    return str(context.request_id)

            # Fallback to UUID if available in headers
            if "x-request-id" in request.headers:
                return request.headers["x-request-id"]

        except Exception:
            pass  # Ignore errors and use fallback

        return "unknown"

    def _get_timestamp_prefix(self, request: Request) -> str | None:
        """Extract timestamp prefix from request context.

        Args:
            request: The HTTP request

        Returns:
            Timestamp prefix string or None if not found
        """
        try:
            # Try to get from request context
            if hasattr(request.state, "context"):
                context = request.state.context
                if hasattr(context, "get_log_timestamp_prefix"):
                    result = context.get_log_timestamp_prefix()
                    return str(result) if result is not None else None
        except Exception:
            pass  # Ignore errors and use fallback

        return None

    async def _log_request(
        self, request: Request, request_id: str, timestamp: str | None
    ) -> None:
        """Log incoming HTTP request content.

        Args:
            request: The HTTP request
            request_id: Request identifier
            timestamp: Timestamp prefix for the log file
        """
        try:
            # Read request body
            body = await request.body()

            # Create request log data
            request_data = {
                "method": request.method,
                "url": str(request.url),
                "headers": dict(request.headers),
                "query_params": dict(request.query_params),
                "path_params": dict(request.path_params)
                if hasattr(request, "path_params")
                else {},
                "body_size": len(body) if body else 0,
                "body": None,
            }

            # Try to parse body as JSON, fallback to string
            if body:
                try:
                    request_data["body"] = json.loads(body.decode("utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    try:
                        request_data["body"] = body.decode("utf-8", errors="replace")
                    except Exception:
                        request_data["body"] = f"<binary data of length {len(body)}>"

            await write_request_log(
                request_id=request_id,
                log_type="middleware_request",
                data=request_data,
                timestamp=timestamp,
            )

        except Exception as e:
            logger.error(
                "failed_to_log_request_content",
                request_id=request_id,
                error=str(e),
            )

    async def _log_response(
        self, response: Response, request_id: str, timestamp: str | None
    ) -> None:
        """Log outgoing HTTP response content.

        Args:
            response: The HTTP response
            request_id: Request identifier
            timestamp: Timestamp prefix for the log file
        """
        try:
            if isinstance(response, StreamingResponse):
                # Handle streaming response
                await self._log_streaming_response(response, request_id, timestamp)
            else:
                # Handle regular response
                await self._log_regular_response(response, request_id, timestamp)

        except Exception as e:
            logger.error(
                "failed_to_log_response_content",
                request_id=request_id,
                error=str(e),
            )

    async def _log_regular_response(
        self, response: Response, request_id: str, timestamp: str | None
    ) -> None:
        """Log regular (non-streaming) HTTP response.

        Args:
            response: The HTTP response
            request_id: Request identifier
            timestamp: Timestamp prefix for the log file
        """
        # Create response log data
        response_data = {
            "status_code": response.status_code,
            "headers": dict(response.headers),
            "body": None,
        }

        # Try to get response body
        if hasattr(response, "body") and response.body:
            body = response.body
            response_data["body_size"] = len(body)

            try:
                # Convert to bytes if needed
                body_bytes = bytes(body) if isinstance(body, memoryview) else body
                # Try to parse as JSON
                response_data["body"] = json.loads(body_bytes.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                try:
                    # Fallback to string
                    body_bytes = bytes(body) if isinstance(body, memoryview) else body
                    response_data["body"] = body_bytes.decode("utf-8", errors="replace")
                except Exception:
                    response_data["body"] = f"<binary data of length {len(body)}>"
        else:
            response_data["body_size"] = 0

        await write_request_log(
            request_id=request_id,
            log_type="middleware_response",
            data=response_data,
            timestamp=timestamp,
        )

    async def _log_streaming_response(
        self, response: StreamingResponse, request_id: str, timestamp: str | None
    ) -> None:
        """Log streaming HTTP response by wrapping the stream.

        Args:
            response: The streaming HTTP response
            request_id: Request identifier
            timestamp: Timestamp prefix for the log file
        """
        # Log response metadata first
        response_data = {
            "status_code": response.status_code,
            "headers": dict(response.headers),
            "body_type": "streaming",
            "media_type": response.media_type,
        }

        await write_request_log(
            request_id=request_id,
            log_type="middleware_response",
            data=response_data,
            timestamp=timestamp,
        )

        # Wrap the streaming response to capture content
        original_body_iterator = response.body_iterator

        def create_logged_body_iterator() -> AsyncGenerator[
            str | bytes | memoryview[int], None
        ]:
            """Create wrapper around the original body iterator to log streaming content."""

            async def logged_body_iterator() -> AsyncGenerator[
                str | bytes | memoryview[int], None
            ]:
                try:
                    async for chunk in original_body_iterator:
                        # Log chunk as raw data
                        if isinstance(chunk, bytes | bytearray):
                            await append_streaming_log(
                                request_id=request_id,
                                log_type="middleware_streaming",
                                data=bytes(chunk),
                                timestamp=timestamp,
                            )
                        elif isinstance(chunk, str):
                            await append_streaming_log(
                                request_id=request_id,
                                log_type="middleware_streaming",
                                data=chunk.encode("utf-8"),
                                timestamp=timestamp,
                            )

                        yield chunk

                except Exception as e:
                    logger.error(
                        "error_in_streaming_response_logging",
                        request_id=request_id,
                        error=str(e),
                    )
                    # Continue with original iterator if logging fails
                    async for chunk in original_body_iterator:
                        yield chunk

            return logged_body_iterator()

        # Replace the body iterator with our logged version
        response.body_iterator = create_logged_body_iterator()
