"""Request content logging hook that captures provider request/response bodies."""

import json

import structlog

from ccproxy.hooks.base import Hook, HookContext
from ccproxy.hooks.events import HookEvent
from ccproxy.utils.simple_request_logger import write_request_log


logger = structlog.get_logger(__name__)


class RequestContentLoggingHook(Hook):
    """Logs full request and response content at the provider level."""

    name = "request_content_logger"
    events = [
        HookEvent.PROVIDER_REQUEST_SENT,
        HookEvent.PROVIDER_RESPONSE_RECEIVED,
        HookEvent.PROVIDER_ERROR,
        HookEvent.PROVIDER_STREAM_START,
        HookEvent.PROVIDER_STREAM_END,
        HookEvent.PROVIDER_STREAM_CHUNK,
    ]
    priority = 20  # Run before access logging (priority 50)

    def __init__(self, logger: structlog.BoundLogger | None = None):
        """Initialize the request content logging hook.

        Args:
            logger: Optional structured logger instance
        """
        self.logger = logger or structlog.get_logger(__name__)

    async def __call__(self, context: HookContext) -> None:
        """Handle content logging based on event type.

        Args:
            context: Hook context containing request/response data
        """
        if context.event == HookEvent.PROVIDER_REQUEST_SENT:
            await self._log_provider_request(context)
        elif context.event == HookEvent.PROVIDER_RESPONSE_RECEIVED:
            await self._log_provider_response(context)
        elif context.event == HookEvent.PROVIDER_ERROR:
            await self._log_provider_error(context)
        elif context.event == HookEvent.PROVIDER_STREAM_START:
            await self._log_stream_start(context)
        elif context.event == HookEvent.PROVIDER_STREAM_END:
            await self._log_stream_end(context)
        elif context.event == HookEvent.PROVIDER_STREAM_CHUNK:
            await self._log_stream_chunk(context)

    async def _log_provider_request(self, context: HookContext) -> None:
        """Log provider request content.

        Args:
            context: Hook context with request data
        """
        data = context.data or {}

        # Extract request ID from context if available
        request_id = self._get_request_id(context)
        timestamp = self._get_timestamp(context)

        # Create request log data
        request_data = {
            "method": data.get("method"),
            "url": data.get("url"),
            "headers": data.get("headers", {}),
            "body_size": data.get("body_size", 0),
            "body": None,
        }

        # Try to parse body
        body = data.get("body")
        if body:
            try:
                if isinstance(body, bytes):
                    request_data["body"] = json.loads(body.decode("utf-8"))
                elif isinstance(body, str):
                    request_data["body"] = json.loads(body)
                else:
                    request_data["body"] = body
            except (json.JSONDecodeError, UnicodeDecodeError):
                try:
                    if isinstance(body, bytes):
                        request_data["body"] = body.decode("utf-8", errors="replace")
                    else:
                        request_data["body"] = str(body)
                except Exception:
                    request_data["body"] = f"<binary data of length {len(body)}>"

        try:
            await write_request_log(
                request_id=request_id,
                log_type="provider_request",
                data=request_data,
                timestamp=timestamp,
            )
        except Exception as e:
            logger.error(
                "failed_to_log_provider_request",
                request_id=request_id,
                error=str(e),
                exc_info=e,
            )

    async def _log_provider_response(self, context: HookContext) -> None:
        """Log provider response content.

        Args:
            context: Hook context with response data
        """
        data = context.data or {}

        # Extract request ID from context if available
        request_id = self._get_request_id(context)
        timestamp = self._get_timestamp(context)

        # Create response log data
        response_data = {
            "status_code": data.get("status_code"),
            "headers": data.get("headers", {}),
            "body_size": data.get("body_size", 0),
            "body": None,
        }

        # Try to parse body
        body = data.get("body")
        if body:
            try:
                if isinstance(body, bytes):
                    response_data["body"] = json.loads(body.decode("utf-8"))
                elif isinstance(body, str):
                    response_data["body"] = json.loads(body)
                else:
                    response_data["body"] = body
            except (json.JSONDecodeError, UnicodeDecodeError):
                try:
                    if isinstance(body, bytes):
                        response_data["body"] = body.decode("utf-8", errors="replace")
                    else:
                        response_data["body"] = str(body)
                except Exception:
                    response_data["body"] = f"<binary data of length {len(body)}>"

        try:
            await write_request_log(
                request_id=request_id,
                log_type="provider_response",
                data=response_data,
                timestamp=timestamp,
            )
        except Exception as e:
            logger.error(
                "failed_to_log_provider_response",
                request_id=request_id,
                error=str(e),
                exc_info=e,
            )

    async def _log_provider_error(self, context: HookContext) -> None:
        """Log provider error details.

        Args:
            context: Hook context with error information
        """
        data = context.data or {}

        # Extract request ID from context if available
        request_id = self._get_request_id(context)
        timestamp = self._get_timestamp(context)

        # Create error log data
        error_data = {
            "method": data.get("method"),
            "url": data.get("url"),
            "error_type": data.get("error_type"),
            "error_message": data.get("error_message"),
            "error_details": str(context.error) if context.error else None,
        }

        try:
            await write_request_log(
                request_id=request_id,
                log_type="provider_error",
                data=error_data,
                timestamp=timestamp,
            )
        except Exception as e:
            logger.error(
                "failed_to_log_provider_error",
                request_id=request_id,
                error=str(e),
                exc_info=e,
            )

    def _get_request_id(self, context: HookContext) -> str:
        """Extract request ID from context.

        Args:
            context: Hook context

        Returns:
            Request ID or 'unknown'
        """
        # Try from request object
        if context.request and hasattr(context.request, "state"):
            if hasattr(context.request.state, "request_id"):
                return str(context.request.state.request_id)
            if hasattr(context.request.state, "context"):
                ctx = context.request.state.context
                if hasattr(ctx, "request_id"):
                    return str(ctx.request_id)

        # Try from metadata
        if context.metadata and "request_id" in context.metadata:
            return str(context.metadata["request_id"])

        # Try from data
        if context.data and "request_id" in context.data:
            return str(context.data["request_id"])

        return "unknown"

    def _get_timestamp(self, context: HookContext) -> str | None:
        """Extract timestamp prefix from context.

        Args:
            context: Hook context

        Returns:
            Timestamp prefix or None
        """
        # Try from request context
        if (
            context.request
            and hasattr(context.request, "state")
            and hasattr(context.request.state, "context")
        ):
            ctx = context.request.state.context
            if hasattr(ctx, "get_log_timestamp_prefix"):
                try:
                    result = ctx.get_log_timestamp_prefix()
                    return str(result) if result is not None else None
                except Exception:
                    pass

        # Try from metadata
        if context.metadata and "timestamp_prefix" in context.metadata:
            return str(context.metadata["timestamp_prefix"])

        return None

    async def _log_stream_start(self, context: HookContext) -> None:
        """Log streaming start event.

        Args:
            context: Hook context with streaming start data
        """
        data = context.data or {}
        request_id = self._get_request_id(context)
        timestamp = self._get_timestamp(context)

        stream_data = {
            "event": "stream_start",
            "url": data.get("url"),
            "method": data.get("method"),
            "headers": data.get("headers", {}),
        }

        try:
            await write_request_log(
                request_id=request_id,
                log_type="stream_start",
                data=stream_data,
                timestamp=timestamp,
            )
        except Exception as e:
            logger.error(
                "failed_to_log_stream_start",
                request_id=request_id,
                error=str(e),
                exc_info=e,
            )

    async def _log_stream_end(self, context: HookContext) -> None:
        """Log streaming end event.

        Args:
            context: Hook context with streaming end data
        """
        data = context.data or {}
        request_id = self._get_request_id(context)
        timestamp = self._get_timestamp(context)

        stream_data = {
            "event": "stream_end",
            "total_chunks": data.get("total_chunks"),
            "total_bytes": data.get("total_bytes"),
        }

        try:
            await write_request_log(
                request_id=request_id,
                log_type="stream_end",
                data=stream_data,
                timestamp=timestamp,
            )
        except Exception as e:
            logger.error(
                "failed_to_log_stream_end",
                request_id=request_id,
                error=str(e),
                exc_info=e,
            )

    async def _log_stream_chunk(self, context: HookContext) -> None:
        """Log provider streaming chunk event.

        Args:
            context: Hook context with chunk data
        """
        from ccproxy.utils.simple_request_logger import append_streaming_log

        data = context.data or {}
        request_id = self._get_request_id(context)
        timestamp = self._get_timestamp(context)

        # Use consistent log type for provider stream chunks
        log_type = "provider_stream_chunk"

        # For PROVIDER_STREAM_CHUNK events, the chunk data is in metadata
        chunk = (
            context.metadata.get("chunk_data")
            if context.metadata
            else data.get("chunk")
        )
        chunk_type = data.get("chunk_type", "unknown")

        # Convert chunk to bytes for logging
        chunk_bytes = b""
        if chunk:
            try:
                if isinstance(chunk, bytes):
                    chunk_bytes = chunk
                elif isinstance(chunk, str):
                    chunk_bytes = chunk.encode("utf-8")
                elif isinstance(chunk, dict):
                    # JSON chunk - serialize it
                    chunk_bytes = json.dumps(chunk, separators=(",", ":")).encode(
                        "utf-8"
                    )
                else:
                    chunk_bytes = str(chunk).encode("utf-8")
            except Exception as e:
                logger.warning(
                    "failed_to_convert_chunk_to_bytes",
                    request_id=request_id,
                    chunk_type=type(chunk).__name__,
                    error=str(e),
                )
                return

        try:
            # Use append_streaming_log for chunk data
            await append_streaming_log(
                request_id=request_id,
                log_type=log_type,
                data=chunk_bytes,
                timestamp=timestamp,
            )
        except Exception as e:
            # Don't log every chunk failure as error - too noisy
            logger.debug(
                "failed_to_log_stream_chunk",
                request_id=request_id,
                log_type=log_type,
                chunk_type=chunk_type,
                error=str(e),
            )
