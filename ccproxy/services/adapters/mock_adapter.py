"""Mock adapter for bypass mode."""

import json
import time
from typing import Any, cast

import structlog
from fastapi import Request
from fastapi.responses import Response
from starlette.responses import StreamingResponse

from ccproxy.core import logging
from ccproxy.core.constants import (
    FORMAT_ANTHROPIC_MESSAGES,
    FORMAT_OPENAI_CHAT,
    FORMAT_OPENAI_RESPONSES,
)
from ccproxy.core.request_context import RequestContext
from ccproxy.services.adapters.base import BaseAdapter
from ccproxy.services.mocking.mock_handler import MockResponseHandler
from ccproxy.streaming import DeferredStreaming


logger = logging.get_logger(__name__)


class MockAdapter(BaseAdapter):
    """Adapter for bypass/mock mode."""

    def __init__(self, mock_handler: MockResponseHandler) -> None:
        self.mock_handler = mock_handler

    async def cleanup(self) -> None:
        """Release adapter resources."""
        return None

    def _detect_format_from_endpoint(self, endpoint: str | None) -> str | None:
        """Map known route patterns to the expected output format."""

        if not endpoint:
            return None

        endpoint_lower = endpoint.lower()
        if "chat/completions" in endpoint_lower:
            return FORMAT_OPENAI_CHAT
        if "responses" in endpoint_lower:
            return FORMAT_OPENAI_RESPONSES
        return None

    def _resolve_target_format(self, request: Request, endpoint: str) -> str:
        """Infer the response format expected by the current route."""

        ctx = getattr(request.state, "context", None)
        format_chain = getattr(ctx, "format_chain", None)
        if isinstance(format_chain, list) and format_chain:
            first: str = format_chain[0]
            if first in {
                FORMAT_OPENAI_CHAT,
                FORMAT_OPENAI_RESPONSES,
                FORMAT_ANTHROPIC_MESSAGES,
            }:
                return first

        for candidate in (endpoint, getattr(request.url, "path", None)):
            detected_format = self._detect_format_from_endpoint(candidate)
            if detected_format:
                return detected_format

        return FORMAT_ANTHROPIC_MESSAGES

    def _extract_stream_flag(self, body: bytes) -> bool:
        """Check if request asks for streaming."""
        try:
            if body:
                body_json = json.loads(body)
                return bool(body_json.get("stream", False))
        except json.JSONDecodeError:
            pass
        except UnicodeDecodeError:
            pass
        except Exception as e:
            logger.debug("stream_flag_extraction_error", error=str(e))
        return False

    async def handle_request(
        self, request: Request
    ) -> Response | StreamingResponse | DeferredStreaming:
        """Handle request using mock handler."""
        body = await request.body()
        message_type = self.mock_handler.extract_message_type(body)
        prompt_text = self.mock_handler.extract_prompt_text(body)

        # Get endpoint from context or request URL
        endpoint = request.url.path
        if hasattr(request.state, "context"):
            ctx = request.state.context
            endpoint = ctx.metadata.get("endpoint", request.url.path)

        target_format = self._resolve_target_format(request, endpoint)
        model = "unknown"
        try:
            body_json = json.loads(body) if body else {}
            model = body_json.get("model", "unknown")
        except json.JSONDecodeError:
            pass
        except UnicodeDecodeError:
            pass
        except Exception as e:
            logger.debug("stream_flag_extraction_error", error=str(e))

        # Create request context
        ctx = RequestContext(
            request_id="mock-request",
            start_time=time.perf_counter(),
            logger=structlog.get_logger(__name__),
        )

        if self._extract_stream_flag(body):
            return cast(
                StreamingResponse | DeferredStreaming,
                await self.mock_handler.generate_streaming_response(
                    model, target_format, ctx, message_type, prompt_text
                ),
            )
        else:
            (
                status,
                headers,
                response_body,
            ) = await self.mock_handler.generate_standard_response(
                model, target_format, ctx, message_type, prompt_text
            )
            return Response(content=response_body, status_code=status, headers=headers)

    async def handle_streaming(
        self, request: Request, endpoint: str, **kwargs: Any
    ) -> StreamingResponse:
        """Handle a streaming request."""
        body = await request.body()
        message_type = self.mock_handler.extract_message_type(body)
        prompt_text = self.mock_handler.extract_prompt_text(body)
        target_format = self._resolve_target_format(request, endpoint)
        model = "unknown"
        try:
            body_json = json.loads(body) if body else {}
            model = body_json.get("model", "unknown")
        except json.JSONDecodeError:
            pass
        except UnicodeDecodeError:
            pass
        except Exception as e:
            logger.debug("stream_flag_extraction_error", error=str(e))

        # Create request context
        ctx = RequestContext(
            request_id=kwargs.get("request_id", "mock-stream-request"),
            start_time=time.perf_counter(),
            logger=structlog.get_logger(__name__),
        )

        return cast(
            StreamingResponse,
            await self.mock_handler.generate_streaming_response(
                model, target_format, ctx, message_type, prompt_text
            ),
        )
