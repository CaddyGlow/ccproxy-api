"""Mock adapter for bypass mode."""

import json
import time
from typing import Any

import structlog
from fastapi import Request
from fastapi.responses import Response
from starlette.responses import StreamingResponse

from ccproxy.observability.context import RequestContext
from ccproxy.services.adapters.base import BaseAdapter
from ccproxy.services.mocking.mock_handler import MockResponseHandler


class MockAdapter(BaseAdapter):
    """Adapter for bypass/mock mode."""

    def __init__(self, mock_handler: MockResponseHandler) -> None:
        self.mock_handler = mock_handler

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
            structlog.get_logger(__name__).debug(
                "stream_flag_extraction_error", error=str(e)
            )
            pass
        return False

    async def handle_request(
        self, request: Request, endpoint: str, method: str, **kwargs: Any
    ) -> Response:
        """Handle request using mock handler."""
        body = await request.body()
        message_type = self.mock_handler.extract_message_type(body)
        is_openai = "openai" in endpoint
        model = "unknown"
        try:
            body_json = json.loads(body) if body else {}
            model = body_json.get("model", "unknown")
        except json.JSONDecodeError:
            pass
        except UnicodeDecodeError:
            pass
        except Exception as e:
            structlog.get_logger(__name__).debug(
                "stream_flag_extraction_error", error=str(e)
            )
            pass

        # Create request context
        ctx = RequestContext(
            request_id=kwargs.get("request_id", "mock-request"),
            start_time=time.perf_counter(),
            logger=structlog.get_logger(__name__),
        )

        if self._extract_stream_flag(body):
            return await self.mock_handler.generate_streaming_response(
                model, is_openai, ctx, message_type
            )
        else:
            (
                status,
                headers,
                response_body,
            ) = await self.mock_handler.generate_standard_response(
                model, is_openai, ctx, message_type
            )
            return Response(content=response_body, status_code=status, headers=headers)

    async def handle_streaming(
        self, request: Request, endpoint: str, **kwargs: Any
    ) -> StreamingResponse:
        """Handle a streaming request."""
        body = await request.body()
        message_type = self.mock_handler.extract_message_type(body)
        is_openai = "openai" in endpoint
        model = "unknown"
        try:
            body_json = json.loads(body) if body else {}
            model = body_json.get("model", "unknown")
        except json.JSONDecodeError:
            pass
        except UnicodeDecodeError:
            pass
        except Exception as e:
            structlog.get_logger(__name__).debug(
                "stream_flag_extraction_error", error=str(e)
            )
            pass

        # Create request context
        ctx = RequestContext(
            request_id=kwargs.get("request_id", "mock-stream-request"),
            start_time=time.perf_counter(),
            logger=structlog.get_logger(__name__),
        )

        return await self.mock_handler.generate_streaming_response(
            model, is_openai, ctx, message_type
        )
