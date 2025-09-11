import json
import time
import uuid
from typing import Any

import httpx
from starlette.responses import Response, StreamingResponse

from ccproxy.core.logging import get_plugin_logger
from ccproxy.core.request_context import RequestContext
from ccproxy.services.adapters.http_adapter import BaseHTTPAdapter
from ccproxy.streaming import DeferredStreaming
from ccproxy.utils.headers import (
    extract_response_headers,
    filter_request_headers,
    to_canonical_headers,
)

from .detection_service import CodexDetectionService


logger = get_plugin_logger()


class CodexAdapter(BaseHTTPAdapter):
    """Simplified Codex adapter."""

    def __init__(self, detection_service: CodexDetectionService, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.detection_service = detection_service
        self.base_url = self.config.base_url.rstrip("/")

    async def get_target_url(self, endpoint: str) -> str:
        # Old URL: https://chat.openai.com/backend-anon/responses (308 redirect)
        return f"{self.base_url}/responses"

    async def prepare_provider_request(
        self, body: bytes, headers: dict[str, str], endpoint: str
    ) -> tuple[bytes, dict[str, str]]:
        # Get auth credentials and profile
        auth_data = await self.auth_manager.load_credentials()
        if not auth_data:
            raise ValueError("No authentication credentials available")

        # Get profile to extract chatgpt_account_id
        profile = await self.auth_manager.get_profile_quick()
        chatgpt_account_id = profile.chatgpt_account_id if profile else None

        # Parse body (format conversion is now handled by format chain)
        body_data = json.loads(body.decode()) if body else {}

        # Inject instructions if not present
        if "instructions" not in body_data or body_data.get("instructions") is None:
            body_data["instructions"] = self._get_instructions()

        # Store user's original streaming preference
        user_requested_streaming = body_data.get("stream", True)

        # Codex backend requires stream=true, always override
        body_data["stream"] = True
        body_data["store"] = False

        # Store original preference temporarily for response handling
        # Since we don't have access to request context here, store it on the adapter instance
        if not hasattr(self, "_user_streaming_preferences"):
            self._user_streaming_preferences = {}
        # Use a simple key since we don't have request ID access here
        self._user_streaming_preferences["current"] = user_requested_streaming

        # Remove any prefixed metadata fields that shouldn't be sent to the API
        body_data = self._remove_metadata_fields(body_data)

        # Filter and add headers
        filtered_headers = filter_request_headers(headers, preserve_auth=False)
        base_headers = {
            "authorization": f"Bearer {auth_data.access_token}",
            "session_id": str(uuid.uuid4()),
            "content-type": "application/json",
        }
        # Add chatgpt-account-id only if available
        if chatgpt_account_id is not None:
            base_headers["chatgpt-account-id"] = chatgpt_account_id

        filtered_headers.update(base_headers)

        # Add CLI headers
        if self.detection_service:
            cached_data = self.detection_service.get_cached_data()
            if cached_data and cached_data.headers:
                cli_headers = cached_data.headers.to_headers_dict()
                for key, value in cli_headers.items():
                    filtered_headers[key.lower()] = value

        return json.dumps(body_data).encode(), filtered_headers

    async def process_provider_response(
        self, response: httpx.Response, endpoint: str, ctx: Any = None
    ) -> Response | StreamingResponse | DeferredStreaming:
        """Process provider response with streaming support and format conversion."""
        logger.debug(
            "codex_processing_provider_response",
            endpoint=endpoint,
            status_code=response.status_code,
            content_type=response.headers.get("content-type"),
            category="response_processing",
        )

        # Check if this is a streaming response
        content_type = response.headers.get("content-type", "")
        backend_is_streaming = (
            "text/event-stream" in content_type or "stream" in content_type.lower()
        )

        # Check if user originally requested streaming
        user_requested_streaming = True  # Default to streaming
        if (
            hasattr(self, "_user_streaming_preferences")
            and "current" in self._user_streaming_preferences
        ):
            user_requested_streaming = self._user_streaming_preferences["current"]
            logger.debug(
                "codex_retrieved_user_streaming_preference",
                user_requested_streaming=user_requested_streaming,
                category="streaming_conversion",
            )

        if backend_is_streaming and user_requested_streaming:
            logger.debug(
                "codex_streaming_response_for_streaming_request",
                endpoint=endpoint,
                content_type=content_type,
                category="streaming_conversion",
            )
            # User wants streaming, backend provides streaming
            return await self._create_streaming_response(response, endpoint)
        elif backend_is_streaming and not user_requested_streaming:
            logger.debug(
                "codex_converting_stream_to_non_streaming",
                endpoint=endpoint,
                content_type=content_type,
                user_requested_streaming=user_requested_streaming,
                category="streaming_conversion",
            )
            # User wants non-streaming, but backend returned streaming
            # Use the existing StreamingBufferService to convert stream to JSON
            from ccproxy.streaming.buffer import StreamingBufferService

            # Create buffer service instance
            buffer_service = StreamingBufferService(
                http_client=await self.http_pool_manager.get_client(),
                http_pool_manager=self.http_pool_manager,
            )

            # Create minimal handler config for the buffer service
            from ccproxy.services.handler_config import HandlerConfig

            handler_config = HandlerConfig(supports_streaming=True)

            # Get the original request details from the response
            original_request = response.request

            # Create a proper RequestContext instance
            request_context = RequestContext(
                request_id="codex-buffer",
                start_time=time.perf_counter(),
                logger=logger,
                metadata={},
                metrics={},
            )

            # Use the buffer service to convert the stream to non-streaming response
            return await buffer_service.handle_buffered_streaming_request(
                method=original_request.method,
                url=str(original_request.url),
                headers=dict(original_request.headers),
                body=original_request.content,
                handler_config=handler_config,
                request_context=request_context,
                provider_name="codex",
            )
        else:
            # Non-streaming response - handle as before
            logger.debug(
                "codex_non_streaming_response",
                endpoint=endpoint,
                category="response_processing",
            )

            # Check if response is JSON before parsing
            try:
                response_data = json.loads(response.content)
            except json.JSONDecodeError as e:
                logger.error(
                    "invalid_json_response",
                    status_code=response.status_code,
                    content_type=response.headers.get("content-type"),
                    content_preview=response.content[:200]
                    if response.content
                    else None,
                    error=str(e),
                )
                # For non-JSON responses (like redirects), return as-is
                response_headers = extract_response_headers(response)
                return Response(
                    content=response.content,
                    status_code=response.status_code,
                    headers=response_headers,
                )

            # Response format conversion is now handled by format chain
            response_headers = extract_response_headers(response)

            return Response(
                content=response.content,
                status_code=response.status_code,
                headers=to_canonical_headers(response_headers),
            )

    async def _create_streaming_response(
        self, response: httpx.Response, endpoint: str
    ) -> DeferredStreaming:
        """Create streaming response with format conversion support (new architecture)."""

        # Check if format conversion is needed based on endpoint
        needs_conversion = self._needs_format_conversion(endpoint)
        response_adapter = None

        if needs_conversion and self.format_registry:
            try:
                # Determine the format conversion direction based on endpoint
                from_format, to_format = self._get_response_format_conversion(endpoint)
                response_adapter = self.format_registry.get_adapter(
                    from_format, to_format
                )

                logger.debug(
                    "codex_format_adapter_loaded",
                    endpoint=endpoint,
                    from_format=from_format,
                    to_format=to_format,
                    has_response_adapter=bool(response_adapter),
                    category="streaming_conversion",
                )
            except Exception as e:
                logger.warning(
                    "codex_format_adapter_loading_failed",
                    error=str(e),
                    endpoint=endpoint,
                    category="streaming_conversion",
                )

        # Get HTTP client from pool manager (new architecture)
        client = await self.http_pool_manager.get_client()

        # NEW ARCHITECTURE: Create minimal HandlerConfig only for format conversion
        # This avoids the full delegation pattern while preserving format conversion
        handler_config = None
        if response_adapter:
            from ccproxy.services.handler_config import HandlerConfig

            handler_config = HandlerConfig(
                response_adapter=response_adapter,
                supports_streaming=True,
            )
            from_format, to_format = self._get_response_format_conversion(endpoint)
            logger.debug(
                "codex_minimal_handler_config_created",
                endpoint=endpoint,
                from_format=from_format,
                to_format=to_format,
                category="streaming_conversion",
            )

        # Create DeferredStreaming with minimal handler config for format conversion
        return DeferredStreaming(
            method="POST",
            url=str(response.url),
            headers=dict(response.request.headers),
            body=response.request.content,
            client=client,
            handler_config=handler_config,
        )

    def _needs_format_conversion(self, endpoint: str) -> bool:
        """Check if this endpoint needs format conversion."""
        # Both OpenAI and Anthropic format endpoints need conversion
        return (
            endpoint.endswith("/chat/completions")
            or endpoint.endswith("/v1/chat/completions")
            or endpoint.endswith("/messages")
            or endpoint.endswith("/v1/messages")
        )

    def _get_response_format_conversion(self, endpoint: str) -> tuple[str, str]:
        """Get the response format conversion direction based on endpoint."""
        if endpoint.endswith("/messages") or endpoint.endswith("/v1/messages"):
            # Anthropic format endpoints: response_api -> anthropic
            return ("response_api", "anthropic")
        elif endpoint.endswith("/chat/completions") or endpoint.endswith(
            "/v1/chat/completions"
        ):
            # OpenAI format endpoints: response_api -> openai
            return ("response_api", "openai")
        else:
            # Default fallback (shouldn't happen if _needs_format_conversion is correct)
            return ("response_api", "openai")

    # Helper methods
    def _remove_metadata_fields(self, data: dict[str, Any]) -> dict[str, Any]:
        """Remove fields that start with '_' as they are internal metadata.

        Args:
            data: Dictionary that may contain metadata fields

        Returns:
            Cleaned dictionary without metadata fields
        """
        if not isinstance(data, dict):
            return data

        # Create a new dict without keys starting with '_'
        cleaned_data: dict[str, Any] = {}
        for key, value in data.items():
            if not key.startswith("_"):
                # Recursively clean nested dictionaries
                if isinstance(value, dict):
                    cleaned_data[key] = self._remove_metadata_fields(value)
                elif isinstance(value, list):
                    # Clean list items if they are dictionaries
                    cleaned_items: list[Any] = []
                    for item in value:
                        if isinstance(item, dict):
                            cleaned_items.append(self._remove_metadata_fields(item))
                        else:
                            cleaned_items.append(item)
                    cleaned_data[key] = cleaned_items
                else:
                    cleaned_data[key] = value

        return cleaned_data

    def _get_instructions(self) -> str:
        if self.detection_service:
            cached_data = self.detection_service.get_cached_data()
            if cached_data and cached_data.instructions:
                return cached_data.instructions.instructions_field
        return "You are a coding agent..."

    def adapt_error(self, error_body: dict[str, Any]) -> dict[str, Any]:
        """Convert Codex error format to appropriate API error format.

        Args:
            error_body: Codex error response

        Returns:
            API-formatted error response
        """
        # Handle the specific "Stream must be set to true" error
        if isinstance(error_body, dict) and "detail" in error_body:
            detail = error_body["detail"]
            if "Stream must be set to true" in detail:
                # Convert to generic invalid request error
                return {
                    "error": {
                        "type": "invalid_request_error",
                        "message": "Invalid streaming parameter",
                    }
                }

        # Handle other error formats that might have "error" key
        if "error" in error_body:
            return error_body

        # Default: wrap non-standard errors
        return {
            "error": {
                "type": "internal_server_error",
                "message": "An error occurred processing the request",
            }
        }

    async def _convert_stream_to_json(
        self, response: httpx.Response, endpoint: str
    ) -> Response:
        """Convert a streaming response to a single JSON response.

        This is used when the user requests non-streaming but the backend
        returns a stream (which is required by Codex).
        """
        import json

        from starlette.responses import Response as StarletteResponse

        logger.debug(
            "codex_collecting_stream_for_json_conversion",
            endpoint=endpoint,
            category="streaming_conversion",
        )

        try:
            # Collect all streaming chunks
            collected_content = []
            accumulated_text = ""

            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data_part = line[6:]  # Remove "data: " prefix
                    if data_part.strip() == "[DONE]":
                        break

                    try:
                        chunk_data = json.loads(data_part)
                        collected_content.append(chunk_data)

                        # For Response API format, accumulate text content
                        if "response" in chunk_data:
                            response_data = chunk_data["response"]
                            if "output" in response_data:
                                for output_item in response_data["output"]:
                                    if output_item.get("type") == "message":
                                        for content_block in output_item.get(
                                            "content", []
                                        ):
                                            if content_block.get("type") == "text":
                                                accumulated_text += content_block.get(
                                                    "text", ""
                                                )

                    except json.JSONDecodeError:
                        continue

            # Create a consolidated response in the expected format
            if self._needs_format_conversion(endpoint):
                # For OpenAI format endpoints
                consolidated_response = {
                    "id": f"resp_{collected_content[0]['response']['id'] if collected_content and 'response' in collected_content[0] else 'unknown'}",
                    "object": "chat.completion",
                    "created": int(
                        collected_content[0]["response"]["created_at"]
                        if collected_content and "response" in collected_content[0]
                        else 0
                    ),
                    "model": "gpt-5",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": accumulated_text,
                            },
                            "finish_reason": "stop",
                        }
                    ],
                }
            else:
                # For Anthropic format endpoints
                consolidated_response = {
                    "content": [{"type": "text", "text": accumulated_text}],
                    "stop_reason": "end_turn",
                }

            # Apply format conversion if needed
            if self._needs_format_conversion(endpoint):
                from_format, to_format = self._get_response_format_conversion(endpoint)
                if self.format_registry:
                    adapter = self.format_registry.get_adapter(from_format, to_format)
                    consolidated_response = await adapter.adapt_response(
                        consolidated_response
                    )

            # Return as regular JSON response
            response_headers = extract_response_headers(response)
            response_headers["content-type"] = "application/json"

            return StarletteResponse(
                content=json.dumps(consolidated_response),
                status_code=response.status_code,
                headers=response_headers,
                media_type="application/json",
            )

        except Exception as e:
            logger.error(
                "codex_stream_to_json_conversion_failed",
                endpoint=endpoint,
                error=str(e),
                category="streaming_conversion",
            )
            # Fallback to error response
            return StarletteResponse(
                content=json.dumps(
                    {
                        "error": {
                            "type": "internal_server_error",
                            "message": "Failed to process streaming response",
                        }
                    }
                ),
                status_code=500,
                headers={"content-type": "application/json"},
                media_type="application/json",
            )
