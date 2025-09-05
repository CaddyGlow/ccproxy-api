"""HTTP client with hook support for request/response interception."""

from typing import Any

import httpx
from httpx._types import (
    HeaderTypes,
    QueryParamTypes,
    RequestContent,
    RequestData,
    RequestFiles,
)

from ccproxy.core.logging import get_logger
from ccproxy.core.request_context import RequestContext
from ccproxy.hooks.events import HookEvent


logger = get_logger(__name__)


class HookableHTTPClient(httpx.AsyncClient):
    """HTTP client wrapper that emits hooks for all requests/responses."""

    def __init__(self, *args: Any, hook_manager: Any | None = None, **kwargs: Any):
        """Initialize HTTP client with optional hook support.

        Args:
            *args: Arguments for httpx.AsyncClient
            hook_manager: Optional HookManager instance for emitting hooks
            **kwargs: Keyword arguments for httpx.AsyncClient
        """
        super().__init__(*args, **kwargs)
        self.hook_manager = hook_manager

    async def request(
        self,
        method: str,
        url: httpx.URL | str,
        *,
        content: RequestContent | None = None,
        data: RequestData | None = None,
        files: RequestFiles | None = None,
        params: QueryParamTypes | None = None,
        headers: HeaderTypes | None = None,
        json: Any | None = None,
        **kwargs: Any,
    ) -> httpx.Response:
        """Make an HTTP request with hook emissions.

        Emits:
            - HTTP_REQUEST before sending
            - HTTP_RESPONSE after receiving response
            - HTTP_ERROR on errors
        """
        # Build request context for hooks
        request_context: dict[str, Any] = {
            "method": method,
            "url": str(url),
            "headers": dict(headers) if headers else {},
        }

        # Try to get current request ID from RequestContext
        try:
            current_context = RequestContext.get_current()
            if current_context and hasattr(current_context, "request_id"):
                request_context["request_id"] = current_context.request_id
        except Exception:
            # If no request context available, hooks will generate their own ID
            pass

        # Add body information
        if json is not None:
            request_context["body"] = json
            request_context["is_json"] = True
        elif data is not None:
            request_context["body"] = data
            request_context["is_json"] = False
        elif content is not None:
            # Handle content parameter - could be bytes, string, or other
            if isinstance(content, bytes | str):
                try:
                    # Try to parse as JSON if it's a string/bytes that looks like JSON
                    import json as json_module

                    if isinstance(content, bytes):
                        content_str = content.decode("utf-8")
                    else:
                        content_str = content

                    if content_str.strip().startswith(("{", "[")):
                        request_context["body"] = json_module.loads(content_str)
                        request_context["is_json"] = True
                    else:
                        request_context["body"] = content
                        request_context["is_json"] = False
                except Exception:
                    # If parsing fails, just include as-is
                    request_context["body"] = content
                    request_context["is_json"] = False
            else:
                request_context["body"] = content
                request_context["is_json"] = False

        # Emit pre-request hook
        if self.hook_manager:
            try:
                await self.hook_manager.emit(
                    HookEvent.HTTP_REQUEST,
                    request_context,
                )
            except Exception as e:
                logger.debug(
                    "http_request_hook_error",
                    error=str(e),
                    method=method,
                    url=str(url),
                )

        try:
            # Make the actual request
            response = await super().request(
                method,
                url,
                content=content,
                data=data,
                files=files,
                json=json,
                params=params,
                headers=headers,
                **kwargs,
            )

            # Emit post-response hook
            if self.hook_manager:
                response_context = {
                    **request_context,  # Include request info
                    "status_code": response.status_code,
                    "response_headers": dict(response.headers),
                }

                # Try to include response body
                try:
                    content_type = response.headers.get("content-type", "")
                    if "application/json" in content_type:
                        # Try to parse as JSON first
                        try:
                            response_context["response_body"] = response.json()
                        except Exception:
                            # If JSON parsing fails, fall back to text
                            response_context["response_body"] = response.text
                    else:
                        # For non-JSON content, include as text
                        response_context["response_body"] = response.text
                except Exception:
                    # Can't get body content, that's OK
                    pass

                try:
                    await self.hook_manager.emit(
                        HookEvent.HTTP_RESPONSE,
                        response_context,
                    )
                except Exception as e:
                    logger.debug(
                        "http_response_hook_error",
                        error=str(e),
                        status_code=response.status_code,
                    )

            return response

        except Exception as error:
            # Emit error hook
            if self.hook_manager:
                error_context = {
                    **request_context,
                    "error_type": type(error).__name__,
                    "error_detail": str(error),
                }

                # Add response info if it's an HTTPStatusError
                if isinstance(error, httpx.HTTPStatusError):
                    error_context["status_code"] = error.response.status_code
                    error_context["response_body"] = error.response.text

                try:
                    await self.hook_manager.emit(
                        HookEvent.HTTP_ERROR,
                        error_context,
                    )
                except Exception as e:
                    logger.debug(
                        "http_error_hook_error",
                        error=str(e),
                        original_error=str(error),
                    )

            # Re-raise the original error
            raise
