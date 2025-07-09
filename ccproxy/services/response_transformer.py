"""Response transformation service for reverse proxy."""

import json
from typing import Any

from ccproxy.utils.logging import get_logger
from ccproxy.utils.openai import is_openai_request


logger = get_logger(__name__)


class ResponseTransformer:
    """Handles response transformations for reverse proxy."""

    def transform_response_body(
        self, body: bytes, path: str, mode: str = "full"
    ) -> bytes:
        """Transform response body based on the endpoint.

        Args:
            body: Original response body
            path: Original request path for context
            mode: Proxy mode - "minimal" or "full"

        Returns:
            Transformed response body
        """
        # In minimal mode, don't transform responses
        if mode == "minimal":
            return body

        # For claude_code, full, and api modes - transform OpenAI requests
        # Check if this is an OpenAI request that needs response conversion
        if self._is_openai_request(path):
            return self._transform_anthropic_to_openai_response(body, path)

        # For direct Anthropic requests, pass through without transformation
        return body

    def transform_response_headers(
        self,
        headers: dict[str, str],
        path: str = "",
        body_size: int | None = None,
        mode: str = "full",
    ) -> dict[str, str]:
        """Transform response headers if needed.

        Args:
            headers: Original response headers
            path: Request path for context
            body_size: New body size if content was transformed
            mode: Proxy mode - "minimal" or "full"

        Returns:
            Transformed response headers
        """
        # Create a copy to avoid modifying original
        transformed_headers = headers.copy()

        # Remove any headers that might cause issues
        headers_to_remove = [
            "content-encoding",  # Let the client handle encoding
            "transfer-encoding",  # FastAPI will handle this
        ]

        for header in headers_to_remove:
            # Case-insensitive removal
            for key in list(transformed_headers.keys()):
                if key.lower() == header.lower():
                    del transformed_headers[key]

        # Update Content-Length if body was transformed
        if body_size is not None and self._is_openai_request(path):
            # Find and update Content-Length header (case-insensitive)
            for key in list(transformed_headers.keys()):
                if key.lower() == "content-length":
                    transformed_headers[key] = str(body_size)
                    logger.debug(
                        f"Updated Content-Length from {headers.get(key, 'unknown')} to {body_size}"
                    )
                    break

        return transformed_headers

    def _is_openai_request(self, path: str) -> bool:
        """Check if request path suggests OpenAI format response needed.

        Args:
            path: Request path

        Returns:
            True if this is an OpenAI format request
        """
        return is_openai_request(path)

    def _transform_anthropic_to_openai_response(self, body: bytes, path: str) -> bytes:
        """Transform Anthropic response to OpenAI format.

        Args:
            body: Anthropic response body
            path: Request path for context

        Returns:
            OpenAI format response body
        """
        try:
            from ccproxy.services.translator import OpenAITranslator

            anthropic_data = json.loads(body.decode("utf-8"))
            translator = OpenAITranslator()

            # Extract model from original request if possible
            # For now, use a default model name
            original_model = "gpt-4o"  # Default fallback

            # Convert Anthropic response to OpenAI format
            openai_data = translator.anthropic_to_openai_response(
                anthropic_data, original_model
            )

            logger.debug(
                f"Converted Anthropic response to OpenAI format: {openai_data}"
            )
            return json.dumps(openai_data).encode("utf-8")

        except Exception as e:
            logger.debug(f"Failed to transform Anthropic response to OpenAI: {e}")
            return body
