"""Response processing utilities for testing."""

import json
from typing import Any

import httpx

from ccproxy.testing.config import RequestScenario


class ResponseHandler:
    """Handle responses from both Anthropic and OpenAI formats."""

    def process_response(
        self, response: httpx.Response, scenario: RequestScenario
    ) -> dict[str, Any]:
        """Process response based on format and streaming."""

        if scenario.streaming:
            return self._process_streaming_response(response, scenario)
        else:
            return self._process_standard_response(response, scenario)

    def _process_standard_response(
        self, response: httpx.Response, scenario: RequestScenario
    ) -> dict[str, Any]:
        """Process non-streaming response."""

        try:
            response_data = response.json()

            # Extract metrics based on format
            if scenario.api_format == "openai":
                tokens_input = response_data.get("usage", {}).get("prompt_tokens")
                tokens_output = response_data.get("usage", {}).get("completion_tokens")
                content = (
                    response_data.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content")
                )
            else:  # anthropic
                usage = response_data.get("usage", {})
                tokens_input = usage.get("input_tokens")
                tokens_output = usage.get("output_tokens")
                content = ""
                for block in response_data.get("content", []):
                    if block.get("type") == "text":
                        content += block.get("text", "")

            return {
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "data": response_data,
                "tokens_input": tokens_input,
                "tokens_output": tokens_output,
                "content_preview": content[:100] if content else None,
                "format": scenario.api_format,
            }

        except json.JSONDecodeError as e:
            return {
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "error": f"Failed to parse {scenario.api_format} JSON response: {str(e)}",
                "raw_text": response.text[:500] if hasattr(response, "text") else "",
            }
        except (OSError, PermissionError) as e:
            return {
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "error": f"IO/Permission error parsing {scenario.api_format} response: {str(e)}",
                "raw_text": response.text[:500] if hasattr(response, "text") else "",
            }
        except Exception as e:
            return {
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "error": f"Failed to parse {scenario.api_format} response: {str(e)}",
                "raw_text": response.text[:500] if hasattr(response, "text") else "",
            }

    def _process_streaming_response(
        self, response: httpx.Response, scenario: RequestScenario
    ) -> dict[str, Any]:
        """Process streaming response."""

        chunks = []
        total_content = ""

        try:
            for line in response.iter_lines():
                if line.startswith("data: "):
                    data_str = line[6:].strip()
                    if data_str and data_str != "[DONE]":
                        try:
                            chunk_data = json.loads(data_str)
                            chunks.append(chunk_data)

                            # Extract content based on format
                            if scenario.api_format == "openai":
                                delta_content = (
                                    chunk_data.get("choices", [{}])[0]
                                    .get("delta", {})
                                    .get("content", "")
                                )
                                total_content += delta_content
                            else:  # anthropic
                                if chunk_data.get("type") == "content_block_delta":
                                    delta_text = chunk_data.get("delta", {}).get(
                                        "text", ""
                                    )
                                    total_content += delta_text
                        except json.JSONDecodeError:
                            continue

            return {
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "chunks": chunks,
                "chunk_count": len(chunks),
                "total_content": total_content,
                "content_preview": total_content[:100] if total_content else None,
                "format": scenario.api_format,
            }

        except (OSError, PermissionError) as e:
            return {
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "error": f"IO/Permission error processing {scenario.api_format} stream: {str(e)}",
            }
        except Exception as e:
            return {
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "error": f"Failed to process {scenario.api_format} stream: {str(e)}",
            }


class MetricsExtractor:
    """Extract metrics from API responses."""

    @staticmethod
    def extract_token_metrics(
        response_data: dict[str, Any], api_format: str
    ) -> dict[str, int | None]:
        """Extract token usage from response data."""
        if api_format == "openai":
            usage = response_data.get("usage", {})
            return {
                "input_tokens": usage.get("prompt_tokens"),
                "output_tokens": usage.get("completion_tokens"),
                "cache_read_tokens": None,  # OpenAI doesn't expose cache metrics
                "cache_write_tokens": None,
            }
        else:  # anthropic
            usage = response_data.get("usage", {})
            return {
                "input_tokens": usage.get("input_tokens"),
                "output_tokens": usage.get("output_tokens"),
                "cache_read_tokens": usage.get("cache_read_input_tokens"),
                "cache_write_tokens": usage.get("cache_creation_input_tokens"),
            }

    @staticmethod
    def extract_content(response_data: dict[str, Any], api_format: str) -> str:
        """Extract text content from response data."""
        if api_format == "openai":
            content = (
                response_data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )
            return content if isinstance(content, str) else ""
        else:  # anthropic
            content = ""
            for block in response_data.get("content", []):
                if block.get("type") == "text":
                    text = block.get("text", "")
                    content += text if isinstance(text, str) else ""
            return content
