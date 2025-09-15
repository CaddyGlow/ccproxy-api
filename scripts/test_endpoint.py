#!/usr/bin/env pyth`on3
"""Test endpoint script converted from test_endpoint.sh with response validation."""

import argparse
import asyncio
import json
import logging
import sys
from dataclasses import dataclass
from typing import Any

import httpx
import structlog

# Import typed models from ccproxy/llms/
from ccproxy.llms.anthropic.models import (
    MessageResponse as AnthropicMessageResponse,
)
from ccproxy.llms.anthropic.models import (
    MessageStartEvent as AnthropicMessageStartEvent,
)
from ccproxy.llms.openai.models import (
    BaseStreamEvent as OpenAIBaseStreamEvent,
)
from ccproxy.llms.openai.models import (
    ChatCompletionChunk as OpenAIChatCompletionChunk,
)
from ccproxy.llms.openai.models import (
    ChatCompletionResponse as OpenAIChatCompletionResponse,
)
from ccproxy.llms.openai.models import (
    ResponseMessage as OpenAIResponseMessage,
)
from ccproxy.llms.openai.models import (
    ResponseObject as OpenAIResponseObject,
)


# Configure structlog similar to the codebase pattern
logger = structlog.get_logger(__name__)


# ANSI color codes
class Colors:
    """ANSI color codes for terminal output."""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    CYAN = "\033[36m"
    MAGENTA = "\033[35m"
    YELLOW = "\033[33m"
    GREEN = "\033[32m"
    RED = "\033[31m"
    BLUE = "\033[34m"


def colored_header(title: str) -> str:
    """Create a colored header similar to the bash script."""
    return (
        f"\n\n{Colors.BOLD}{Colors.CYAN}########## {title} ##########{Colors.RESET}\n"
    )


def colored_success(text: str) -> str:
    """Color text as success (green)."""
    return f"{Colors.GREEN}{text}{Colors.RESET}"


def colored_error(text: str) -> str:
    """Color text as error (red)."""
    return f"{Colors.RED}{text}{Colors.RESET}"


def colored_info(text: str) -> str:
    """Color text as info (blue)."""
    return f"{Colors.BLUE}{text}{Colors.RESET}"


def colored_warning(text: str) -> str:
    """Color text as warning (yellow)."""
    return f"{Colors.YELLOW}{text}{Colors.RESET}"


@dataclass()
class EndpointTest:
    """Configuration for a single endpoint test."""

    name: str
    endpoint: str
    stream: bool
    request: str  # Key in request_data
    model: str
    description: str = ""

    def __post_init__(self):
        if not self.description:
            stream_str = "streaming" if self.stream else "non-streaming"
            self.description = f"{self.name} ({stream_str})"


# Centralized message payloads per provider
MESSAGE_PAYLOADS = {
    "openai": [{"role": "user", "content": "Hello"}],
    "anthropic": [{"role": "user", "content": "Hello"}],
    "response_api": [
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "Hello"}],
        }
    ],
}

# Request payload templates with model_class for validation
REQUEST_DATA = {
    "openai_stream": {
        "model": "{model}",
        "messages": MESSAGE_PAYLOADS["openai"],
        "max_tokens": 100,
        "stream": True,
        "model_class": OpenAIChatCompletionResponse,
        "chunk_model_class": OpenAIChatCompletionChunk,  # For SSE chunk validation
    },
    "openai_non_stream": {
        "model": "{model}",
        "messages": MESSAGE_PAYLOADS["openai"],
        "max_tokens": 100,
        "stream": False,
        "model_class": OpenAIChatCompletionResponse,
    },
    "response_api_stream": {
        "model": "{model}",
        "stream": True,
        "max_completion_tokens": 1000,
        "input": MESSAGE_PAYLOADS["response_api"],
        # For Responses API streaming, chunks are SSE events with event+data
        "model_class": OpenAIBaseStreamEvent,
        "chunk_model_class": OpenAIBaseStreamEvent,
    },
    "response_api_non_stream": {
        "model": "{model}",
        "stream": False,
        "max_completion_tokens": 1000,
        "input": MESSAGE_PAYLOADS["response_api"],
        # Validate the assistant message payload using OpenAIResponseObject
        "model_class": OpenAIResponseObject,
    },
    "anthropic_stream": {
        "model": "{model}",
        "max_tokens": 1000,
        "stream": True,
        "messages": MESSAGE_PAYLOADS["anthropic"],
        "model_class": AnthropicMessageResponse,
        "chunk_model_class": AnthropicMessageStartEvent,
    },
    "anthropic_non_stream": {
        "model": "{model}",
        "max_tokens": 1000,
        "stream": False,
        "messages": MESSAGE_PAYLOADS["anthropic"],
        "model_class": AnthropicMessageResponse,
    },
}


# Endpoint configurations (no response_type, type is in REQUEST_DATA)
ENDPOINT_TESTS = [
    EndpointTest(
        name="copilot_chat_completions",
        endpoint="/copilot/v1/chat/completions",
        stream=True,
        request="openai_stream",
        model="gpt-4o",
        description="Copilot chat completions streaming",
    ),
    EndpointTest(
        name="copilot_chat_completions",
        endpoint="/copilot/v1/chat/completions",
        stream=False,
        request="openai_non_stream",
        model="gpt-4o",
        description="Copilot chat completions non-streaming",
    ),
    EndpointTest(
        name="copilot_responses",
        endpoint="/copilot/v1/responses",
        stream=True,
        request="response_api_stream",
        model="gpt-4o",
        description="Copilot responses streaming",
    ),
    EndpointTest(
        name="copilot_responses",
        endpoint="/copilot/v1/responses",
        stream=False,
        request="response_api_non_stream",
        model="gpt-4o",
        description="Copilot responses non-streaming",
    ),
    EndpointTest(
        name="anthropic_api_openai",
        endpoint="/api/v1/chat/completions",
        stream=True,
        request="openai_stream",
        model="claude-sonnet-4-20250514",
        description="Claude API OpenAI format streaming",
    ),
    EndpointTest(
        name="anthropic_api_openai",
        endpoint="/api/v1/chat/completions",
        stream=False,
        request="openai_non_stream",
        model="claude-sonnet-4-20250514",
        description="Claude API OpenAI format non-streaming",
    ),
    EndpointTest(
        name="anthropic_api_responses",
        endpoint="/api/v1/responses",
        stream=True,
        request="response_api_stream",
        model="claude-sonnet-4-20250514",
        description="Claude API Response format streaming",
    ),
    EndpointTest(
        name="anthropic_api_responses",
        endpoint="/api/v1/responses",
        stream=False,
        request="response_api_non_stream",
        model="claude-sonnet-4-20250514",
        description="Claude API Response format non-streaming",
    ),
    EndpointTest(
        name="codex_chat_completions",
        endpoint="/api/codex/v1/chat/completions",
        stream=True,
        request="openai_stream",
        model="gpt-5",
        description="Codex chat completions streaming",
    ),
    EndpointTest(
        name="codex_chat_completions",
        endpoint="/api/codex/v1/chat/completions",
        stream=False,
        request="openai_non_stream",
        model="gpt-5",
        description="Codex chat completions non-streaming",
    ),
]


def get_request_payload(test: EndpointTest) -> dict[str, Any]:
    """Get formatted request payload for a test, excluding validation classes."""
    template = REQUEST_DATA[test.request].copy()

    # Remove validation classes from the payload - they shouldn't be sent to server
    validation_keys = {"model_class", "chunk_model_class"}
    template = {k: v for k, v in template.items() if k not in validation_keys}

    def format_value(value):
        if isinstance(value, str):
            return value.format(model=test.model)
        elif isinstance(value, dict):
            return {k: format_value(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [format_value(item) for item in value]
        return value

    return format_value(template)


class TestEndpoint:
    """Test endpoint utility for CCProxy API testing."""

    def __init__(self, base_url: str = "http://127.0.0.1:8000", trace: bool = False):
        self.base_url = base_url
        self.trace = trace
        self.client = httpx.AsyncClient(timeout=30.0)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()

    # No registry needed; validation type is in payload

    async def post_json(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Post JSON request and return parsed response."""
        headers = {"Content-Type": "application/json"}

        print(colored_info(f"→ Making JSON request to {url}"))
        logger.info(
            "Making JSON request",
            url=url,
            payload_model=payload.get("model"),
            payload_stream=payload.get("stream"),
        )

        response = await self.client.post(url, json=payload, headers=headers)

        logger.info(
            "Received JSON response",
            status_code=response.status_code,
            headers=dict(response.headers),
        )

        if response.status_code != 200:
            print(colored_error(f"✗ Request failed: HTTP {response.status_code}"))
            logger.error(
                "Request failed",
                status_code=response.status_code,
                response_text=response.text,
            )
            return {"error": f"HTTP {response.status_code}: {response.text}"}

        try:
            return response.json()
        except json.JSONDecodeError as e:
            logger.error("Failed to parse JSON response", error=str(e))
            return {"error": f"JSON decode error: {e}"}

    async def post_stream(self, url: str, payload: dict[str, Any]) -> list[str]:
        """Post streaming request and return list of SSE events."""
        headers = {
            "Accept": "text/event-stream",
            "Content-Type": "application/json",
        }

        print(colored_info(f"→ Making streaming request to {url}"))
        logger.info(
            "Making streaming request",
            url=url,
            payload_model=payload.get("model"),
            payload_stream=payload.get("stream"),
        )

        events = []
        try:
            async with self.client.stream(
                "POST", url, json=payload, headers=headers
            ) as response:
                logger.info(
                    "Received streaming response",
                    status_code=response.status_code,
                    headers=dict(response.headers),
                )

                if response.status_code != 200:
                    error_text = await response.aread()
                    print(
                        colored_error(
                            f"✗ Streaming request failed: HTTP {response.status_code}"
                        )
                    )
                    logger.error(
                        "Streaming request failed",
                        status_code=response.status_code,
                        response_text=error_text.decode(),
                    )
                    return [
                        f"error: HTTP {response.status_code}: {error_text.decode()}"
                    ]

                async for chunk in response.aiter_text():
                    if chunk.strip():
                        events.append(chunk.strip())

        except Exception as e:
            logger.error("Streaming request exception", error=str(e))
            events.append(f"error: {e}")

        logger.info("Streaming completed", event_count=len(events))
        return events

    def validate_response(
        self, response: dict[str, Any], model_class, is_streaming: bool = False
    ) -> bool:
        """Validate response using the provided model_class."""
        try:
            payload = response
            # Special handling for OpenAIResponseMessage: extract assistant message
            if model_class is OpenAIResponseMessage:
                payload = self._extract_openai_response_message(response)
            model_class.model_validate(payload)
            print(colored_success(f"✓ {model_class.__name__} validation passed"))
            logger.info(f"{model_class.__name__} validation passed")
            return True
        except Exception as e:
            print(colored_error(f"✗ {model_class.__name__} validation failed: {e}"))
            logger.error(f"{model_class.__name__} validation failed", error=str(e))
            return False

    def _extract_openai_response_message(
        self, response: dict[str, Any]
    ) -> dict[str, Any]:
        """Coerce various response shapes into an OpenAIResponseMessage dict.

        Supports:
        - Chat Completions: { choices: [{ message: {...} }] }
        - Responses API (non-stream): { output: [ { type: 'message', content: [...] } ] }
        """
        # Case 1: Chat Completions format
        try:
            if isinstance(response, dict) and "choices" in response:
                choices = response.get("choices") or []
                if choices and isinstance(choices[0], dict):
                    msg = choices[0].get("message")
                    if isinstance(msg, dict):
                        return msg
        except Exception:
            pass

        # Case 2: Responses API-like format with output message
        try:
            output = response.get("output") if isinstance(response, dict) else None
            if isinstance(output, list):
                for item in output:
                    if isinstance(item, dict) and item.get("type") == "message":
                        content_blocks = item.get("content") or []
                        text_parts: list[str] = []
                        for block in content_blocks:
                            if (
                                isinstance(block, dict)
                                and block.get("type") in ("text", "output_text")
                                and block.get("text")
                            ):
                                text_parts.append(block["text"])
                        content_text = "".join(text_parts) if text_parts else None
                        return {"role": "assistant", "content": content_text}
        except Exception:
            pass

        # Fallback: empty assistant message
        return {"role": "assistant", "content": None}

    def validate_sse_event(self, event: str) -> bool:
        """Validate SSE event structure (basic check)."""
        return event.startswith("data: ")

    def validate_stream_chunk(self, chunk: dict[str, Any], chunk_model_class) -> bool:
        """Validate a streaming chunk using the provided chunk_model_class."""
        try:
            chunk_model_class.model_validate(chunk)
            print(
                colored_success(
                    f"✓ {chunk_model_class.__name__} chunk validation passed"
                )
            )
            return True
        except Exception as e:
            print(
                colored_error(
                    f"✗ {chunk_model_class.__name__} chunk validation failed: {e}"
                )
            )
            return False

    async def run_endpoint_test(self, test: EndpointTest):
        """Run a single endpoint test based on configuration."""
        full_url = f"{self.base_url}{test.endpoint}"
        payload = get_request_payload(test)

        # Get validation classes from original template
        template = REQUEST_DATA[test.request]
        model_class = template.get("model_class")
        chunk_model_class = template.get("chunk_model_class")

        logger.info(
            "Running endpoint test",
            name=test.name,
            endpoint=test.endpoint,
            stream=test.stream,
            model_class=getattr(model_class, "__name__", None) if model_class else None,
        )

        print(colored_header(test.description))

        if test.stream:
            # Streaming test
            stream_events = await self.post_stream(full_url, payload)

            # Track last SSE event name for Responses API
            last_event_name: str | None = None

            # Print and validate streaming events
            for event in stream_events:
                print(event)

                # Capture SSE event name lines
                if event.startswith("event: "):
                    last_event_name = event[len("event: ") :].strip()
                    continue

                if self.validate_sse_event(event) and not event.endswith("[DONE]"):
                    try:
                        data = json.loads(event[6:])  # Remove "data: " prefix
                        if chunk_model_class:
                            # If validating Responses API SSE events, wrap with event name
                            if chunk_model_class is OpenAIBaseStreamEvent:
                                wrapped = {"event": last_event_name, "data": data}
                                self.validate_stream_chunk(wrapped, chunk_model_class)
                            else:
                                # Skip Copilot prelude chunks lacking required fields
                                if chunk_model_class is OpenAIChatCompletionChunk and (
                                    not isinstance(data, dict)
                                    or not data.get("model")
                                    or not data.get("choices")
                                ):
                                    logger.info(
                                        "Skipping non-standard prelude chunk",
                                        has_model=data.get("model")
                                        if isinstance(data, dict)
                                        else False,
                                        has_choices=bool(data.get("choices"))
                                        if isinstance(data, dict)
                                        else False,
                                    )
                                else:
                                    self.validate_stream_chunk(data, chunk_model_class)
                        # elif model_class:
                        #     self.validate_response(data, model_class, is_streaming=True)
                    except json.JSONDecodeError:
                        logger.warning("Invalid JSON in streaming event", event=event)
        else:
            # Non-streaming test
            response = await self.post_json(full_url, payload)

            print(json.dumps(response, indent=2))
            if "error" not in response and model_class:
                self.validate_response(response, model_class, is_streaming=False)

    async def run_all_tests(self):
        """Run all endpoint tests."""
        print(colored_header("CCProxy Endpoint Tests"))
        print(colored_info(f"Testing endpoints at {self.base_url}"))
        logger.info("Starting all endpoint tests", base_url=self.base_url)

        print(colored_info(f"Running {len(ENDPOINT_TESTS)} configured tests"))
        logger.info(
            "Starting all endpoint tests",
            base_url=self.base_url,
            test_count=len(ENDPOINT_TESTS),
        )

        # Run all configured tests
        for test in ENDPOINT_TESTS:
            await self.run_endpoint_test(test)

        print(
            colored_success(f"\n🎉 All {len(ENDPOINT_TESTS)} endpoint tests completed!")
        )
        logger.info("All endpoint tests completed")


def setup_logging(level: str = "warn") -> None:
    """Setup structured logging with specified level."""
    log_level_map = {
        "warn": logging.WARNING,
        "info": logging.INFO,
        "debug": logging.DEBUG,
        "error": logging.ERROR,
    }

    # Configure basic logging for structlog
    logging.basicConfig(
        level=log_level_map.get(level, logging.WARNING),
        format="%(message)s",
    )

    # Configure structlog with console renderer for pretty output
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.CallsiteParameterAdder(
                parameters=[
                    structlog.processors.CallsiteParameter.FILENAME,
                    structlog.processors.CallsiteParameter.LINENO,
                ]
            ),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.dev.ConsoleRenderer(colors=True),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Test CCProxy endpoints with response validation"
    )
    parser.add_argument(
        "--base",
        default="http://127.0.0.1:8000",
        help="Base URL for the API server (default: http://127.0.0.1:8000)",
    )
    parser.add_argument(
        "-v",
        action="store_true",
        help="Set log level to INFO",
    )
    parser.add_argument(
        "-vv",
        action="store_true",
        help="Set log level to DEBUG",
    )
    parser.add_argument(
        "-vvv",
        action="store_true",
        help="Set log level to DEBUG (same as -vv)",
    )
    parser.add_argument(
        "--log-level",
        choices=["warn", "info", "debug", "error"],
        default="warn",
        help="Set log level explicitly (default: warn)",
    )

    args = parser.parse_args()

    # Determine final log level
    log_level = args.log_level
    if args.v:
        log_level = "info"
    elif args.vv or args.vvv:
        log_level = "debug"

    setup_logging(log_level)

    async def run_tests():
        async with TestEndpoint(base_url=args.base) as tester:
            await tester.run_all_tests()

    try:
        asyncio.run(run_tests())
    except KeyboardInterrupt:
        logger.info("Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error("Test execution failed", error=str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
