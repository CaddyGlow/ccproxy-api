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
    MessageResponse,
    MessageStartEvent,
)
from ccproxy.llms.openai.models import (
    BaseStreamEvent,
    ChatCompletionChunk,
    ChatCompletionResponse,
    ResponseMessage,
    ResponseObject,
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
        "model_class": ChatCompletionResponse,
        "chunk_model_class": ChatCompletionChunk,  # For SSE chunk validation
    },
    "openai_non_stream": {
        "model": "{model}",
        "messages": MESSAGE_PAYLOADS["openai"],
        "max_tokens": 100,
        "stream": False,
        "model_class": ChatCompletionResponse,
    },
    "response_api_stream": {
        "model": "{model}",
        "stream": True,
        "max_completion_tokens": 1000,
        "input": MESSAGE_PAYLOADS["response_api"],
        # For Responses API streaming, chunks are SSE events with event+data
        "model_class": BaseStreamEvent,
        "chunk_model_class": BaseStreamEvent,
    },
    "response_api_non_stream": {
        "model": "{model}",
        "stream": False,
        "max_completion_tokens": 1000,
        "input": MESSAGE_PAYLOADS["response_api"],
        # Validate the assistant message payload using ResponseObject
        "model_class": ResponseObject,
    },
    "anthropic_stream": {
        "model": "{model}",
        "max_tokens": 1000,
        "stream": True,
        "messages": MESSAGE_PAYLOADS["anthropic"],
        "model_class": MessageResponse,
        "chunk_model_class": MessageStartEvent,
    },
    "anthropic_non_stream": {
        "model": "{model}",
        "max_tokens": 1000,
        "stream": False,
        "messages": MESSAGE_PAYLOADS["anthropic"],
        "model_class": MessageResponse,
    },
}


# Endpoint configurations (no response_type, type is in REQUEST_DATA)
ENDPOINT_TESTS = [
    EndpointTest(
        name="copilot_chat_completions_stream",
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
        name="copilot_responsess_stream",
        endpoint="/copilot/v1/responses",
        stream=True,
        request="response_api_stream",
        model="gpt-4o",
        description="Copilot responses streaming",
    ),
    EndpointTest(
        name="copilot_responses_stream",
        endpoint="/copilot/v1/responses",
        stream=False,
        request="response_api_non_stream",
        model="gpt-4o",
        description="Copilot responses non-streaming",
    ),
    EndpointTest(
        name="anthropic_api_openai_stream",
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
        name="anthropic_api_responses_stream",
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
        name="codex_chat_completions_stream",
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
            # Special handling for ResponseMessage: extract assistant message
            if model_class is ResponseMessage:
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
                            if chunk_model_class is BaseStreamEvent:
                                wrapped = {"event": last_event_name, "data": data}
                                self.validate_stream_chunk(wrapped, chunk_model_class)
                            else:
                                # Skip Copilot prelude chunks lacking required fields
                                if chunk_model_class is ChatCompletionChunk and (
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

    async def run_all_tests(self, selected_indices: list[int] | None = None):
        """Run endpoint tests, optionally filtered by selected indices."""
        print(colored_header("CCProxy Endpoint Tests"))
        print(colored_info(f"Testing endpoints at {self.base_url}"))
        logger.info("Starting endpoint tests", base_url=self.base_url)

        # Filter tests if selection provided
        tests_to_run = ENDPOINT_TESTS
        if selected_indices is not None:
            tests_to_run = [
                ENDPOINT_TESTS[i]
                for i in selected_indices
                if 0 <= i < len(ENDPOINT_TESTS)
            ]
            print(
                colored_info(
                    f"Running {len(tests_to_run)} selected tests (out of {len(ENDPOINT_TESTS)} total)"
                )
            )
            logger.info(
                "Running selected tests",
                selected_count=len(tests_to_run),
                total_count=len(ENDPOINT_TESTS),
                selected_indices=selected_indices,
            )
        else:
            print(colored_info(f"Running all {len(ENDPOINT_TESTS)} configured tests"))
            logger.info(
                "Running all tests",
                test_count=len(ENDPOINT_TESTS),
            )

        # Run selected tests
        for i, test in enumerate(tests_to_run, 1):
            if selected_indices is not None:
                # Show original test number when running subset
                original_index = ENDPOINT_TESTS.index(test) + 1
                print(
                    colored_info(
                        f"[Test {i}/{len(tests_to_run)}] #{original_index}: {test.description}"
                    )
                )
            await self.run_endpoint_test(test)

        print(colored_success(f"\n🎉 {len(tests_to_run)} endpoint tests completed!"))
        logger.info("Endpoint tests completed", completed_count=len(tests_to_run))


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


def parse_test_selection(selection: str, total_tests: int) -> list[int]:
    """Parse test selection string into list of test indices (0-based).

    Supports:
    - Single numbers: "1" -> [0]
    - Comma-separated: "1,3,5" -> [0,2,4]
    - Ranges: "1..3" -> [0,1,2]
    - Open ranges: "4.." -> [3,4,5,...]
    - Prefix ranges: "..3" -> [0,1,2]
    - Mixed: "1,3..5,7" -> [0,2,3,4,6]
    """
    indices = set()

    for part in selection.split(","):
        part = part.strip()

        if ".." in part:
            # Range syntax
            if part.startswith(".."):
                # ..3 means 1 to 3
                end = int(part[2:])
                indices.update(range(0, end))
            elif part.endswith(".."):
                # 4.. means 4 to end
                start = int(part[:-2]) - 1  # Convert to 0-based
                indices.update(range(start, total_tests))
            else:
                # 1..3 means 1 to 3
                start_str, end_str = part.split("..", 1)
                start = int(start_str) - 1  # Convert to 0-based
                end = int(end_str)
                indices.update(range(start, end))
        else:
            # Single number
            index = int(part) - 1  # Convert to 0-based
            if 0 <= index < total_tests:
                indices.add(index)

    return sorted(indices)


def list_available_tests() -> str:
    """Generate a formatted list of available tests for help text."""
    lines = ["Available tests:"]
    for i, test in enumerate(ENDPOINT_TESTS, 1):
        lines.append(f"  {i:2d}. {test.description}")
    return "\n".join(lines)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Test CCProxy endpoints with response validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
{list_available_tests()}

Test selection examples:
  --tests 1          # Run test 1 only
  --tests 1,3,5      # Run tests 1, 3, and 5
  --tests 1..3       # Run tests 1 through 3
  --tests 4..        # Run tests 4 through end
  --tests ..3        # Run tests 1 through 3
  --tests 1,4..6,8   # Run test 1, tests 4-6, and test 8
""",
    )
    parser.add_argument(
        "--base",
        default="http://127.0.0.1:8000",
        help="Base URL for the API server (default: http://127.0.0.1:8000)",
    )
    parser.add_argument(
        "--tests",
        help="Select specific tests to run (e.g., 1,2,3 or 1..3 or 4.. or ..3)",
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

    # Parse test selection if provided
    selected_indices = None
    if args.tests:
        try:
            selected_indices = parse_test_selection(args.tests, len(ENDPOINT_TESTS))
            if not selected_indices:
                logger.error("No valid tests selected")
                sys.exit(1)
        except ValueError as e:
            logger.error(
                "Invalid test selection format", selection=args.tests, error=str(e)
            )
            sys.exit(1)

    async def run_tests():
        async with TestEndpoint(base_url=args.base) as tester:
            await tester.run_all_tests(selected_indices)

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
