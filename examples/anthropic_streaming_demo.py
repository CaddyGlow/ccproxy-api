#!/usr/bin/env python3
"""
Anthropic SDK Streaming Demonstration

This script demonstrates how to use streaming responses with the Anthropic SDK,
showing real-time token streaming and event handling.
"""

import argparse
import logging
import os
from typing import Any

import anthropic
import httpx
from anthropic.types import MessageParam
from httpx import URL
from structlog import get_logger


def setup_logging(debug: bool = False) -> None:
    """Setup logging configuration.

    Args:
        debug: Whether to enable debug logging
    """
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Set levels for external libraries
    if debug:
        logging.getLogger("httpx").setLevel(logging.DEBUG)
        logging.getLogger("anthropic").setLevel(logging.DEBUG)
    else:
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("anthropic").setLevel(logging.WARNING)


logger = get_logger(__name__)


class LoggingHTTPClient(httpx.Client):
    """Custom HTTP client that logs requests and responses"""

    def request(self, method: str, url: URL | str, **kwargs: Any) -> httpx.Response:
        logger.info("http_request_start")
        logger.info(
            "http_request_details",
            method=method,
            url=str(url),
            headers=kwargs.get("headers", {}),
        )
        if "content" in kwargs:
            try:
                content = kwargs["content"]
                if isinstance(content, bytes):
                    content = content.decode("utf-8")
                logger.info("http_request_body", body=content)
            except Exception as e:
                logger.info("http_request_body_decode_error", error=str(e))

        response = super().request(method, url, **kwargs)

        logger.info(
            "http_response_start",
            status_code=response.status_code,
            headers=dict(response.headers),
        )
        return response


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Anthropic SDK Streaming Demonstration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 anthropic_streaming_demo.py
  python3 anthropic_streaming_demo.py --debug
  python3 anthropic_streaming_demo.py -d
        """,
    )
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="Enable debug logging (shows HTTP requests/responses)",
    )
    return parser.parse_args()


def main() -> None:
    """
    Main demonstration function.
    """
    args = parse_args()
    setup_logging(debug=args.debug)

    print("Anthropic SDK Streaming Demonstration")
    print("=" * 40)
    if args.debug:
        print("Debug logging enabled")
        print("=" * 40)

    api_key = os.getenv("ANTHROPIC_API_KEY")
    base_url = os.getenv("ANTHROPIC_BASE_URL")
    base_url_default = "http://127.0.0.1:8000"

    if not api_key:
        logger.warning(
            "api_key_missing", message="ANTHROPIC_API_KEY not set, using dummy key"
        )
        os.environ["ANTHROPIC_API_KEY"] = "dummy"
    if not base_url:
        logger.warning(
            "base_url_missing",
            message="ANTHROPIC_BASE_URL not set",
            default_url=base_url_default,
        )
        os.environ["ANTHROPIC_BASE_URL"] = base_url_default

    # Initialize Anthropic client with custom HTTP client
    try:
        http_client = LoggingHTTPClient()
        client = anthropic.Anthropic(http_client=http_client)
        logger.info(
            "anthropic_client_initialized",
            message="Client initialized with logging HTTP client",
        )

        # Example streaming conversation
        messages: list[MessageParam] = [
            {
                "role": "user",
                "content": "Write a short story about a robot learning to paint. Make it creative and engaging.",
            }
        ]

        print("\n" + "=" * 40)
        print("Starting streaming conversation with Claude...")
        print("=" * 40)

        logger.info("claude_streaming_request_start")
        logger.info("initial_message", content=messages[0]["content"])

        # Log the complete request structure
        logger.info(
            "request_structure",
            model="claude-3-5-sonnet-20241022",
            max_tokens=1000,
            stream=True,
            messages=messages,
        )

        print("\nClaude's streaming response:")
        print("-" * 40)

        # Create streaming response
        stream = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1000,
            messages=messages,
            stream=True,
        )

        full_response = ""
        event_count = 0

        # Process streaming events
        for event in stream:
            event_count += 1
            logger.debug(
                "stream_event_received", event_number=event_count, event_type=event.type
            )

            # Handle different event types
            if event.type == "message_start":
                logger.info(
                    "message_start_event",
                    message_id=event.message.id,
                    model=event.message.model,
                    stop_reason=event.message.stop_reason,
                    usage=event.message.usage.model_dump()
                    if hasattr(event.message.usage, "model_dump")
                    else dict(event.message.usage)
                    if event.message.usage
                    else None,
                )
                print(f"Message started (ID: {event.message.id})")

            elif event.type == "content_block_start":
                logger.debug(
                    "content_block_start_event",
                    block_index=event.index,
                    block_type=event.content_block.type,
                )

            elif event.type == "content_block_delta":
                if event.delta.type == "text_delta":
                    text = event.delta.text
                    print(text, end="", flush=True)
                    full_response += text

                    logger.debug("text_delta", text=text)

            elif event.type == "content_block_stop":
                logger.debug(
                    "content_block_stop_event",
                    block_index=event.index,
                )

            elif event.type == "message_delta":
                if event.delta.stop_reason:
                    logger.info(
                        "message_delta_event",
                        stop_reason=event.delta.stop_reason,
                        usage=event.usage.model_dump()
                        if hasattr(event.usage, "model_dump")
                        else dict(event.usage)
                        if event.usage
                        else None,
                    )

            elif event.type == "message_stop":
                logger.info(
                    "message_stop_event",
                    total_events=event_count,
                    response_length=len(full_response),
                )
                print("\n\nStream finished")
                break

            elif event.type == "error":
                logger.error("stream_error_event", error=event.error)
                print(f"\nError in stream: {event.error}")
                break

        print("\n" + "=" * 40)
        print(f"Complete response ({len(full_response)} characters):")
        print("=" * 40)
        print(full_response)

        logger.info(
            "streaming_session_complete",
            total_events=event_count,
            final_response_length=len(full_response),
        )

    except Exception as e:
        print(f"\nError: {e}")
        print("Make sure you have the ANTHROPIC_API_KEY environment variable set.")
        logger.error("streaming_error", error=str(e))


if __name__ == "__main__":
    main()
