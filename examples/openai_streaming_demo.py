#!/usr/bin/env python3
"""
OpenAI SDK Streaming Demonstration

This script demonstrates how to use streaming responses with the OpenAI SDK
(pointing to Claude via proxy), showing real-time token streaming.
"""

import argparse
import logging
import os
from typing import Any

import httpx
import openai
from httpx import URL
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionUserMessageParam
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
        logging.getLogger("openai").setLevel(logging.DEBUG)
    else:
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("openai").setLevel(logging.WARNING)


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
        description="OpenAI SDK Streaming Demonstration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 openai_streaming_demo.py
  python3 openai_streaming_demo.py --debug
  python3 openai_streaming_demo.py -d
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

    print("OpenAI SDK Streaming Demonstration")
    print("=" * 40)
    if args.debug:
        print("Debug logging enabled")
        print("=" * 40)

    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    base_url_default = "http://127.0.0.1:8000"

    if not api_key:
        logger.warning(
            "api_key_missing", message="OPENAI_API_KEY not set, using dummy key"
        )
        os.environ["OPENAI_API_KEY"] = "dummy"
    if not base_url:
        logger.warning(
            "base_url_missing",
            message="OPENAI_BASE_URL not set",
            default_url=base_url_default,
        )
        os.environ["OPENAI_BASE_URL"] = base_url_default

    # Initialize OpenAI client with custom HTTP client
    try:
        http_client = LoggingHTTPClient()
        client = openai.OpenAI(
            http_client=http_client,
        )
        logger.info(
            "openai_client_initialized",
            message="Client initialized with logging HTTP client",
        )

        # Example streaming conversation
        messages: list[ChatCompletionMessageParam] = [
            ChatCompletionUserMessageParam(
                role="user",
                content="Write a short story about a robot learning to paint. Make it creative and engaging.",
            )
        ]

        print("\n" + "=" * 40)
        print("Starting streaming conversation with Claude via OpenAI API...")
        print("=" * 40)

        logger.info("claude_streaming_request_start")
        logger.info(
            "initial_message", content=getattr(messages[0], "content", "No content")
        )

        # Log the complete request structure
        logger.info(
            "request_structure",
            model="gpt-4o",
            max_tokens=1000,
            stream=True,
            messages=[
                msg.model_dump() if hasattr(msg, "model_dump") else msg
                for msg in messages
            ],
        )

        print("\nClaude's streaming response:")
        print("-" * 40)

        # Create streaming response
        stream = client.chat.completions.create(
            model="gpt-4o",  # Will be mapped to Claude by proxy
            max_tokens=1000,
            messages=messages,
            stream=True,
        )

        full_response = ""
        chunk_count = 0

        # Process streaming chunks
        for chunk in stream:
            chunk_count += 1
            logger.debug("stream_chunk_received", chunk_number=chunk_count)

            if chunk.choices:
                choice = chunk.choices[0]

                # Log chunk details
                logger.debug(
                    "chunk_details",
                    chunk_id=chunk.id,
                    finish_reason=choice.finish_reason,
                    has_content=bool(choice.delta.content),
                )

                # Handle content delta
                if choice.delta.content:
                    content = choice.delta.content
                    print(content, end="", flush=True)
                    full_response += content

                    logger.debug("content_delta", content=content)

                # Handle finish reason
                if choice.finish_reason:
                    logger.info(
                        "stream_finished",
                        finish_reason=choice.finish_reason,
                        total_chunks=chunk_count,
                        response_length=len(full_response),
                    )
                    print(f"\n\nStream finished with reason: {choice.finish_reason}")
                    break

        print("\n" + "=" * 40)
        print(f"Complete response ({len(full_response)} characters):")
        print("=" * 40)
        print(full_response)

        logger.info(
            "streaming_session_complete",
            total_chunks=chunk_count,
            final_response_length=len(full_response),
        )

    except Exception as e:
        print(f"\nError: {e}")
        print("Make sure your proxy server is running on http://127.0.0.1:8000")
        logger.error("streaming_error", error=str(e))


if __name__ == "__main__":
    main()
