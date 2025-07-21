#!/usr/bin/env python3
"""
OpenAI SDK Streaming Demonstration (Refactored)

This script demonstrates how to use streaming responses with the OpenAI SDK
(pointing to Claude via proxy), showing real-time token streaming.
"""

import argparse
import os

import openai
from common_utils import LoggingSyncClient, setup_logging
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionUserMessageParam
from structlog import get_logger


logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="OpenAI SDK Streaming Demonstration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 openai_streaming_demo.py
  python3 openai_streaming_demo.py -v
  python3 openai_streaming_demo.py -vv
        """,
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase verbosity (-v=INFO, -vv=DEBUG).",
    )
    return parser.parse_args()


def main() -> None:
    """
    Main demonstration function.
    """
    args = parse_args()
    setup_logging(verbose=args.verbose)

    print("=== OpenAI SDK Streaming Demonstration ===")

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

    try:
        http_client = LoggingSyncClient()
        client = openai.OpenAI(
            http_client=http_client,
        )

        messages: list[ChatCompletionMessageParam] = [
            ChatCompletionUserMessageParam(
                role="user",
                content="Write a short story about a robot learning to paint. Make it creative and engaging.",
            )
        ]

        print(f"\n{'=' * 25} Turn 1 {'=' * 25}\n")
        print(f"\nUser Message:\n{messages[0]['content']}")

        print("\n--- Starting streaming conversation with Claude via OpenAI API... ---")

        stream = client.chat.completions.create(
            model="gpt-4o",
            max_tokens=1000,
            messages=messages,
            stream=True,
        )

        print("\nAssistant Response (streaming):")
        full_response = ""
        finish_reason = "unknown"

        for chunk in stream:
            logger.info("streaming_chunk", chunk=chunk)
            if (
                chunk.choices
                and chunk.choices[0].delta
                and chunk.choices[0].delta.content
            ):
                content = chunk.choices[0].delta.content
                full_response += content
                print(chunk.choices[0].delta.content, end="", flush=True)
            if chunk.choices and chunk.choices[0].finish_reason:
                finish_reason = chunk.choices[0].finish_reason

        print()  # New line after streaming

        logger.info(
            "Streaming finished",
            finish_reason=finish_reason,
            length=len(full_response),
        )

    except Exception as e:
        logger.error("error", exc_info=True)
        print(f"Error: {str(e)}")
        print("Make sure your proxy server is running on http://127.0.0.1:8000")


if __name__ == "__main__":
    main()
