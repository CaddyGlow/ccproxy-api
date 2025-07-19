#!/usr/bin/env python3
"""
Anthropic SDK Tool Use Demonstration (Refactored)

This script demonstrates how to use tools with the Anthropic SDK, leveraging
a shared library for common functionality.
"""

import argparse
import json
import os

import anthropic
from anthropic.types import MessageParam, ToolParam
from common_utils import (
    LoggingSyncClient,
    calculate_distance,
    generate_json_schema_for_function,
    get_weather,
    handle_tool_call,
    setup_logging,
)
from console_utils import RICH_AVAILABLE, RichConsoleManager
from structlog import get_logger


logger = get_logger(__name__)


def create_anthropic_tools() -> list[ToolParam]:
    """
    Create Anthropic-compatible tool definitions with JSON schemas.

    Returns:
        List of tool definitions
    """
    tools: list[ToolParam] = []

    weather_schema = generate_json_schema_for_function(get_weather)
    tools.append(
        ToolParam(
            name="get_weather",
            description="Get current weather information for a specific location",
            input_schema=weather_schema,
        )
    )

    distance_schema = generate_json_schema_for_function(calculate_distance)
    tools.append(
        ToolParam(
            name="calculate_distance",
            description="Calculate the distance between two geographic coordinates",
            input_schema=distance_schema,
        )
    )

    return tools


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Anthropic SDK Tool Use Demonstration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 anthropic_tools_demo.py
  python3 anthropic_tools_demo.py -v
  python3 anthropic_tools_demo.py -vv
        """,
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase verbosity (-v=INFO, -vv=DEBUG).",
    )
    parser.add_argument(
        "-p", "--plain", action="store_true", help="Disable rich formatting."
    )
    return parser.parse_args()


def main() -> None:
    """
    Main demonstration function.
    """
    args = parse_args()
    setup_logging(verbose=args.verbose)
    console = RichConsoleManager(use_rich=not args.plain)

    console.print_header("Anthropic SDK Tool Use Demonstration")

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

    tools = create_anthropic_tools()
    console.print_tools(tools)

    try:
        http_client = LoggingSyncClient()
        client = anthropic.Anthropic(http_client=http_client)

        messages: list[MessageParam] = [
            {
                "role": "user",
                "content": "What's the weather like in New York, and how far is it from Los Angeles?",
            }
        ]

        console.print_turn_separator(1)
        console.print_user_message(messages[0]["content"])

        console.print_subheader("Starting conversation with Claude...")

        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1000,
            tools=tools,
            messages=messages,
        )

        while True:
            text_content = "".join(
                [c.text for c in response.content if c.type == "text"]
            )
            console.print_response(text_content, response.stop_reason)

            if response.stop_reason == "tool_use":
                tool_results = []

                for content_block in response.content:
                    if content_block.type == "tool_use":
                        tool_name = content_block.name
                        tool_input = (
                            dict(content_block.input) if content_block.input else {}
                        )
                        tool_use_id = content_block.id

                        console.print_tool_call(tool_name, tool_input)
                        result = handle_tool_call(tool_name, tool_input)
                        console.print_tool_result(result)

                        tool_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": tool_use_id,
                                "content": json.dumps(result),
                            }
                        )

                messages.append({"role": "assistant", "content": response.content})
                messages.append({"role": "user", "content": tool_results})

                response = client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=1000,
                    tools=tools,
                    messages=messages,
                )

            else:
                break

    except Exception as e:
        console.print_error(str(e))
        console.print_error(
            "Make sure you have the ANTHROPIC_API_KEY environment variable set."
        )


if __name__ == "__main__":
    main()
