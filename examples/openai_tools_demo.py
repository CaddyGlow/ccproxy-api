#!/usr/bin/env python3
"""
OpenAI SDK Tool Use Demonstration (Refactored)

This script demonstrates how to use tools with the OpenAI SDK, leveraging
a shared library for common functionality.
"""

import argparse
import json
import os

import openai
from common_utils import (
    LoggingSyncClient,
    calculate_distance,
    generate_json_schema_for_function,
    get_weather,
    handle_tool_call,
    setup_logging,
)
from console_utils import RICH_AVAILABLE, RichConsoleManager
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionToolParam,
    ChatCompletionUserMessageParam,
)
from structlog import get_logger


logger = get_logger(__name__)


def create_openai_tools() -> list[ChatCompletionToolParam]:
    """
    Create OpenAI-compatible tool definitions with JSON schemas.

    Returns:
        List of tool definitions
    """
    tools: list[ChatCompletionToolParam] = []

    weather_schema = generate_json_schema_for_function(get_weather)
    tools.append(
        ChatCompletionToolParam(
            type="function",
            function={
                "name": "get_weather",
                "description": "Get current weather information for a specific location",
                "parameters": weather_schema,
            },
        )
    )

    distance_schema = generate_json_schema_for_function(calculate_distance)
    tools.append(
        ChatCompletionToolParam(
            type="function",
            function={
                "name": "calculate_distance",
                "description": "Calculate the distance between two geographic coordinates",
                "parameters": distance_schema,
            },
        )
    )

    return tools


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="OpenAI SDK Tool Use Demonstration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 openai_tools_demo.py
  python3 openai_tools_demo.py -v
  python3 openai_tools_demo.py -vv
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

    console.print_header("OpenAI SDK Tool Use Demonstration")

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

    tools = create_openai_tools()
    console.print_tools(tools)

    try:
        http_client = LoggingSyncClient()
        client = openai.OpenAI(http_client=http_client)

        messages: list[ChatCompletionMessageParam] = [
            ChatCompletionUserMessageParam(
                role="user",
                content="What's the weather like in New York, and how far is it from Los Angeles?",
            )
        ]

        console.print_turn_separator(1)
        console.print_user_message(messages[0]["content"])

        console.print_subheader("Starting conversation with Claude via OpenAI API...")

        response = client.chat.completions.create(
            model="gpt-4o",
            max_tokens=1000,
            tools=tools,
            messages=messages,
        )

        while True:
            choice = response.choices[0]
            console.print_response(choice.message.content or "", choice.finish_reason)

            if choice.finish_reason == "tool_calls" and choice.message.tool_calls:
                tool_messages = []

                for tool_call in choice.message.tool_calls:
                    tool_name = tool_call.function.name
                    tool_input = json.loads(tool_call.function.arguments)
                    tool_call_id = tool_call.id

                    console.print_tool_call(tool_name, tool_input)
                    result = handle_tool_call(tool_name, tool_input)
                    console.print_tool_result(result)

                    tool_messages.append(
                        ChatCompletionToolMessageParam(
                            role="tool",
                            content=json.dumps(result),
                            tool_call_id=tool_call_id,
                        )
                    )

                messages.append(
                    ChatCompletionAssistantMessageParam(
                        role="assistant",
                        content=choice.message.content,
                        tool_calls=[
                            {
                                "id": tc.id,
                                "type": tc.type,
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments,
                                },
                            }
                            for tc in choice.message.tool_calls
                        ],
                    )
                )
                messages.extend(tool_messages)

                response = client.chat.completions.create(
                    model="gpt-4o",
                    max_tokens=1000,
                    tools=tools,
                    messages=messages,
                )

            else:
                break

    except Exception as e:
        console.print_error(str(e))
        console.print_error(
            "Make sure your proxy server is running on http://127.0.0.1:8000"
        )


if __name__ == "__main__":
    main()
