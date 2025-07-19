#!/usr/bin/env python3
"""
OpenAI SDK Thinking Mode Demonstration (Refactored)

This script demonstrates how to use thinking mode with the OpenAI SDK, leveraging
a shared library for common functionality.
"""

import argparse
import asyncio
import json
import os

from common_utils import LoggingAsyncClient, setup_logging
from console_utils import RICH_AVAILABLE, ThinkingRenderer
from openai import AsyncOpenAI
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionUserMessageParam,
)
from structlog import get_logger


logger = get_logger(__name__)


async def demo_streaming(use_rich: bool = True) -> None:
    """Demo streaming responses with thinking blocks."""
    renderer = ThinkingRenderer(use_rich=use_rich)
    renderer.print_turn_separator(1)
    client = AsyncOpenAI(
        base_url=os.getenv("OPENAI_BASE_URL", "http://localhost:8000/api/v1"),
        api_key=os.getenv("OPENAI_API_KEY", "your-api-key-here"),
        http_client=LoggingAsyncClient(),
    )

    messages: list[ChatCompletionMessageParam] = [
        ChatCompletionUserMessageParam(
            role="user",
            content="I need to calculate the factorial of 5. Can you help me think through this step by step?",
        )
    ]
    renderer.print_user_message(messages[0]["content"])

    stream = await client.chat.completions.create(
        model="o1-preview",
        messages=messages,
        stream=True,
        temperature=1.0,
    )

    await renderer.print_streaming_response_with_thinking_async(
        stream, title="Assistant Response (with thinking)"
    )


async def demo_non_streaming(use_rich: bool = True) -> None:
    """Demo non-streaming responses with thinking blocks."""
    renderer = ThinkingRenderer(use_rich=use_rich)
    renderer.print_turn_separator(1)
    client = AsyncOpenAI(
        base_url=os.getenv("OPENAI_BASE_URL", "http://localhost:8000/api/v1"),
        api_key=os.getenv("OPENAI_API_KEY", "your-api-key-here"),
        http_client=LoggingAsyncClient(),
    )

    messages: list[ChatCompletionMessageParam] = [
        ChatCompletionUserMessageParam(
            role="user",
            content="I have a list [3, 1, 4, 1, 5, 9, 2, 6] and I need to find the two numbers that sum to 10. Can you help?",
        )
    ]
    renderer.print_user_message(messages[0]["content"])

    response = await client.chat.completions.create(
        model="o1-mini",
        messages=messages,
        temperature=1.0,
    )

    content = response.choices[0].message.content
    if content:
        renderer.print_response(
            renderer.extract_visible_content(content), response.choices[0].finish_reason
        )
        renderer.render_thinking_blocks(content)


async def demo_tool_use_with_thinking(use_rich: bool = True) -> None:
    """Demo tool use with thinking blocks."""
    renderer = ThinkingRenderer(use_rich=use_rich)
    renderer.print_turn_separator(1)
    client = AsyncOpenAI(
        base_url=os.getenv("OPENAI_BASE_URL", "http://localhost:8000/api/v1"),
        api_key=os.getenv("OPENAI_API_KEY", "your-api-key-here"),
        http_client=LoggingAsyncClient(),
    )

    tools = [
        {
            "type": "function",
            "function": {
                "name": "calculate",
                "description": "Perform basic arithmetic calculations",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "The mathematical expression to evaluate",
                        }
                    },
                    "required": ["expression"],
                },
            },
        }
    ]

    messages: list[ChatCompletionMessageParam] = [
        ChatCompletionUserMessageParam(
            role="user",
            content="I need to calculate the compound interest on $1000 at 5% annual rate for 3 years.",
        )
    ]
    renderer.print_user_message(messages[0]["content"])

    response = await client.chat.completions.create(
        model="o1-mini",
        messages=messages,
        tools=tools,
        tool_choice="auto",
        temperature=1.0,
    )

    message = response.choices[0].message
    if message.content:
        renderer.print_response(
            renderer.extract_visible_content(message.content),
            response.choices[0].finish_reason,
        )
        renderer.render_thinking_blocks(message.content)

    if message.tool_calls:
        tool_messages = []
        for tool_call in message.tool_calls:
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)
            expression = tool_args["expression"].replace("^", "**")
            result = eval(expression)
            tool_messages.append(
                ChatCompletionToolMessageParam(
                    role="tool",
                    content=str(result),
                    tool_call_id=tool_call.id,
                )
            )

        messages.append(message)
        messages.extend(tool_messages)

        final_response = await client.chat.completions.create(
            model="o1-mini",
            messages=messages,
            temperature=1.0,
        )

        final_content = final_response.choices[0].message.content
        if final_content:
            renderer.print_response(
                renderer.extract_visible_content(final_content),
                final_response.choices[0].finish_reason,
            )
            renderer.render_thinking_blocks(final_content)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="OpenAI SDK Thinking Mode Demonstration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 openai_thinking_demo.py -v
  python3 openai_thinking_demo.py -vv --streaming
  python3 openai_thinking_demo.py --tools
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
        "-s", "--streaming", action="store_true", help="Run streaming demo."
    )
    parser.add_argument(
        "-t", "--tools", action="store_true", help="Run tool use with thinking demo."
    )
    parser.add_argument(
        "-p", "--plain", action="store_true", help="Disable rich formatting."
    )
    return parser.parse_args()


async def main() -> None:
    """
    Main demonstration function.
    """
    args = parse_args()
    setup_logging(verbose=args.verbose)
    use_rich = not args.plain and RICH_AVAILABLE

    if args.streaming:
        await demo_streaming(use_rich=use_rich)
    elif args.tools:
        await demo_tool_use_with_thinking(use_rich=use_rich)
    else:
        await demo_non_streaming(use_rich=use_rich)


if __name__ == "__main__":
    asyncio.run(main())
