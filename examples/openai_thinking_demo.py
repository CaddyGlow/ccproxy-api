#!/usr/bin/env python3
"""
OpenAI SDK Thinking Mode Demonstration

This script demonstrates how to use thinking mode with the OpenAI SDK (pointing to Claude via proxy),
showcasing multi-turn conversations that preserve thinking blocks with signatures.
"""

import argparse
import asyncio
import json
import logging
import os
import re
from typing import Any, Optional

import httpx
import structlog
from openai import AsyncOpenAI
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionToolParam,
    ChatCompletionUserMessageParam,
)
from structlog import get_logger


try:
    from rich.align import Align
    from rich.console import Console
    from rich.live import Live
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


def setup_logging(verbose: int = 0) -> None:
    """Setup logging configuration.

    Args:
        verbose: Verbosity level (0=WARNING, 1=INFO, 2=DEBUG)
    """
    if verbose >= 2:
        level = logging.DEBUG
    elif verbose >= 1:
        level = logging.INFO
    else:
        level = logging.WARNING

    logging.basicConfig(
        level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Set the root logger level to ensure structlog respects it
    logging.getLogger().setLevel(level)

    # Configure structlog to use standard logging and respect log levels
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Set levels for external libraries
    if verbose >= 2:
        logging.getLogger("httpx").setLevel(logging.DEBUG)
        logging.getLogger("openai").setLevel(logging.DEBUG)
    elif verbose >= 1:
        logging.getLogger("httpx").setLevel(logging.INFO)
        logging.getLogger("openai").setLevel(logging.INFO)
    else:
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("openai").setLevel(logging.WARNING)


logger = get_logger(__name__)


class ThinkingRenderer:
    """Unified renderer for thinking blocks and responses with rich formatting."""

    def __init__(self, use_rich: bool = True):
        self.use_rich = use_rich and RICH_AVAILABLE
        self.console = Console() if self.use_rich else None

        if not RICH_AVAILABLE and use_rich:
            print("Warning: rich library not available, falling back to plain text")

    def render_thinking_blocks(self, content: str) -> None:
        """Extract and render thinking blocks with rich formatting."""
        thinking_pattern = r'<thinking signature="([^"]*)">(.*?)</thinking>'
        matches = re.findall(thinking_pattern, content, re.DOTALL)

        if matches:
            logger.debug("thinking_blocks_found", count=len(matches))
            for i, (signature, thinking_text) in enumerate(matches, 1):
                if self.use_rich:
                    self._render_thinking_block_rich(i, signature, thinking_text)
                else:
                    self._render_thinking_block_plain(i, signature, thinking_text)
                logger.debug("thinking_block_displayed", signature=signature)

    def _render_thinking_block_rich(
        self, index: int, signature: str, content: str
    ) -> None:
        """Render thinking block with rich formatting."""
        # Create a table for thinking block details
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Label", style="bold yellow")
        table.add_column("Value", style="white")

        table.add_row("Block:", f"{index}")
        table.add_row("Signature:", signature)

        # Create markdown content for the thinking text
        thinking_markdown = Markdown(content.strip())

        panel = Panel(
            Align.left(thinking_markdown),
            title=f"[bold yellow]🤔 THINKING BLOCK {index}[/bold yellow]",
            border_style="yellow",
            title_align="left",
            subtitle=f"[dim]Signature: {signature}[/dim]",
            subtitle_align="right",
        )

        self.console.print()
        self.console.print(panel)

    def _render_thinking_block_plain(
        self, index: int, signature: str, content: str
    ) -> None:
        """Render thinking block with plain text formatting."""
        print(f"\n{'=' * 60}")
        print(f"🤔 THINKING BLOCK {index}")
        print(f"Signature: {signature}")
        print(f"{'=' * 60}")
        print(content.strip())
        print(f"{'=' * 60}")

    def render_response(
        self, turn: int, speaker: str, content: str, show_thinking: bool = True
    ) -> None:
        """Render complete response with optional thinking blocks."""
        if show_thinking:
            # First show thinking blocks
            self.render_thinking_blocks(content)

        # Then show visible content
        visible_content = self.extract_visible_content(content)
        if visible_content:
            if self.use_rich:
                self._render_response_rich(turn, speaker, visible_content)
            else:
                self._render_response_plain(turn, speaker, visible_content)

    def _render_response_rich(self, turn: int, speaker: str, content: str) -> None:
        """Render response with rich formatting."""
        # Choose color based on speaker
        if "streaming" in speaker.lower():
            color = "cyan"
        elif "tool" in speaker.lower():
            color = "magenta"
        else:
            color = "green"

        header = f"Turn {turn}: {speaker}"

        # Create markdown content
        markdown_content = Markdown(content)
        panel = Panel(
            markdown_content,
            title=f"[bold {color}]{header}[/bold {color}]",
            border_style=color,
            title_align="left",
        )

        self.console.print()
        self.console.print(panel)

    def _render_response_plain(self, turn: int, speaker: str, content: str) -> None:
        """Render response with plain text formatting."""
        print(f"\n{'=' * 60}")
        print(f"Turn {turn}: {speaker}")
        print(f"{'=' * 60}")
        print(content)
        print(f"{'=' * 60}")

    def render_streaming(self, turn: int, speaker: str, text_generator) -> str:
        """Render streaming response with live updates."""
        if self.use_rich:
            return self._render_streaming_rich(turn, speaker, text_generator)
        else:
            return self._render_streaming_plain(turn, speaker, text_generator)

    def _render_streaming_rich(self, turn: int, speaker: str, text_generator) -> str:
        """Render streaming response with rich Live updates."""
        color = "cyan"
        header = f"Turn {turn}: {speaker}"
        accumulated_text = ""

        with Live(console=self.console, refresh_per_second=10) as live:
            for text_chunk in text_generator:
                accumulated_text += text_chunk

                # Show only visible content during streaming
                visible = self.extract_visible_content(accumulated_text)
                if visible:
                    markdown_content = Markdown(visible)
                    panel = Panel(
                        markdown_content,
                        title=f"[bold {color}]{header} (streaming...)[/bold {color}]",
                        border_style=color,
                        title_align="left",
                    )
                    live.update(panel)

        # Final update without "streaming..." label
        final_visible = self.extract_visible_content(accumulated_text)
        if final_visible:
            final_markdown = Markdown(final_visible)
            final_panel = Panel(
                final_markdown,
                title=f"[bold {color}]{header}[/bold {color}]",
                border_style=color,
                title_align="left",
            )
            self.console.print()
            self.console.print(final_panel)

        # Show thinking blocks after streaming completes
        self.render_thinking_blocks(accumulated_text)

        return accumulated_text

    def _render_streaming_plain(self, turn: int, speaker: str, text_generator) -> str:
        """Render streaming response with plain text."""
        print(f"\n{speaker} (streaming): ", end="", flush=True)

        accumulated_text = ""
        for text_chunk in text_generator:
            accumulated_text += text_chunk
            # Extract visible content from accumulated text to avoid showing partial markdown
            visible_content = self.extract_visible_content(accumulated_text)
            if visible_content:
                # Clear current line and reprint the visible content
                print(f"\r{speaker} (streaming): {visible_content}", end="", flush=True)

        print()  # New line after streaming

        # Show thinking blocks after streaming completes
        self.render_thinking_blocks(accumulated_text)

        print(f"{'=' * 60}")
        print(f"Turn complete: {len(accumulated_text)} characters")
        print(f"{'=' * 60}")

        return accumulated_text

    def extract_visible_content(self, content: str) -> str:
        """Extract only the visible content (not thinking blocks)."""
        thinking_pattern = r'<thinking signature="[^"]*">.*?</thinking>'
        visible_content = re.sub(thinking_pattern, "", content, flags=re.DOTALL).strip()
        logger.debug(
            "content_processed",
            original_length=len(content),
            visible_length=len(visible_content),
        )
        return visible_content


class LoggingHTTPClient(httpx.AsyncClient):
    """Custom HTTP client that logs requests and responses"""

    async def send(self, request: httpx.Request, **kwargs: Any) -> httpx.Response:
        logger.debug("http_request_start")
        logger.debug(
            "http_request_details",
            method=request.method,
            url=str(request.url),
            headers=dict(request.headers),
        )
        if request.content:
            try:
                content = request.content
                if isinstance(content, bytes):
                    content = content.decode("utf-8")
                logger.debug("http_request_body", body=content)
            except Exception as e:
                logger.debug("http_request_body_decode_error", error=str(e))

        response = await super().send(request, **kwargs)

        logger.debug(
            "http_response_start",
            status_code=response.status_code,
            headers=dict(response.headers),
        )

        # For non-streaming responses, log the body
        if not response.headers.get("content-type", "").startswith("text/event-stream"):
            try:
                logger.debug("http_response_body", body=response.text)
            except Exception as e:
                logger.debug("http_response_body_decode_error", error=str(e))
        else:
            logger.debug("http_response_body", body="[Streaming response]")

        return response


async def demo_streaming(use_rich: bool = True) -> None:
    """Demo streaming responses with thinking blocks.

    Demonstrates how thinking blocks work with streaming responses
    and multi-turn conversations.
    """
    renderer = ThinkingRenderer(use_rich=use_rich)
    if use_rich:
        console = Console()
        console.print("\n[bold cyan]=== STREAMING DEMO ===[/bold cyan]\n")
    else:
        print("=== STREAMING DEMO ===\n")
    logger.info("streaming_demo_start")

    # Initialize the client pointing to the proxy
    client = AsyncOpenAI(
        base_url=os.getenv("OPENAI_BASE_URL", "http://localhost:8000/api/v1"),
        api_key=os.getenv("OPENAI_API_KEY", "your-api-key-here"),
    )

    # First message - this will trigger thinking
    messages: list[ChatCompletionMessageParam] = [
        ChatCompletionUserMessageParam(
            role="user",
            content="I need to calculate the factorial of 5. Can you help me think through this step by step?",
        )
    ]

    logger.debug(
        "streaming_request_start", model="o1-preview", base_url=str(client.base_url)
    )
    user_content = str(getattr(messages[0], "content", ""))
    if use_rich:
        console = Console()
        console.print(f"[bold blue]User:[/bold blue] {user_content}")
        console.print("\n[bold cyan]Assistant (streaming):[/bold cyan]")
    else:
        print("User: " + user_content)
        print("\nAssistant (streaming):")

    # Stream the response
    full_content = ""
    stream = await client.chat.completions.create(
        model="o1-preview",  # This maps to Claude with thinking enabled
        messages=messages,
        stream=True,
        temperature=1.0,  # Must be 1.0 for thinking mode
    )

    if use_rich:
        # Use rich streaming with live updates
        chunks = []
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                chunks.append(chunk.choices[0].delta.content)

        def text_generator():
            yield from chunks

        full_content = renderer.render_streaming(
            turn=1, speaker="Assistant (streaming)", text_generator=text_generator()
        )
    else:
        # Plain text streaming
        full_content = ""
        print("Assistant (streaming): ", end="", flush=True)
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                full_content += content
                # Extract visible content from accumulated text to avoid showing partial markdown
                visible_content = renderer.extract_visible_content(full_content)
                if visible_content:
                    # Clear current line and reprint the visible content
                    print(
                        f"\rAssistant (streaming): {visible_content}",
                        end="",
                        flush=True,
                    )

    print("\n")

    logger.debug(
        "streaming_response_complete",
        content_length=len(full_content),
        has_thinking_blocks="<thinking" in full_content,
    )

    # Show thinking blocks after streaming completes
    renderer = ThinkingRenderer(use_rich=use_rich)
    renderer.render_thinking_blocks(full_content)

    # Add assistant's response to messages for multi-turn
    messages.append(
        ChatCompletionAssistantMessageParam(role="assistant", content=full_content)
    )

    # Second turn - reference previous thinking
    messages.append(
        ChatCompletionUserMessageParam(
            role="user",
            content="Great! Now can you use the same approach to calculate 7 factorial?",
        )
    )

    user_content = str(getattr(messages[-1], "content", ""))
    if use_rich:
        console = Console()
        console.print(f"\n[bold blue]User:[/bold blue] {user_content}")
        console.print("\n[bold cyan]Assistant (streaming):[/bold cyan]")
    else:
        print("\nUser: " + user_content)
        print("\nAssistant (streaming):")

    # Stream second response
    logger.debug("streaming_followup_request", message_count=len(messages))
    stream2 = await client.chat.completions.create(
        model="o1-preview",
        messages=messages,
        stream=True,
        temperature=1.0,  # Must be 1.0 for thinking mode
    )

    if use_rich:
        # Use rich streaming with live updates for second response
        chunks2 = []
        async for chunk in stream2:
            if chunk.choices[0].delta.content:
                chunks2.append(chunk.choices[0].delta.content)

        def text_generator2():
            yield from chunks2

        full_content_2 = renderer.render_streaming(
            turn=2, speaker="Assistant (streaming)", text_generator=text_generator2()
        )
    else:
        # Plain text streaming for second response
        full_content_2 = ""
        print("Assistant (streaming): ", end="", flush=True)
        async for chunk in stream2:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                full_content_2 += content
                # Extract visible content from accumulated text to avoid showing partial markdown
                visible_content = renderer.extract_visible_content(full_content_2)
                if visible_content:
                    # Clear current line and reprint the visible content
                    print(
                        f"\rAssistant (streaming): {visible_content}",
                        end="",
                        flush=True,
                    )

    print("\n")
    renderer = ThinkingRenderer(use_rich=use_rich)
    renderer.render_thinking_blocks(full_content_2)
    logger.info("streaming_demo_complete")


async def demo_non_streaming(use_rich: bool = True) -> None:
    """Demo non-streaming responses with thinking blocks.

    Demonstrates how thinking blocks work with non-streaming responses
    and multi-turn conversations with detailed logging.
    """
    renderer = ThinkingRenderer(use_rich=use_rich)
    if use_rich:
        console = Console()
        console.print("\n[bold green]=== NON-STREAMING DEMO ===[/bold green]\n")
    else:
        print("\n=== NON-STREAMING DEMO ===\n")
    logger.info("non_streaming_demo_start")

    # Initialize the client with custom HTTP client for logging
    client = AsyncOpenAI(
        base_url=os.getenv("OPENAI_BASE_URL", "http://localhost:8000/api/v1"),
        api_key=os.getenv("OPENAI_API_KEY", "your-api-key-here"),
        http_client=LoggingHTTPClient(),
    )

    # Create a conversation about a coding problem
    messages: list[ChatCompletionMessageParam] = [
        ChatCompletionUserMessageParam(
            role="user",
            content="I have a list [3, 1, 4, 1, 5, 9, 2, 6] and I need to find the two numbers that sum to 10. Can you help?",
        )
    ]

    logger.debug("non_streaming_request_start", model="o1-mini")
    user_content = str(getattr(messages[0], "content", ""))
    if use_rich:
        console = Console()
        console.print(f"[bold blue]User:[/bold blue] {user_content}")
    else:
        print("User: " + user_content)

    # Get response
    response = await client.chat.completions.create(
        model="o1-mini",  # Maps to Claude with thinking
        messages=messages,
        temperature=1.0,  # Must be 1.0 for thinking mode
    )

    logger.debug(
        "non_streaming_response_received",
        response_id=response.id,
        model=response.model,
        usage=response.usage.model_dump()
        if hasattr(response.usage, "model_dump")
        else dict(response.usage)
        if response.usage
        else None,
    )

    content = response.choices[0].message.content
    if not content:
        logger.warning("no_content_in_response", response=str(response))
        if use_rich:
            console = Console()
            console.print("\n[bold red]Assistant: [No content][/bold red]")
        else:
            print("\nAssistant: [No content]")
        return

    logger.debug(
        "response_analysis",
        content_length=len(content),
        has_thinking_blocks="<thinking" in content,
    )

    visible_content = renderer.extract_visible_content(content)
    if use_rich:
        console = Console()
        console.print(f"\n[bold green]Assistant:[/bold green] {visible_content}")
    else:
        print("\nAssistant: " + visible_content)
    renderer.render_thinking_blocks(content)

    # Continue conversation
    messages.append(
        ChatCompletionAssistantMessageParam(role="assistant", content=content)
    )
    messages.append(
        ChatCompletionUserMessageParam(
            role="user",
            content="What if I needed to find three numbers that sum to 15?",
        )
    )

    user_content = str(getattr(messages[-1], "content", ""))
    if use_rich:
        console = Console()
        console.print(f"\n[bold blue]User:[/bold blue] {user_content}")
    else:
        print("\nUser: " + user_content)

    # Get second response
    logger.debug("non_streaming_followup_request", message_count=len(messages))
    response2 = await client.chat.completions.create(
        model="o1-mini",
        messages=messages,
        temperature=1.0,  # Must be 1.0 for thinking mode
    )

    content2 = response2.choices[0].message.content
    if content2:
        visible_content = renderer.extract_visible_content(content2)
        if use_rich:
            console = Console()
            console.print(f"\n[bold green]Assistant:[/bold green] {visible_content}")
        else:
            print("\nAssistant: " + visible_content)
        renderer.render_thinking_blocks(content2)

    logger.info("non_streaming_demo_complete")


async def demo_tool_use_with_thinking(use_rich: bool = True) -> None:
    """Demo tool use with thinking blocks.

    Demonstrates how thinking blocks work when tools are used,
    showing the thinking process before and after tool execution.
    """
    renderer = ThinkingRenderer(use_rich=use_rich)
    if use_rich:
        console = Console()
        console.print(
            "\n[bold magenta]=== TOOL USE WITH THINKING DEMO ===[/bold magenta]\n"
        )
    else:
        print("\n=== TOOL USE WITH THINKING DEMO ===\n")
    logger.info("tool_use_thinking_demo_start")

    client = AsyncOpenAI(
        base_url=os.getenv("OPENAI_BASE_URL", "http://localhost:8000/api/v1"),
        api_key=os.getenv("OPENAI_API_KEY", "your-api-key-here"),
    )

    # Define a simple calculator tool
    tools: list[ChatCompletionToolParam] = [
        ChatCompletionToolParam(
            type="function",
            function={
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
        )
    ]

    messages: list[ChatCompletionMessageParam] = [
        ChatCompletionUserMessageParam(
            role="user",
            content="I need to calculate the compound interest on $1000 at 5% annual rate for 3 years.",
        )
    ]

    logger.debug("tool_use_request_start", model="o1-mini", tools_count=len(tools))
    user_content = str(getattr(messages[0], "content", ""))
    if use_rich:
        console = Console()
        console.print(f"[bold blue]User:[/bold blue] {user_content}")
    else:
        print("User: " + user_content)

    # Get response with tool use
    response = await client.chat.completions.create(
        model="o1-mini",
        messages=messages,
        tools=tools,
        tool_choice="auto",
        temperature=1.0,  # Must be 1.0 for thinking mode
    )

    message = response.choices[0].message
    logger.debug(
        "tool_use_response_received",
        has_content=bool(message.content),
        has_tool_calls=bool(message.tool_calls),
        tool_calls_count=len(message.tool_calls) if message.tool_calls else 0,
    )

    # Print response content (including thinking)
    if message.content:
        visible_content = renderer.extract_visible_content(message.content)
        if use_rich:
            console = Console()
            console.print(f"\n[bold green]Assistant:[/bold green] {visible_content}")
        else:
            print("\nAssistant: " + visible_content)
        renderer.render_thinking_blocks(message.content)

    # Handle tool calls
    if message.tool_calls:
        if use_rich:
            console = Console()
            console.print("\n[bold yellow][TOOL CALLS][/bold yellow]")
        else:
            print("\n[TOOL CALLS]")
        tool_messages = []

        for tool_call in message.tool_calls:
            tool_name = tool_call.function.name
            tool_args = tool_call.function.arguments
            tool_call_id = tool_call.id

            logger.debug(
                "tool_call_execution",
                tool_name=tool_name,
                tool_args=tool_args,
                tool_call_id=tool_call_id,
            )

            if use_rich:
                console = Console()
                console.print(
                    f"  [cyan]Calling {tool_name}[/cyan] with args: [yellow]{tool_args}[/yellow]"
                )
            else:
                print(f"  Calling {tool_name} with args: {tool_args}")

            # Simulate tool execution
            try:
                args = json.loads(tool_args)
                # Fix ^ to ** for exponentiation in Python
                expression = args["expression"].replace("^", "**")
                result = eval(expression)  # Note: Don't use eval in production!

                logger.debug(
                    "tool_execution_result", expression=expression, result=result
                )

                tool_messages.append(
                    ChatCompletionToolMessageParam(
                        role="tool",
                        content=str(result),
                        tool_call_id=tool_call_id,
                    )
                )
            except Exception as e:
                logger.error("tool_execution_error", error=str(e))
                tool_messages.append(
                    ChatCompletionToolMessageParam(
                        role="tool",
                        content=f"Error: {str(e)}",
                        tool_call_id=tool_call_id,
                    )
                )

        # Add the assistant message to conversation (with thinking blocks preserved)
        messages.append(
            ChatCompletionAssistantMessageParam(
                role="assistant",
                content=message.content,
                tool_calls=[
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in message.tool_calls
                ],
            )
        )
        messages.extend(tool_messages)

        # Get final response
        logger.debug(
            "tool_use_final_request",
            message="Getting final response after tool execution",
            message_count=len(messages),
        )

        if use_rich:
            console = Console()
            console.print(
                "\n[dim][Getting final response after tool execution...][/dim]"
            )
        else:
            print("\n[Getting final response after tool execution...]")
        final_response = await client.chat.completions.create(
            model="o1-mini",
            messages=messages,
            temperature=1.0,  # Must be 1.0 for thinking mode
        )

        final_content = final_response.choices[0].message.content
        if final_content:
            visible_content = renderer.extract_visible_content(final_content)
            if use_rich:
                console = Console()
                console.print(
                    f"\n[bold green]Assistant (after tool use):[/bold green] {visible_content}"
                )
            else:
                print("\nAssistant (after tool use): " + visible_content)
            renderer.render_thinking_blocks(final_content)

    logger.info("tool_use_thinking_demo_complete")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="OpenAI SDK Thinking Mode Demonstration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 openai_thinking_demo.py                    # WARNING level (quiet)
  python3 openai_thinking_demo.py -v                 # INFO level (basic logging)
  python3 openai_thinking_demo.py -vv                # DEBUG level (detailed logging)
  python3 openai_thinking_demo.py --debug            # DEBUG level (equivalent to -vv)
  python3 openai_thinking_demo.py -v --streaming     # INFO level with streaming
        """,
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase verbosity (-v=INFO, -vv=DEBUG). Default is WARNING level.",
    )
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="Enable debug logging (equivalent to -vv)",
    )
    parser.add_argument(
        "-s",
        "--streaming",
        action="store_true",
        help="Run streaming demo instead of non-streaming",
    )
    parser.add_argument(
        "-t",
        "--tools",
        action="store_true",
        help="Run tool use with thinking demo",
    )
    parser.add_argument(
        "-p",
        "--plain",
        action="store_true",
        help="Disable rich formatting and use plain text output",
    )
    return parser.parse_args()


async def main() -> None:
    """
    Main demonstration function.
    """
    args = parse_args()
    # Handle debug flag as -vv for backwards compatibility
    verbose_level = max(args.verbose, 2 if args.debug else 0)
    setup_logging(verbose=verbose_level)

    use_rich = not args.plain and RICH_AVAILABLE

    # Check environment variables
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    base_url_default = "http://127.0.0.1:8000/api/v1"

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

    if use_rich:
        console = Console()

        # Header
        console.print("\n[bold blue]OpenAI SDK Thinking Mode Demonstration[/bold blue]")
        console.print("=" * 60)

        # Warnings
        if not api_key:
            console.print(
                "[yellow]Warning: OPENAI_API_KEY not set, using dummy key[/yellow]"
            )
        if not base_url:
            console.print(
                "[yellow]Warning: OPENAI_BASE_URL not set, using default[/yellow]"
            )

        # Configuration table
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Setting", style="bold cyan")
        table.add_column("Value", style="green")

        if args.streaming:
            table.add_row("Demo mode:", "streaming")
        elif args.tools:
            table.add_row("Demo mode:", "tool use with thinking")
        else:
            table.add_row("Demo mode:", "non-streaming")

        table.add_row("Base URL:", base_url_default)

        # Show logging level
        if verbose_level >= 2:
            table.add_row("Log level:", "DEBUG (-vv)")
        elif verbose_level >= 1:
            table.add_row("Log level:", "INFO (-v)")
        else:
            table.add_row("Log level:", "WARNING (default)")

        if args.plain:
            table.add_row("Rich formatting:", "disabled")
        else:
            table.add_row("Rich formatting:", "enabled (default)")

        panel = Panel(
            table,
            title="[bold magenta]Demo Configuration[/bold magenta]",
            border_style="magenta",
            title_align="left",
        )

        console.print(panel)
        console.print()
        console.print(
            "[dim]This demo shows how thinking blocks are preserved in multi-turn conversations.[/dim]"
        )
        console.print(
            "[dim]Make sure the proxy server is running on http://localhost:8000[/dim]"
        )
        console.print("=" * 60)

    else:
        # Fallback to plain text
        print("OpenAI SDK Thinking Mode Demonstration")
        print("=" * 60)

        if not api_key:
            print("Warning: OPENAI_API_KEY not set, using dummy key")
        if not base_url:
            print("Warning: OPENAI_BASE_URL not set, using default")

        if args.streaming:
            print("Demo mode: streaming")
        elif args.tools:
            print("Demo mode: tool use with thinking")
        else:
            print("Demo mode: non-streaming")

        print(f"Base URL: {base_url_default}")

        # Show logging level
        if verbose_level >= 2:
            print("Log level: DEBUG (-vv)")
        elif verbose_level >= 1:
            print("Log level: INFO (-v)")
        else:
            print("Log level: WARNING (default)")

        if args.plain:
            print("Rich formatting: disabled")
        else:
            print("Rich formatting: enabled (default)")

        print(
            "\nThis demo shows how thinking blocks are preserved in multi-turn conversations."
        )
        print("Make sure the proxy server is running on http://localhost:8000")
        print("=" * 60)

    try:
        if args.streaming:
            logger.info("demo_mode_selected", mode="streaming")
            await demo_streaming(use_rich=use_rich)
        elif args.tools:
            logger.info("demo_mode_selected", mode="tool_use_with_thinking")
            await demo_tool_use_with_thinking(use_rich=use_rich)
        else:
            logger.info("demo_mode_selected", mode="non_streaming")
            await demo_non_streaming(use_rich=use_rich)
    except KeyboardInterrupt:
        if use_rich:
            console = Console()
            console.print("\n[yellow]Demo interrupted by user[/yellow]")
        else:
            print("\nDemo interrupted by user")
    except Exception as e:
        if use_rich:
            console = Console()
            console.print(f"\n[bold red]Error:[/bold red] {e}")
            console.print(
                "[yellow]Make sure your proxy server is running on http://127.0.0.1:8000[/yellow]"
            )
        else:
            print(f"\nError: {e}")
            print("Make sure your proxy server is running on http://127.0.0.1:8000")
        logger.error("demo_error", error=str(e))


if __name__ == "__main__":
    asyncio.run(main())
