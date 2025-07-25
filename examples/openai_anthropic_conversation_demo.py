#!/usr/bin/env python3
"""
OpenAI-to-Anthropic Bidirectional Conversation Demo

This script demonstrates a conversation loop between OpenAI and Anthropic clients
through the proxy server. It starts with a topic, then alternates between:
1. OpenAI client -> Proxy -> Claude (Anthropic)
2. Response forwarded back with roles swapped
3. Continues the conversation loop

The proxy acts as a bridge, forwarding messages and swapping assistant/user roles
to create a natural conversation flow between the two AI models.
"""

import argparse
import asyncio
import logging
import os

import anthropic
import openai
from anthropic.types import MessageParam
from openai.types.chat import ChatCompletionMessageParam


try:
    from rich.console import Console
    from rich.live import Live
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.table import Table

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

logger = logging.getLogger(__name__)


class MarkdownRenderer:
    """Unified markdown renderer for both streaming and non-streaming modes."""

    def __init__(self, use_markdown: bool = True):
        self.use_markdown = use_markdown and RICH_AVAILABLE
        self.console = Console() if self.use_markdown else None

        if not RICH_AVAILABLE and use_markdown:
            print("Warning: rich library not available, falling back to plain text")

    def render_complete(self, turn: int, speaker: str, text: str) -> None:
        """Render complete response with markdown formatting."""
        if self.use_markdown:
            self._render_markdown_complete(turn, speaker, text)
        else:
            self._render_plain_complete(turn, speaker, text)

    def render_streaming(self, turn: int, speaker: str, text_generator) -> str:
        """Render streaming response with live markdown updates."""
        if self.use_markdown:
            return self._render_markdown_streaming(turn, speaker, text_generator)
        else:
            return self._render_plain_streaming(turn, speaker, text_generator)

    def _render_markdown_complete(self, turn: int, speaker: str, text: str) -> None:
        """Render complete markdown response using rich."""
        color = "cyan" if "OpenAI" in speaker else "green"

        # Create header
        header = f"Turn {turn}: {speaker}"

        # Render markdown content
        markdown_content = Markdown(text)
        panel = Panel(
            markdown_content, title=header, border_style=color, title_align="left"
        )

        self.console.print()
        self.console.print(panel)

    def _render_markdown_streaming(
        self, turn: int, speaker: str, text_generator
    ) -> str:
        """Render streaming markdown response using rich Live."""
        color = "cyan" if "OpenAI" in speaker else "green"
        header = f"Turn {turn}: {speaker}"

        accumulated_text = ""

        with Live(console=self.console, refresh_per_second=10) as live:
            for text_chunk in text_generator:
                accumulated_text += text_chunk

                # Create markdown content
                markdown_content = Markdown(accumulated_text)
                panel = Panel(
                    markdown_content,
                    title=f"{header} (streaming...)",
                    border_style=color,
                    title_align="left",
                )

                live.update(panel)

        # Final update without "streaming..." label
        final_markdown = Markdown(accumulated_text)
        final_panel = Panel(
            final_markdown, title=header, border_style=color, title_align="left"
        )
        self.console.print()
        self.console.print(final_panel)

        return accumulated_text

    def _render_plain_complete(self, turn: int, speaker: str, text: str) -> None:
        """Render complete plain text response."""
        print(f"\n{'=' * 60}")
        print(f"Turn {turn}: {speaker}")
        print(f"{'=' * 60}")
        print(text)
        print(f"{'=' * 60}")

    def _render_plain_streaming(self, turn: int, speaker: str, text_generator) -> str:
        """Render streaming plain text response."""
        print(f"\n{speaker} (streaming): ", end="", flush=True)

        accumulated_text = ""
        for text_chunk in text_generator:
            print(text_chunk, end="", flush=True)
            accumulated_text += text_chunk

        print()  # New line after streaming

        print(f"{'=' * 60}")
        print(f"Turn complete: {len(accumulated_text)} characters")
        print(f"{'=' * 60}")

        return accumulated_text


def setup_logging(debug: bool = False) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Set the root logger level to ensure structlog respects it
    logging.getLogger().setLevel(level)

    if debug:
        logging.getLogger("httpx").setLevel(logging.DEBUG)
        logging.getLogger("openai").setLevel(logging.DEBUG)
        logging.getLogger("anthropic").setLevel(logging.DEBUG)
    else:
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("openai").setLevel(logging.WARNING)
        logging.getLogger("anthropic").setLevel(logging.WARNING)


class ConversationManager:
    """Manages the bidirectional conversation between OpenAI and Anthropic clients."""

    def __init__(
        self,
        proxy_url: str = "http://127.0.0.1:8000/api",
        debug: bool = False,
        stream: bool = False,
        use_markdown: bool = True,
    ):
        self.proxy_url = proxy_url
        self.debug = debug
        self.stream = stream
        self.use_markdown = use_markdown

        # Initialize clients
        self.openai_client = openai.OpenAI(
            api_key=os.getenv("OPENAI_API_KEY", "dummy"),
            base_url=f"{proxy_url}/v1",
        )

        self.anthropic_client = anthropic.Anthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY", "dummy"), base_url=f"{proxy_url}"
        )

        # Initialize markdown renderer
        self.renderer = MarkdownRenderer(use_markdown=use_markdown)

        # Conversation history
        self.openai_messages: list[ChatCompletionMessageParam] = []
        self.anthropic_messages: list[MessageParam] = []

        logger.debug(
            "conversation_manager_initialized",
            proxy_url=proxy_url,
            stream=stream,
            markdown=use_markdown,
        )

    def add_initial_topic(self, topic: str) -> None:
        """Add the initial topic to start the conversation."""
        self.openai_messages.append(
            {
                "role": "user",
                "content": f"Let's discuss: {topic}. Please share your thoughts and ask me a question to continue the conversation.",
            }
        )

        self.anthropic_messages.append(
            {
                "role": "user",
                "content": f"Let's discuss: {topic}. Please share your thoughts and ask me a question to continue the conversation.",
            }
        )

        logger.debug("initial_topic_added", topic=topic)

    def openai_to_anthropic_messages(
        self, openai_messages: list[ChatCompletionMessageParam]
    ) -> list[MessageParam]:
        """Convert OpenAI message format to Anthropic format with role swapping."""
        anthropic_msgs = []

        for msg in openai_messages:
            role = msg["role"]
            content = msg["content"]

            # Skip system messages for now
            if role == "system":
                continue

            # Swap roles: assistant becomes user, user becomes assistant
            if role == "assistant":
                new_role = "user"
            elif role == "user":
                new_role = "assistant"
            else:
                new_role = role

            anthropic_msgs.append({"role": new_role, "content": content})

        return anthropic_msgs

    def anthropic_to_openai_messages(
        self, anthropic_messages: list[MessageParam]
    ) -> list[ChatCompletionMessageParam]:
        """Convert Anthropic message format to OpenAI format with role swapping."""
        openai_msgs = []

        for msg in anthropic_messages:
            role = msg["role"]
            content = msg["content"]

            # Swap roles: assistant becomes user, user becomes assistant
            if role == "assistant":
                new_role = "user"
            elif role == "user":
                new_role = "assistant"
            else:
                new_role = role

            openai_msgs.append({"role": new_role, "content": content})

        return openai_msgs

    async def send_to_openai(
        self, messages: list[ChatCompletionMessageParam], turn: int = 0
    ) -> str:
        """Send messages to OpenAI client via proxy."""
        logger.debug(
            "sending_to_openai", message_count=len(messages), stream=self.stream
        )

        if self.debug:
            logger.debug("openai_request_messages", messages=messages)

        try:
            if self.stream:
                return await self._send_to_openai_stream(messages, turn)
            else:
                response = self.openai_client.chat.completions.create(
                    model="gpt-4o", messages=messages, max_tokens=500, temperature=0.7
                )

                if not response.choices:
                    raise Exception("No choices in OpenAI response")

                content = response.choices[0].message.content or ""

                logger.debug("openai_response_received", content_length=len(content))
                if self.debug:
                    logger.debug("openai_response_content", content=content)

                return content

        except Exception as e:
            logger.error("openai_request_failed", error=str(e))
            raise

    async def _send_to_openai_stream(
        self, messages: list[ChatCompletionMessageParam], turn: int
    ) -> str:
        """Send messages to OpenAI client via proxy with streaming."""
        logger.debug("sending_to_openai_stream", message_count=len(messages))

        try:
            stream = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=500,
                temperature=0.7,
                stream=True,
            )

            def text_generator():
                for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content

            # Use renderer for streaming output
            content = self.renderer.render_streaming(
                turn=turn, speaker="OpenAI (via proxy)", text_generator=text_generator()
            )

            logger.debug("openai_stream_response_received", content_length=len(content))
            if self.debug:
                logger.debug("openai_stream_response_content", content=content)

            return content

        except Exception as e:
            logger.error("openai_stream_request_failed", error=str(e))
            raise

    async def send_to_anthropic(
        self, messages: list[MessageParam], turn: int = 0
    ) -> str:
        """Send messages to Anthropic client via proxy."""
        logger.debug(
            "sending_to_anthropic", message_count=len(messages), stream=self.stream
        )

        if self.debug:
            logger.debug("anthropic_request_messages", messages=messages)

        try:
            if self.stream:
                return await self._send_to_anthropic_stream(messages, turn)
            else:
                response = self.anthropic_client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    messages=messages,
                    max_tokens=500,
                    temperature=0.7,
                )

                content = ""
                for block in response.content:
                    if block.type == "text":
                        content += block.text

                logger.debug("anthropic_response_received", content_length=len(content))
                if self.debug:
                    logger.debug("anthropic_response_content", content=content)

                return content

        except Exception as e:
            logger.error("anthropic_request_failed", error=str(e))
            raise

    async def _send_to_anthropic_stream(
        self, messages: list[MessageParam], turn: int
    ) -> str:
        """Send messages to Anthropic client via proxy with streaming."""
        logger.debug("sending_to_anthropic_stream", message_count=len(messages))

        try:
            stream = self.anthropic_client.messages.create(
                model="claude-3-5-sonnet-20241022",
                messages=messages,
                max_tokens=500,
                temperature=0.7,
                stream=True,
            )

            def text_generator():
                for event in stream:
                    if (
                        event.type == "content_block_delta"
                        and event.delta.type == "text_delta"
                    ):
                        yield event.delta.text

            # Use renderer for streaming output
            content = self.renderer.render_streaming(
                turn=turn,
                speaker="Anthropic (via proxy)",
                text_generator=text_generator(),
            )

            logger.debug(
                "anthropic_stream_response_received", content_length=len(content)
            )
            if self.debug:
                logger.debug("anthropic_stream_response_content", content=content)

            return content

        except Exception as e:
            logger.error("anthropic_stream_request_failed", error=str(e))
            raise

    def print_conversation_state(self, turn: int, speaker: str, message: str) -> None:
        """Print the current conversation state."""
        if (
            not self.stream
        ):  # Only print if not streaming (streaming prints in real-time)
            self.renderer.render_complete(turn, speaker, message)

    def print_conversation_start(self, topic: str, max_turns: int) -> None:
        """Print conversation start information using rich formatting."""
        if self.use_markdown:
            table = Table(show_header=False, box=None, padding=(0, 1))
            table.add_column("Label", style="bold cyan")
            table.add_column("Value", style="green")

            table.add_row("Topic:", topic)
            table.add_row("Max turns:", str(max_turns))
            table.add_row("Proxy URL:", self.proxy_url)
            table.add_row("Streaming:", "enabled" if self.stream else "disabled")
            table.add_row("Markdown:", "enabled" if self.use_markdown else "disabled")

            panel = Panel(
                table,
                title="[bold blue]Conversation Configuration[/bold blue]",
                border_style="blue",
                title_align="left",
            )

            self.renderer.console.print()
            self.renderer.console.print(panel)
            self.renderer.console.print()
        else:
            print(f"Starting conversation about: {topic}")
            print(f"Max turns: {max_turns}")
            print(f"Proxy URL: {self.proxy_url}")
            print(f"Streaming: {'enabled' if self.stream else 'disabled'}")
            print()

    def print_conversation_end(self, total_turns: int) -> None:
        """Print conversation end information using rich formatting."""
        if self.use_markdown:
            table = Table(show_header=False, box=None, padding=(0, 1))
            table.add_column("Label", style="bold cyan")
            table.add_column("Value", style="green")

            table.add_row("Total turns:", str(total_turns))
            table.add_row("OpenAI messages:", str(len(self.openai_messages)))
            table.add_row("Anthropic messages:", str(len(self.anthropic_messages)))

            panel = Panel(
                table,
                title="[bold green]Conversation Completed[/bold green]",
                border_style="green",
                title_align="left",
            )

            self.renderer.console.print()
            self.renderer.console.print(panel)
        else:
            print(f"\n{'=' * 60}")
            print("Conversation completed!")
            print(f"Total turns: {total_turns}")
            print(f"{'=' * 60}")

    async def run_conversation(self, topic: str, max_turns: int = 6) -> None:
        """Run the bidirectional conversation."""
        self.print_conversation_start(topic, max_turns)
        self.add_initial_topic(topic)

        for turn in range(1, max_turns + 1):
            try:
                if turn % 2 == 1:  # Odd turns: OpenAI speaks
                    logger.debug("conversation_turn_start", turn=turn, speaker="OpenAI")

                    # Send current conversation to OpenAI
                    response = await self.send_to_openai(self.openai_messages, turn)

                    # Add OpenAI's response to its conversation history
                    self.openai_messages.append(
                        {"role": "assistant", "content": response}
                    )

                    # Convert and add to Anthropic's history (with role swap)
                    # OpenAI's assistant response becomes user input for Anthropic
                    self.anthropic_messages.append(
                        {"role": "user", "content": response}
                    )

                    self.print_conversation_state(turn, "OpenAI (via proxy)", response)

                else:  # Even turns: Anthropic speaks
                    logger.debug(
                        "conversation_turn_start", turn=turn, speaker="Anthropic"
                    )

                    # Send current conversation to Anthropic
                    response = await self.send_to_anthropic(
                        self.anthropic_messages, turn
                    )

                    # Add Anthropic's response to its conversation history
                    self.anthropic_messages.append(
                        {"role": "assistant", "content": response}
                    )

                    # Convert and add to OpenAI's history (with role swap)
                    # Anthropic's assistant response becomes user input for OpenAI
                    self.openai_messages.append({"role": "user", "content": response})

                    self.print_conversation_state(
                        turn, "Anthropic (via proxy)", response
                    )

                # Small delay between turns
                await asyncio.sleep(1)

            except Exception as e:
                logger.error("conversation_turn_failed", turn=turn, error=str(e))
                print(f"\nError in turn {turn}: {e}")
                break

        self.print_conversation_end(min(turn, max_turns))

        # Log final conversation state
        logger.debug(
            "conversation_completed",
            total_turns=min(turn, max_turns),
            openai_messages=len(self.openai_messages),
            anthropic_messages=len(self.anthropic_messages),
        )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="OpenAI-to-Anthropic Bidirectional Conversation Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 openai_anthropic_conversation_demo.py
  python3 openai_anthropic_conversation_demo.py --topic "artificial intelligence ethics"
  python3 openai_anthropic_conversation_demo.py --turns 10 --debug
  python3 openai_anthropic_conversation_demo.py --stream --topic "robotics"
  python3 openai_anthropic_conversation_demo.py --plain --stream
        """,
    )
    parser.add_argument(
        "--topic",
        default="the future of technology",
        help="Topic for the conversation (default: 'the future of technology')",
    )
    parser.add_argument(
        "--turns",
        type=int,
        default=6,
        help="Maximum number of conversation turns (default: 6)",
    )
    parser.add_argument(
        "--proxy-url",
        default="http://127.0.0.1:8000/api",
        help="Proxy server URL (default: http://127.0.0.1:8000)",
    )
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="Enable debug logging (shows HTTP requests/responses)",
    )
    parser.add_argument(
        "-s",
        "--stream",
        action="store_true",
        help="Enable streaming mode (shows real-time responses)",
    )
    parser.add_argument(
        "-p",
        "--plain",
        action="store_true",
        help="Disable markdown formatting and use plain text output",
    )
    return parser.parse_args()


async def main() -> None:
    """Main function."""
    args = parse_args()
    setup_logging(debug=args.debug)

    # Check environment variables
    openai_key = os.getenv("OPENAI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")

    if not args.plain and RICH_AVAILABLE:
        console = Console()

        # Header
        console.print(
            "\n[bold blue]OpenAI-to-Anthropic Bidirectional Conversation Demo[/bold blue]"
        )
        console.print("=" * 60)

        # Warnings
        if not openai_key:
            console.print(
                "[yellow]Warning: OPENAI_API_KEY not set, using dummy key[/yellow]"
            )
        if not anthropic_key:
            console.print(
                "[yellow]Warning: ANTHROPIC_API_KEY not set, using dummy key[/yellow]"
            )

        # Configuration table
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Setting", style="bold cyan")
        table.add_column("Value", style="green")

        table.add_row("Topic:", args.topic)
        table.add_row("Max turns:", str(args.turns))
        table.add_row("Proxy URL:", args.proxy_url)

        if args.debug:
            table.add_row("Debug logging:", "enabled")
        if args.stream:
            table.add_row("Streaming:", "enabled")
        if args.plain:
            table.add_row("Markdown formatting:", "disabled")
        else:
            table.add_row("Markdown formatting:", "enabled (default)")

        panel = Panel(
            table,
            title="[bold magenta]Demo Configuration[/bold magenta]",
            border_style="magenta",
            title_align="left",
        )

        console.print(panel)
        console.print("=" * 60)

    else:
        # Fallback to plain text
        print("OpenAI-to-Anthropic Bidirectional Conversation Demo")
        print("=" * 60)

        if not openai_key:
            print("Warning: OPENAI_API_KEY not set, using dummy key")
        if not anthropic_key:
            print("Warning: ANTHROPIC_API_KEY not set, using dummy key")

        print(f"Topic: {args.topic}")
        print(f"Max turns: {args.turns}")
        print(f"Proxy URL: {args.proxy_url}")
        if args.debug:
            print("Debug logging: enabled")
        if args.stream:
            print("Streaming: enabled")
        if args.plain:
            print("Markdown formatting: disabled")
        else:
            print("Markdown formatting: enabled (default)")
        print("=" * 60)

    try:
        manager = ConversationManager(
            proxy_url=args.proxy_url,
            debug=args.debug,
            stream=args.stream,
            use_markdown=not args.plain,
        )
        await manager.run_conversation(args.topic, args.turns)

    except KeyboardInterrupt:
        if not args.plain and RICH_AVAILABLE:
            console = Console()
            console.print("\n[yellow]Conversation interrupted by user[/yellow]")
        else:
            print("\nConversation interrupted by user")
    except Exception as e:
        if not args.plain and RICH_AVAILABLE:
            console = Console()
            console.print(f"\n[bold red]Error:[/bold red] {e}")
            console.print(
                "[yellow]Make sure your proxy server is running and accessible[/yellow]"
            )
        else:
            print(f"\nError: {e}")
            print("Make sure your proxy server is running and accessible")
        logger.error("main_error", error=str(e))


if __name__ == "__main__":
    asyncio.run(main())
