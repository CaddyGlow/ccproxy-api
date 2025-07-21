import json
import logging
import re

import structlog


try:
    from rich.align import Align
    from rich.console import Console, Group
    from rich.live import Live
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.table import Table

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

logger = structlog.get_logger(__name__)


class RichConsoleManager:
    """A manager for printing rich output to the console."""

    def __init__(self, use_rich: bool = True):
        self.use_rich = use_rich and RICH_AVAILABLE
        self.console = Console() if self.use_rich else None

        if not RICH_AVAILABLE and use_rich:
            print("Warning: rich library not available, falling back to plain text")

    def print_header(self, title: str):
        if self.use_rich:
            self.console.print(Panel(title, style="bold blue", expand=False))
        else:
            print(f"=== {title} ===")

    def print_subheader(self, text: str):
        if self.use_rich:
            self.console.print(f"[bold green]{text}[/bold green]")
        else:
            print(f"--- {text} ---")

    def print_tools(self, tools: list):
        if not self.use_rich:
            print("\nGenerated Tools:")
            for tool in tools:
                print(json.dumps(tool, indent=2))
            return

        table = Table(title="Generated Tools")
        table.add_column("Name", style="cyan")
        table.add_column("Description", style="magenta")
        table.add_column("Schema", style="yellow")

        for tool in tools:
            tool_dict = tool if isinstance(tool, dict) else tool.model_dump()
            if "function" in tool_dict:
                func = tool_dict["function"]
                name = func.get("name", "N/A")
                desc = func.get("description", "N/A")
                schema = func.get("parameters", {})
            else:  # Anthropic format
                name = tool_dict.get("name", "N/A")
                desc = tool_dict.get("description", "N/A")
                schema = tool_dict.get("input_schema", {})

            table.add_row(name, desc, json.dumps(schema, indent=2))

        self.console.print(table)

    def print_tool_call(self, tool_name: str, tool_input: dict):
        if self.use_rich:
            panel = Panel(
                json.dumps(tool_input, indent=2),
                title=f"[bold cyan]Tool Call: {tool_name}[/bold cyan]",
                border_style="cyan",
            )
            self.console.print(panel)
        else:
            print(f"\nTool Call: {tool_name}")
            print(f"Input: {json.dumps(tool_input, indent=2)}")

    def print_tool_result(self, result: dict):
        if self.use_rich:
            panel = Panel(
                json.dumps(result, indent=2),
                title="[bold green]Tool Result[/bold green]",
                border_style="green",
            )
            self.console.print(panel)
        else:
            print(f"Result: {json.dumps(result, indent=2)}")

    def print_response(self, text: str, stop_reason: str):
        if self.use_rich:
            panel = Panel(
                Markdown(text),
                title="[bold magenta]Assistant Response[/bold magenta]",
                border_style="magenta",
                subtitle=f"Stop Reason: {stop_reason}",
            )
            self.console.print(panel)
        else:
            print(f"\nAssistant Response (Stop Reason: {stop_reason}):")
            print(text)

    def print_turn_separator(self, turn_number: int | None = None):
        if self.use_rich:
            if turn_number:
                self.console.rule(f"[bold] Turn {turn_number} [/bold]", style="blue")
            else:
                self.console.rule(style="blue")
        else:
            if turn_number:
                print(f"\n{'=' * 25} Turn {turn_number} {'=' * 25}\n")
            else:
                print(f"\n{'=' * 60}\n")

    def print_user_message(self, content: str):
        if self.use_rich:
            panel = Panel(
                Markdown(content),
                title="[bold blue]User Message[/bold blue]",
                border_style="blue",
            )
            self.console.print(panel)
        else:
            print(f"\nUser Message:\n{content}")

    def print_error(self, error_message: str):
        if self.use_rich:
            self.console.print(f"[bold red]Error:[/bold red] {error_message}")
        else:
            print(f"Error: {error_message}")

    def _get_live_panel(self, title: str) -> "Panel":
        return Panel(
            "",
            title=f"[bold magenta]{title}[/bold magenta]",
            border_style="magenta",
        )

    def _process_chunk(self, chunk, full_response):
        content = ""
        finish_reason = None
        logger.info("Processing chunk", chunk=chunk)
        if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
            content = chunk.choices[0].delta.content
            full_response += content
        if chunk.choices and chunk.choices[0].finish_reason:
            finish_reason = chunk.choices[0].finish_reason
        return full_response, content, finish_reason

    def print_streaming_response(self, stream, title="Assistant Response"):
        full_response = ""
        finish_reason = "unknown"

        if self.use_rich:
            panel = self._get_live_panel(title)
            with Live(
                panel,
                console=self.console,
                auto_refresh=False,
                vertical_overflow="visible",
            ) as live:
                for chunk in stream:
                    full_response, _, new_finish_reason = self._process_chunk(
                        chunk, full_response
                    )
                    if new_finish_reason:
                        finish_reason = new_finish_reason
                    panel.renderable = Markdown(full_response)
                    live.refresh()
                panel.subtitle = f"Stop Reason: {finish_reason}"
                live.refresh()
        else:
            print(f"\n{title} (streaming):")
            for chunk in stream:
                full_response, content, new_finish_reason = self._process_chunk(
                    chunk, full_response
                )
                if new_finish_reason:
                    finish_reason = new_finish_reason
                if content:
                    print(content, end="", flush=True)
            print()
        return full_response, finish_reason

    async def print_streaming_response_async(self, stream, title="Assistant Response"):
        full_response = ""
        finish_reason = "unknown"

        if self.use_rich:
            panel = self._get_live_panel(title)
            with Live(
                panel,
                console=self.console,
                auto_refresh=False,
                vertical_overflow="visible",
            ) as live:
                async for chunk in stream:
                    full_response, _, new_finish_reason = self._process_chunk(
                        chunk, full_response
                    )
                    if new_finish_reason:
                        finish_reason = new_finish_reason
                    panel.renderable = Markdown(full_response)
                    live.refresh()
                panel.subtitle = f"Stop Reason: {finish_reason}"
                live.refresh()
        else:
            print(f"\n{title} (streaming):")
            async for chunk in stream:
                full_response, content, new_finish_reason = self._process_chunk(
                    chunk, full_response
                )
                if new_finish_reason:
                    finish_reason = new_finish_reason
                if content:
                    print(content, end="", flush=True)
            print()
        return full_response, finish_reason


class ThinkingRenderer(RichConsoleManager):
    """Unified renderer for thinking blocks and responses with rich formatting."""

    def render_thinking_blocks(self, content: str) -> None:
        """Extract and render thinking blocks with rich formatting."""
        thinking_pattern = r'<thinking signature="([^"]*)">(.*?)</thinking>'
        matches = re.findall(thinking_pattern, content, re.DOTALL)

        if matches:
            for i, (signature, thinking_text) in enumerate(matches, 1):
                if self.use_rich:
                    self._render_thinking_block_rich(i, signature, thinking_text)
                else:
                    self._render_thinking_block_plain(i, signature, thinking_text)

    def _render_thinking_block_rich(
        self, index: int, signature: str, content: str
    ) -> None:
        """Render thinking block with rich formatting."""
        thinking_markdown = Markdown(content.strip())
        panel = Panel(
            Align.left(thinking_markdown),
            title=f"[bold yellow][THINKING BLOCK {index}][/bold yellow]",
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
        print(f"[THINKING BLOCK {index}]")
        print(f"Signature: {signature}")
        print(f"{'=' * 60}")
        print(content.strip())
        print(f"{'=' * 60}")

    def extract_visible_content(self, content: str) -> str:
        """Extract only the visible content (not thinking blocks)."""
        thinking_pattern = r'<thinking signature="[^"]*">.*?</thinking>'
        return re.sub(thinking_pattern, "", content, flags=re.DOTALL).strip()

    async def print_streaming_response_with_thinking_async(
        self, stream, title="Assistant Response"
    ):
        if not self.use_rich:
            # Fallback for plain text mode is the existing async streaming function
            full_content, _ = await self.print_streaming_response_async(stream, title)
            self.print_response(self.extract_visible_content(full_content), "streaming")
            self.render_thinking_blocks(full_content)
            return

        full_response = ""
        finish_reason = "unknown"
        live_group = Group()

        with Live(
            live_group,
            console=self.console,
            auto_refresh=False,
            vertical_overflow="visible",
        ) as live:
            async for chunk in stream:
                full_response, _, new_finish_reason = self._process_chunk(
                    chunk, full_response
                )
                if new_finish_reason:
                    finish_reason = new_finish_reason

                visible_content = ""
                thinking_blocks = []
                last_index = 0
                thinking_pattern = r'<thinking signature="([^"]*)">(.*?)</thinking>'
                open_thinking_pattern = r'<thinking signature="([^"]*)">(.*)'

                # Find all complete thinking blocks
                for match in re.finditer(thinking_pattern, full_response, re.DOTALL):
                    visible_content += full_response[last_index : match.start()]
                    signature = match.group(1)
                    thinking_text = match.group(2)
                    thinking_blocks.append((signature, thinking_text))
                    last_index = match.end()

                # Add the content after the last complete block
                remaining_content = full_response[last_index:]

                # Check if there's an open thinking block in the remaining content
                open_match = re.search(
                    open_thinking_pattern, remaining_content, re.DOTALL
                )
                if open_match:
                    # Add content before the open block to visible
                    visible_content += remaining_content[: open_match.start()]
                    # The open block itself
                    signature = open_match.group(1)
                    thinking_text = open_match.group(2)
                    thinking_blocks.append((signature, thinking_text))
                else:
                    # No open block, all remaining content is visible
                    visible_content += remaining_content

                # --- Update renderables ---
                panels = []

                # Thinking block panels
                for i, (signature, thinking_text) in enumerate(thinking_blocks):
                    thinking_panel = Panel(
                        Align.left(Markdown(thinking_text.strip())),
                        title=f"[bold yellow][THINKING BLOCK {i + 1}][/bold yellow]",
                        border_style="yellow",
                        title_align="left",
                        subtitle=f"[dim]Signature: {signature}[/dim]",
                        subtitle_align="right",
                    )
                    panels.append(thinking_panel)

                # Main response panel
                main_panel = Panel(
                    Markdown(visible_content.strip()),
                    title=f"[bold magenta]{title}[/bold magenta]",
                    border_style="magenta",
                )
                panels.append(main_panel)

                live.update(Group(*panels), refresh=True)

            # Final update with stop reason
            main_panel.subtitle = f"Stop Reason: {finish_reason}"
            live.refresh()
