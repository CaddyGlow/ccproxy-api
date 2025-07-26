#!/usr/bin/env python3
"""
Textual Chat Agent with Vim Mode Input

A simple chat agent using Anthropic SDK with streaming responses
and a textual GUI with vim mode for input.
"""

import asyncio
import os
from collections.abc import AsyncGenerator
from typing import Any

from anthropic import AsyncAnthropic
from anthropic.types import MessageParam
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal
from textual.events import Key
from textual.widgets import Button, Input, RichLog, Static


class ChatAgent:
    """Simple chat agent using Anthropic SDK."""

    def __init__(self, api_key: str | None = None) -> None:
        # Get configuration from environment
        api_key = os.getenv("ANTHROPIC_API_KEY")
        base_url = os.getenv("ANTHROPIC_BASE_URL")
        base_url_default = "http://127.0.0.1:8000/sdk"

        if not api_key:
            # logger.warning("ANTHROPIC_API_KEY not set, using dummy key")
            os.environ["ANTHROPIC_API_KEY"] = "dummy"
        if not base_url:
            # logger.warning(f"ANTHROPIC_BASE_URL not set, using {base_url_default}")
            os.environ["ANTHROPIC_BASE_URL"] = base_url_default

        # Create client with type-safe parameters
        # Don't modify the base URL - let the proxy server handle the path routing
        if base_url:
            self.client = AsyncAnthropic(api_key=api_key, base_url=base_url)
        else:
            self.client = AsyncAnthropic(api_key=api_key)

        self.messages: list[MessageParam] = []

        # Store config for debugging
        self.api_key = api_key
        self.base_url = base_url

    async def send_message(self, message: str) -> AsyncGenerator[str, None]:
        """Send a message and stream the response."""
        self.messages.append({"role": "user", "content": message})

        try:
            yield f"[DEBUG] Starting request to {self.client.base_url or 'https://api.anthropic.com'}"
            yield f"[DEBUG] Using API key: {self.api_key[:10] + '...' if self.api_key else 'None'}"

            # Process streaming events for real-time response
            async with self.client.messages.stream(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1024,
                messages=self.messages,
            ) as stream:
                response_content = ""
                thinking_content = ""
                in_thinking_block = False
                event_count = 0

                # Try to get the accumulated text first
                try:
                    final_message = await stream.get_final_message()
                    yield f"[DEBUG] Final message received: {len(final_message.content)} content blocks"

                    for content_block in final_message.content:
                        if content_block.type == "text":
                            response_content = content_block.text
                            yield f"[DEBUG] Text content: {response_content[:100]}{'...' if len(response_content) > 100 else ''}"
                            # Yield the full text
                            yield response_content
                except Exception as e:
                    yield f"[DEBUG] Failed to get final message: {e}"

                    # Fall back to processing raw events
                    async for event in stream:
                        event_count += 1
                        yield f"[DEBUG] Event {event_count}: {event.type}"

                        if event.type == "content_block_start":
                            content_block = event.content_block
                            yield f"[DEBUG] Content block: {content_block.type}"
                            if (
                                content_block.type == "text"
                                and hasattr(content_block, "thinking")
                                and content_block.thinking
                            ):
                                in_thinking_block = True
                                yield "[THINKING_START]"

                        elif event.type == "content_block_delta":
                            delta = event.delta
                            yield f"[DEBUG] Delta type: {getattr(delta, 'type', 'no type')}"
                            if (
                                hasattr(delta, "type")
                                and delta.type == "thinking_delta"
                            ):
                                thinking_text = getattr(delta, "thinking", "")
                                thinking_content += thinking_text
                                yield thinking_text
                            elif hasattr(delta, "type") and delta.type == "text_delta":
                                text = getattr(delta, "text", "")
                                response_content += text
                                if in_thinking_block:
                                    in_thinking_block = False
                                    yield "[THINKING_END]"
                                yield text

                        elif event.type == "content_block_stop":
                            if in_thinking_block:
                                in_thinking_block = False
                                yield "[THINKING_END]"

                yield f"[DEBUG] Stream completed. Total events: {event_count}, Response length: {len(response_content)}"

                # Store the complete response (without thinking content)
                self.messages.append({"role": "assistant", "content": response_content})

        except Exception as e:
            yield f"Error: {type(e).__name__}: {str(e)}"
            # Also yield more detailed error info if available
            if hasattr(e, "response") and e.response:
                yield f"\nResponse status: {e.response.status_code}"
                yield f"Response headers: {dict(e.response.headers)}"
                yield f"Response body: {e.response.text}"
            if hasattr(e, "__cause__") and e.__cause__:
                yield f"\nUnderlying cause: {type(e.__cause__).__name__}: {str(e.__cause__)}"


class VimInput(Input):
    """Input widget with vim mode support."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.vim_mode = False
        self.vim_command = ""

    def on_key(self, event: Key) -> None:
        """Handle vim mode key presses."""
        # Handle Shift+Enter for new lines in insert mode
        if event.key == "shift+enter" and not self.vim_mode:
            # Insert a newline at cursor position
            cursor_pos = self.cursor_position
            self.value = self.value[:cursor_pos] + "\n" + self.value[cursor_pos:]
            self.cursor_position = cursor_pos + 1
            return

        if event.key == "escape":
            self.vim_mode = True
            self.vim_command = ""
            # Update the app's status bar
            self.app.update_status_bar(self.vim_mode)
            return

        if self.vim_mode:
            if event.key == "i":
                self.vim_mode = False
                self.app.update_status_bar(self.vim_mode)
                return
            elif event.key == "a":
                self.vim_mode = False
                self.cursor_position = len(self.value)
                self.app.update_status_bar(self.vim_mode)
                return
            elif event.key == "o":
                self.vim_mode = False
                self.value += "\n"
                self.cursor_position = len(self.value)
                self.app.update_status_bar(self.vim_mode)
                return
            elif event.key == "x":
                if self.cursor_position < len(self.value):
                    self.value = (
                        self.value[: self.cursor_position]
                        + self.value[self.cursor_position + 1 :]
                    )
                return
            elif event.key == "h":
                if self.cursor_position > 0:
                    self.cursor_position -= 1
                return
            elif event.key == "l":
                if self.cursor_position < len(self.value):
                    self.cursor_position += 1
                return
            elif event.key == "0":
                self.cursor_position = 0
                return
            elif event.key == "dollar":
                self.cursor_position = len(self.value)
                return
            # In vim mode, don't process other keys
            return

        # If not in vim mode, let the input handle normally
        # Don't call super().on_key() as it doesn't exist


class ChatApp(App):
    """Main chat application."""

    CSS = """
    .chat-container {
        height: 100%;
    }

    .chat-log {
        height: 1fr;
        border: solid $primary;
        padding: 1;
        margin: 1;
    }

    .input-container {
        height: auto;
        margin: 1;
    }

    .vim-input {
        height: 3;
        margin: 1;
    }

    .send-button {
        height: 3;
        margin: 1;
    }

    .status-bar {
        height: 1;
        background: $primary;
        color: $text;
        padding: 0 1;
    }

    .vim-mode {
        background: $warning;
        color: $text;
    }

    .insert-mode {
        background: $success;
        color: $text;
    }
    """

    def __init__(self) -> None:
        super().__init__()
        self.chat_agent = ChatAgent()
        self.is_sending = False

    def compose(self) -> ComposeResult:
        """Compose the UI."""
        yield Container(
            RichLog(classes="chat-log", highlight=True, markup=True, wrap=True),
            Horizontal(
                VimInput(
                    placeholder="Type your message (ESC for vim mode, i/a/o to insert)",
                    classes="vim-input",
                ),
                Button("Send", id="send-btn", classes="send-button"),
                classes="input-container",
            ),
            Static(
                "INSERT MODE | ESC: vim mode | Shift+Enter: new line | Ctrl+C: quit",
                classes="status-bar insert-mode",
            ),
            classes="chat-container",
        )

    def on_mount(self) -> None:
        """Initialize the app."""
        self.chat_log = self.query_one(RichLog)
        self.input_field = self.query_one(VimInput)
        self.send_button = self.query_one("#send-btn", Button)
        self.status_bar = self.query_one(Static)

        # Welcome message
        self.chat_log.write("[bold blue]Chat Agent[/bold blue] - Ready to chat!")
        self.chat_log.write(
            "[dim]Use ESC to enter vim mode, then i/a/o to insert text. Shift+Enter for new lines. Responses support markdown formatting.[/dim]"
        )

        # Show configuration
        api_key_display = "Not set"
        if self.chat_agent.api_key:
            if len(self.chat_agent.api_key) > 14:
                api_key_display = (
                    f"{self.chat_agent.api_key[:10]}...{self.chat_agent.api_key[-4:]}"
                )
            else:
                api_key_display = self.chat_agent.api_key
        self.chat_log.write(f"[dim]API Key: {api_key_display}[/dim]")

        if self.chat_agent.base_url:
            self.chat_log.write(f"[dim]Base URL: {self.chat_agent.base_url}[/dim]")
        else:
            self.chat_log.write(
                "[dim]Base URL: Default (https://api.anthropic.com)[/dim]"
            )
        self.chat_log.write("")

        # Set focus to input
        self.input_field.focus()

    def on_key(self, event: Key) -> None:
        """Handle global key events."""
        if event.key == "ctrl+c" or event.key == "ctrl+q":
            self.exit()

    def update_status_bar(self, vim_mode: bool) -> None:
        """Update the status bar based on vim mode."""
        if vim_mode:
            self.status_bar.update(
                "VIM MODE | i: insert | a: append | o: new line | h/l: move | x: delete | 0/$: start/end"
            )
            self.status_bar.remove_class("insert-mode")
            self.status_bar.add_class("vim-mode")
        else:
            self.status_bar.update(
                "INSERT MODE | ESC: vim mode | Shift+Enter: new line | Ctrl+C: quit"
            )
            self.status_bar.remove_class("vim-mode")
            self.status_bar.add_class("insert-mode")

    def on_input_submitted(self, event: Any) -> None:
        """Handle input submission."""
        if not self.is_sending and event.input.value.strip():
            self.send_message(event.input.value)

    def on_button_pressed(self, event: Any) -> None:
        """Handle button press."""
        if event.button.id == "send-btn" and not self.is_sending:
            message = self.input_field.value.strip()
            if message:
                self.send_message(message)

    def send_message(self, message: str) -> None:
        """Send a message to the chat agent."""
        if self.is_sending:
            return

        self.is_sending = True
        self.send_button.disabled = True

        # Display user message
        self.chat_log.write(f"[bold green]You:[/bold green] {message}")
        self.chat_log.write("")

        # Clear input
        self.input_field.value = ""

        # Send message asynchronously
        asyncio.create_task(self._handle_response(message))

    async def _handle_response(self, message: str) -> None:
        """Handle the streaming response."""
        try:
            # Start the response
            response_content = ""
            thinking_content = ""
            in_thinking = False
            chunk_count = 0
            self.chat_log.write("[bold blue]Claude:[/bold blue]")

            async for chunk in self.chat_agent.send_message(message):
                chunk_count += 1
                # Debug: show we're receiving chunks
                if chunk_count == 1:
                    self.chat_log.write(
                        f"[dim]Receiving response... (chunk {chunk_count})[/dim]"
                    )

                if chunk == "[THINKING_START]":
                    in_thinking = True
                    self.chat_log.write("[dim italic]Thinking...[/dim italic]")
                    continue
                elif chunk == "[THINKING_END]":
                    in_thinking = False
                    if thinking_content.strip():
                        # Display thinking content in a special format
                        self.chat_log.write(f"[dim]{thinking_content}[/dim]")
                        self.chat_log.write("[dim]---[/dim]")
                        thinking_content = ""
                    continue

                if in_thinking:
                    thinking_content += chunk
                else:
                    response_content += chunk
                    # Display chunk immediately (without markdown for real-time)
                    if chunk.strip():  # Only display non-empty chunks
                        self.chat_log.write(chunk)

            self.chat_log.write(
                f"[dim]Stream completed. Total chunks: {chunk_count}, Content length: {len(response_content)}[/dim]"
            )

            # If no content was received, show a message
            if not response_content.strip():
                self.chat_log.write("[dim]No response content received[/dim]")

            self.chat_log.write("")

        except Exception as e:
            self.chat_log.write(f"[bold red]Error:[/bold red] {str(e)}")
            # Add more debugging info
            import traceback

            self.chat_log.write(f"[red]Traceback:[/red] {traceback.format_exc()}")
            self.chat_log.write("")

        finally:
            self.is_sending = False
            self.send_button.disabled = False
            self.input_field.focus()


def main() -> None:
    """Run the chat application."""
    # Check for API key
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("Please set your ANTHROPIC_API_KEY environment variable")
        return

    app = ChatApp()
    app.run()


if __name__ == "__main__":
    main()
