#!/usr/bin/env python3
"""Rich CLI client for Claude Proxy API Server."""

import asyncio
import json
import sys
from collections.abc import AsyncGenerator
from typing import Any, Optional, cast

import httpx
import typer
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.spinner import Spinner
from rich.status import Status


app = typer.Typer()
console = Console()


class ClaudeProxyClient:
    """Client for Claude Proxy API Server."""

    def __init__(
        self, base_url: str = "http://localhost:8000", api_key: str | None = None
    ):
        self.base_url = base_url.rstrip("/")
        # self.api_key = api_key
        self.client = httpx.AsyncClient(timeout=300.0)

    async def __aenter__(self) -> "ClaudeProxyClient":
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.client.aclose()

    async def health_check(self) -> bool:
        """Check if the server is healthy."""
        try:
            response = await self.client.get(f"{self.base_url}/health")
            return response.status_code == 200
        except Exception:
            return False

    async def chat_completion_stream(
        self,
        messages: list[dict[str, Any]],
        model: str = "claude-3-5-sonnet-20241022",
        max_tokens: int = 4096,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Send a streaming chat completion request."""
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": True,
        }

        headers = {"Content-Type": "application/json"}
        # if self.api_key:
        #     headers["x-api-key"] = self.api_key

        async with self.client.stream(
            "POST",
            f"{self.base_url}/v1/chat/completions",
            json=payload,
            headers=headers,
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data.strip() == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        yield chunk
                    except json.JSONDecodeError:
                        continue

    async def chat_completion(
        self,
        messages: list[dict[str, Any]],
        model: str = "claude-3-5-sonnet-20241022",
        max_tokens: int = 4096,
    ) -> dict[str, Any]:
        """Send a non-streaming chat completion request."""
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": False,
        }

        headers = {"Content-Type": "application/json"}
        # if self.api_key:
        #     headers["x-api-key"] = self.api_key

        response = await self.client.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload,
            headers=headers,
        )
        response.raise_for_status()
        return cast(dict[str, Any], response.json())


async def stream_response(
    client: ClaudeProxyClient, messages: list[dict[str, Any]]
) -> str | None:
    """Stream and display the response from Claude."""
    content = ""

    with Live(console=console, refresh_per_second=10) as live:
        try:
            async for chunk in client.chat_completion_stream(messages):
                if chunk.get("type") == "content_block_delta":
                    delta = chunk.get("delta", {})
                    if delta.get("type") == "text_delta":
                        text = delta.get("text", "")
                        content += text

                        # Update the live display
                        if content.strip():
                            live.update(
                                Panel(
                                    Markdown(content),
                                    title="Claude",
                                    title_align="left",
                                    border_style="blue",
                                )
                            )
        except Exception as e:
            console.print(f"[red]Error during streaming: {e}[/red]")
            return None

    return content


@app.command()
def chat(
    server_url: str = typer.Option(
        "http://localhost:8000",
        "--server",
        "-s",
        help="Claude proxy server URL",
    ),
    api_key: str | None = typer.Option(
        None,
        "--api-key",
        "-k",
        help="API key for authentication",
    ),
    model: str = typer.Option(
        "claude-3-5-sonnet-20241022",
        "--model",
        "-m",
        help="Claude model to use",
    ),
    stream: bool = typer.Option(
        True,
        "--stream/--no-stream",
        help="Enable streaming responses",
    ),
) -> None:
    """Start an interactive chat session with Claude via the proxy server."""
    asyncio.run(chat_session(server_url, api_key, model, stream))


async def chat_session(
    server_url: str, api_key: str | None, model: str, stream: bool
) -> None:
    """Run the interactive chat session."""
    console.print(
        Panel.fit(
            "[bold blue]Claude Proxy Chat Client[/bold blue]\n"
            f"Server: {server_url}\n"
            f"Model: {model}\n"
            f"Streaming: {'enabled' if stream else 'disabled'}\n\n"
            "[dim]Type 'quit', 'exit', or Ctrl+C to end the session[/dim]",
            border_style="blue",
        )
    )

    async with ClaudeProxyClient(server_url, api_key) as client:
        # Health check
        with Status("Checking server health...", spinner="dots"):
            if not await client.health_check():
                console.print("[red]Error: Cannot connect to Claude proxy server[/red]")
                return

        console.print("[green]✓ Connected to Claude proxy server[/green]\n")

        messages = []

        while True:
            try:
                # Get user input
                user_input = Prompt.ask("\n[bold green]You[/bold green]")

                if user_input.lower() in ["quit", "exit", "q"]:
                    console.print("[yellow]Goodbye![/yellow]")
                    break

                if not user_input.strip():
                    continue

                # Add user message
                messages.append({"role": "user", "content": user_input})

                # Get Claude's response
                console.print()
                if stream:
                    response_content = await stream_response(client, messages)
                    if response_content:
                        messages.append(
                            {"role": "assistant", "content": response_content}
                        )
                else:
                    with Status("Thinking...", spinner="dots"):
                        response = await client.chat_completion(messages)

                    content = response["content"][0]["text"]
                    messages.append({"role": "assistant", "content": content})

                    console.print(
                        Panel(
                            Markdown(content),
                            title="Claude",
                            title_align="left",
                            border_style="blue",
                        )
                    )

            except KeyboardInterrupt:
                console.print("\n[yellow]Session interrupted. Goodbye![/yellow]")
                break
            except Exception as e:
                console.print(f"\n[red]Error: {e}[/red]")


@app.command()
def test(
    server_url: str = typer.Option(
        "http://localhost:8000",
        "--server",
        "-s",
        help="Claude proxy server URL",
    ),
    api_key: str | None = typer.Option(
        None,
        "--api-key",
        "-k",
        help="API key for authentication",
    ),
) -> None:
    """Test the connection to the Claude proxy server."""
    asyncio.run(test_connection(server_url, api_key))


async def test_connection(server_url: str, api_key: str | None) -> None:
    """Test the connection to the server."""
    console.print(f"Testing connection to {server_url}...")

    async with ClaudeProxyClient(server_url, api_key) as client:
        # Health check
        if await client.health_check():
            console.print("[green]✓ Server is healthy[/green]")
        else:
            console.print("[red]✗ Server health check failed[/red]")
            return

        # Test a simple completion
        try:
            messages = [{"role": "user", "content": "Say hello in one word."}]
            with Status("Testing completion...", spinner="dots"):
                response = await client.chat_completion(messages)

            console.print("[green]✓ Completion test successful[/green]")
            console.print(f"Response: {response['content'][0]['text']}")

        except Exception as e:
            console.print(f"[red]✗ Completion test failed: {e}[/red]")


if __name__ == "__main__":
    app()
