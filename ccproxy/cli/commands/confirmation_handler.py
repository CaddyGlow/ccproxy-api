"""CLI command for handling confirmation requests via SSE stream."""

import asyncio
import contextlib
import json
from collections.abc import AsyncIterator
from typing import Any, Optional

import httpx
import typer
from rich.console import Console
from structlog import get_logger

from ccproxy.api.services.confirmation_service import ConfirmationRequest
from ccproxy.api.ui.terminal_confirmation_handler import TerminalConfirmationHandler
from ccproxy.config.settings import get_settings


logger = get_logger(__name__)
console = Console()

app = typer.Typer(
    name="confirmation-handler",
    help="Connect to the API server and handle confirmation requests",
    no_args_is_help=True,
)


class SSEConfirmationHandler:
    """Handles confirmation requests received via SSE stream."""

    def __init__(
        self,
        api_url: str,
        terminal_handler: TerminalConfirmationHandler,
        ui: bool = True,
    ):
        """Initialize the SSE handler.

        Args:
            api_url: Base URL of the API server
            terminal_handler: Terminal UI handler for displaying confirmations
        """
        self.api_url = api_url.rstrip("/")
        self.terminal_handler = terminal_handler
        self.client = httpx.AsyncClient(timeout=300.0)  # 5 minutes
        self.max_retries = 5
        self.base_delay = 1.0
        self.max_delay = 60.0
        self.ui = ui

        # Track ongoing confirmation requests to allow cancellation
        self._ongoing_requests: dict[str, asyncio.Task[bool]] = {}
        self._resolved_requests: dict[
            str, tuple[bool, str]
        ] = {}  # request_id -> (allowed, reason)
        self._resolved_by_us: set[str] = (
            set()
        )  # Track requests resolved by this handler

    async def handle_event(self, event_type: str, data: dict[str, Any]) -> None:
        """Handle an SSE event.

        Args:
            event_type: Type of the event
            data: Event data
        """
        logger.debug(
            "sse_event_received",
            event_type=event_type,
            data_keys=list(data.keys()) if isinstance(data, dict) else "non-dict",
        )

        if event_type == "ping":
            logger.debug("sse_ping_received", message=data.get("message"))
            return

        if event_type == "confirmation_request":
            request_id = data.get("request_id")

            # Check if this request was already resolved by another handler
            if request_id in self._resolved_requests:
                allowed, reason = self._resolved_requests[request_id]
                logger.info(
                    "confirmation_already_resolved_by_other_handler",
                    request_id=request_id,
                    allowed=allowed,
                    reason=reason,
                )
                console.print(
                    f"[yellow]Request {request_id[:8]}... already {reason}[/yellow]\n"
                )
                return

            logger.info(
                "confirmation_request_received",
                request_id=request_id,
                tool_name=data.get("tool_name"),
            )

            # Create a ConfirmationRequest object from the event data
            from datetime import datetime

            request = ConfirmationRequest(
                id=data["request_id"],
                tool_name=data["tool_name"],
                input=data["input"],
                created_at=datetime.fromisoformat(data["created_at"]),
                expires_at=datetime.fromisoformat(data["expires_at"]),
            )

            # Create a task to handle the confirmation and track it
            if self.ui and request_id is not None:
                task = asyncio.create_task(
                    self._handle_confirmation_with_cancellation(request)
                )
                self._ongoing_requests[request_id] = task

                # Don't await the task here - let it run in the background
                # This allows the SSE stream to continue processing events
                logger.debug(
                    "confirmation_task_started",
                    request_id=request_id,
                )

        elif event_type == "confirmation_resolved":
            request_id = data.get("request_id")
            allowed = data.get("allowed", False)

            # Only process if we have valid data
            if request_id is not None and allowed is not None:
                # Store the resolution for any future requests
                reason = (
                    "approved by another handler"
                    if allowed
                    else "denied by another handler"
                )
                self._resolved_requests[request_id] = (allowed, reason)

            # Check if this handler resolved it
            was_resolved_by_us = (
                request_id is not None and request_id in self._resolved_by_us
            )

            # Cancel any ongoing handling of this request
            if request_id is not None and request_id in self._ongoing_requests:
                task = self._ongoing_requests[request_id]
                if not task.done() and not was_resolved_by_us:
                    logger.info(
                        "cancelling_ongoing_confirmation",
                        request_id=request_id,
                        allowed=allowed,
                    )

                    # Cancel the terminal handler prompt
                    status_text = "approved" if allowed else "denied"
                    self.terminal_handler.cancel_confirmation(
                        request_id, f"{status_text} by another handler"
                    )

                    # Cancel the task
                    task.cancel()

                    # Wait a moment for the task to be cancelled
                    with contextlib.suppress(TimeoutError, asyncio.CancelledError):
                        await asyncio.wait_for(task, timeout=0.1)

                    # Now show the message after cancellation
                    console.print(
                        f"\n[yellow]✗ Request {request_id[:8]}... {status_text} by another handler[/yellow]\n"
                    )
                else:
                    # Task is already done or we resolved it
                    logger.debug(
                        "confirmation_resolved_by_this_handler",
                        request_id=request_id,
                        allowed=allowed,
                        was_resolved_by_us=was_resolved_by_us,
                    )

                # Always clean up the task reference after handling resolution
                if request_id is not None:
                    self._ongoing_requests.pop(request_id, None)
            else:
                logger.debug(
                    "confirmation_resolved_no_ongoing_request",
                    request_id=request_id,
                    allowed=allowed,
                )

            # Clean up our tracking
            if request_id is not None:
                self._resolved_by_us.discard(request_id)

    async def _handle_confirmation_with_cancellation(
        self, request: ConfirmationRequest
    ) -> bool:
        """Handle confirmation with cancellation support.

        Args:
            request: The confirmation request to handle
        """
        try:
            # Handle the confirmation using the terminal UI
            allowed = await self.terminal_handler.handle_confirmation(request)

            # Check if request was cancelled/resolved while we were processing
            if request.id in self._resolved_requests:
                logger.info(
                    "confirmation_resolved_while_processing",
                    request_id=request.id,
                    our_result=allowed,
                )
                return False  # Don't send response, another handler already did

            # Mark that we resolved this request
            self._resolved_by_us.add(request.id)

            # Send the response back to the API
            await self.send_response(request.id, allowed)

            # Keep the task reference alive briefly to allow other handlers to be cancelled
            # The confirmation_resolved event should arrive soon and clean this up
            await asyncio.sleep(0.5)

            return allowed

        except asyncio.CancelledError:
            # Request was cancelled by another handler resolving it
            logger.info(
                "confirmation_cancelled",
                request_id=request.id,
            )
            raise  # Re-raise to properly handle cancellation

        except Exception as e:
            logger.error(
                "confirmation_handling_error",
                request_id=request.id,
                error=str(e),
                exc_info=True,
            )
            # Send deny response on error (if not already resolved)
            if request.id not in self._resolved_requests:
                await self.send_response(request.id, False)
            return False

    async def send_response(self, request_id: str, allowed: bool) -> None:
        """Send a confirmation response to the API.

        Args:
            request_id: ID of the confirmation request
            allowed: Whether to allow or deny
        """
        try:
            response = await self.client.post(
                f"{self.api_url}/api/v1/confirmations/{request_id}/respond",
                json={"allowed": allowed},
            )

            if response.status_code == 200:
                logger.info(
                    "confirmation_response_sent",
                    request_id=request_id,
                    allowed=allowed,
                )
            else:
                logger.error(
                    "confirmation_response_failed",
                    request_id=request_id,
                    status_code=response.status_code,
                    response=response.text,
                )

        except Exception as e:
            logger.error(
                "confirmation_response_error",
                request_id=request_id,
                error=str(e),
                exc_info=True,
            )

    async def parse_sse_stream(
        self, response: httpx.Response
    ) -> AsyncIterator[tuple[str, dict[str, Any]]]:
        """Parse SSE events from the response stream.

        Args:
            response: The httpx response with streaming content

        Yields:
            Tuples of (event_type, data)
        """
        buffer = ""
        async for chunk in response.aiter_text():
            buffer += chunk

            # Normalize line endings - replace \r\n with \n
            buffer = buffer.replace("\r\n", "\n")

            # Process complete events (SSE events are separated by double newlines)
            while "\n\n" in buffer:
                event_text, buffer = buffer.split("\n\n", 1)

                # Skip empty events
                if not event_text.strip():
                    continue

                # Parse event
                event_type = "message"
                data_lines = []

                for line in event_text.split("\n"):
                    line = line.strip()
                    if line.startswith("event:"):
                        event_type = line[6:].strip()
                    elif line.startswith("data:"):
                        data_lines.append(line[5:].strip())

                if data_lines:
                    try:
                        data_json = " ".join(data_lines)
                        data = json.loads(data_json)
                        logger.debug(
                            "sse_event_parsed",
                            event_type=event_type,
                            data_preview=str(data)[:100] + "..."
                            if len(str(data)) > 100
                            else str(data),
                        )
                        yield event_type, data
                    except json.JSONDecodeError as e:
                        logger.error(
                            "sse_parse_error",
                            event_type=event_type,
                            data=" ".join(data_lines),
                            error=str(e),
                        )

    async def run(self) -> None:
        """Run the SSE client with reconnection logic."""
        stream_url = f"{self.api_url}/api/v1/confirmations/stream"
        retry_count = 0

        console.print(
            f"[cyan]Connecting to confirmation stream at {stream_url}...[/cyan]"
        )

        try:
            while retry_count <= self.max_retries:
                try:
                    await self._connect_and_handle_stream(stream_url)
                    # If we get here, connection ended gracefully
                    break

                except KeyboardInterrupt:
                    console.print(
                        "\n[yellow]Shutting down confirmation handler...[/yellow]"
                    )
                    break

                except (
                    httpx.ConnectError,
                    httpx.TimeoutException,
                    httpx.ReadTimeout,
                ) as e:
                    retry_count += 1
                    if retry_count > self.max_retries:
                        console.print(
                            f"[red]Failed to connect after {self.max_retries} attempts[/red]\n"
                            "[yellow]Make sure the API server is running.[/yellow]"
                        )
                        raise typer.Exit(1) from None

                    # Exponential backoff with jitter
                    delay = min(
                        self.base_delay * (2 ** (retry_count - 1)), self.max_delay
                    )

                    logger.warning(
                        "connection_failed_retrying",
                        attempt=retry_count,
                        max_retries=self.max_retries,
                        retry_delay=delay,
                        error=str(e),
                    )

                    console.print(
                        f"[yellow]Connection failed (attempt {retry_count}/{self.max_retries}). "
                        f"Retrying in {delay:.1f}s...[/yellow]"
                    )

                    await asyncio.sleep(delay)
                    continue

                except Exception as e:
                    logger.error("sse_client_error", error=str(e), exc_info=True)
                    console.print(f"[red]Unexpected error: {e}[/red]")
                    raise typer.Exit(1) from e

        finally:
            await self.client.aclose()

    async def _connect_and_handle_stream(self, stream_url: str) -> None:
        """Connect to the stream and handle events."""
        async with self.client.stream("GET", stream_url) as response:
            if response.status_code != 200:
                error_text = ""
                try:
                    error_bytes = await response.aread()
                    error_text = error_bytes.decode("utf-8")
                except Exception:
                    error_text = "Unable to read error response"

                logger.error(
                    "sse_connection_failed",
                    status_code=response.status_code,
                    response=error_text,
                )

                if response.status_code in (502, 503, 504):
                    # Server errors - retry
                    raise httpx.ConnectError(
                        f"Server error: HTTP {response.status_code}"
                    )
                else:
                    # Client errors - don't retry
                    console.print(
                        f"[red]Failed to connect: HTTP {response.status_code}[/red]\n"
                        f"Response: {error_text}"
                    )
                    raise typer.Exit(1)

            console.print("[green]✓ Connected to confirmation stream[/green]")
            console.print("[yellow]Waiting for confirmation requests...[/yellow]\n")

            logger.info("sse_connection_established", url=stream_url)

            async for event_type, data in self.parse_sse_stream(response):
                try:
                    await self.handle_event(event_type, data)
                except Exception as e:
                    logger.error(
                        "sse_event_error",
                        event_type=event_type,
                        error=str(e),
                        exc_info=True,
                    )


@app.command()
def connect(
    api_url: str | None = typer.Option(
        None,
        "--api-url",
        "-u",
        help="API server URL (defaults to settings)",
    ),
    ui: bool = typer.Option(None, "--ui", "-u", help="Enable UI mode"),
) -> None:
    """Connect to the API server and handle confirmation requests.

    This command connects to the CCProxy API server via Server-Sent Events
    and handles permission confirmation requests in the terminal.
    """
    settings = get_settings()

    # Use provided URL or default from settings
    if not api_url:
        api_url = f"http://{settings.server.host}:{settings.server.port}"

    # Create handlers
    terminal_handler = TerminalConfirmationHandler()
    sse_handler = SSEConfirmationHandler(api_url, terminal_handler, ui)

    # Run the async handler
    try:
        asyncio.run(sse_handler.run())
    except KeyboardInterrupt:
        console.print("\n[green]Confirmation handler stopped.[/green]")
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        raise typer.Exit(1) from e


if __name__ == "__main__":
    app()
