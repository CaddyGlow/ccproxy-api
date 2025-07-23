"""Terminal UI handler for confirmation requests using Rich."""

import asyncio
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm
from rich.table import Table
from structlog import get_logger

from ccproxy.api.services.confirmation_service import ConfirmationRequest


logger = get_logger(__name__)


class TerminalConfirmationHandler:
    """Handles confirmation requests in the terminal using Rich UI."""

    def __init__(self) -> None:
        self._console = Console()
        self._cancellation_events: dict[str, asyncio.Event] = {}

    def cancel_confirmation(self, request_id: str, reason: str = "cancelled") -> None:
        """Cancel an ongoing confirmation request.

        Args:
            request_id: ID of the request to cancel
            reason: Reason for cancellation (for display)
        """
        if request_id in self._cancellation_events:
            self._cancellation_events[request_id].set()

    async def handle_confirmation(self, request: ConfirmationRequest) -> bool:
        """Handle a confirmation request by prompting the user.

        Args:
            request: The confirmation request to handle

        Returns:
            True if approved, False if denied
        """
        cancellation_event = asyncio.Event()
        self._cancellation_events[request.id] = cancellation_event

        try:
            self._display_request(request)

            result = await self._get_user_confirmation_with_cancellation(
                request, cancellation_event
            )

            if not cancellation_event.is_set():
                self._display_result(result)

            return result

        except asyncio.CancelledError:
            # Don't print here - the cancellation message was already shown
            raise

        except Exception as e:
            logger.error(
                "terminal_confirmation_error",
                request_id=request.id,
                error=str(e),
                exc_info=True,
            )
            self._console.print("[red]Error handling confirmation request[/red]\n")
            return False  # Deny on error

        finally:
            self._cancellation_events.pop(request.id, None)

    def _display_request(self, request: ConfirmationRequest) -> None:
        """Display the confirmation request to the user."""
        table = Table(show_header=False, box=None)
        table.add_column(style="bold cyan", width=20)
        table.add_column()

        table.add_row("Tool", request.tool_name)
        table.add_row("Request ID", request.id[:8] + "...")

        for key, value in request.input.items():
            value_str = self._format_value(value)
            table.add_row(f"  {key}", value_str)

        panel = Panel(
            table,
            title="[bold red]!  Permission Request[/bold red]",
            border_style="red",
            expand=False,
        )

        self._console.print("\n")
        self._console.print(panel)

        time_remaining = request.time_remaining()
        self._console.print(
            f"\n[yellow]Timeout in {time_remaining} seconds "
            f"(auto-deny if no response)[/yellow]"
        )

    def _format_value(self, value: Any) -> str:
        """Format a value for display."""
        s = str(value)
        return s if len(s) <= 60 else s[:57] + "..."

    async def _get_user_confirmation_with_cancellation(
        self, request: ConfirmationRequest, cancellation_event: asyncio.Event
    ) -> bool:
        """Get user confirmation with timeout and cancellation handling."""
        loop = asyncio.get_event_loop()

        self._console.print(
            "\n[dim]Note: This prompt will auto-cancel if another handler responds[/dim]"
        )

        def prompt_user() -> bool:
            try:
                return Confirm.ask(
                    "\n[bold]Allow this operation?[/bold]",
                    default=False,
                )
            except (KeyboardInterrupt, EOFError):
                return False

        # run_in_executor returns a Future, not a coroutine
        prompt_task = loop.run_in_executor(None, prompt_user)

        timeout_task = asyncio.create_task(asyncio.sleep(request.time_remaining()))

        async def wait_for_cancellation() -> None:
            await cancellation_event.wait()

        cancellation_task: asyncio.Task[None] = asyncio.create_task(
            wait_for_cancellation()
        )

        try:
            awaitables: list[asyncio.Future[Any]] = [
                prompt_task,
                timeout_task,
                cancellation_task,
            ]
            done, pending = await asyncio.wait(
                awaitables,
                return_when=asyncio.FIRST_COMPLETED,
            )

            for task in pending:
                task.cancel()

            if prompt_task in done:
                return bool(prompt_task.result())
            elif cancellation_task in done:
                if not prompt_task.done():
                    prompt_task.cancel()  # This won't interrupt the blocking input

                raise asyncio.CancelledError("Request cancelled by another handler")
            else:
                if not prompt_task.done():
                    prompt_task.cancel()  # This won't interrupt the blocking input

                self._console.print("\n[red]Timeout - request denied[/red]")
                return False

        except Exception:
            if not prompt_task.done():
                prompt_task.cancel()
            raise

    def _display_result(self, allowed: bool) -> None:
        """Display the confirmation result."""
        msg = (
            "[green]√ Request allowed[/green]"
            if allowed
            else "[red]× Request denied[/red]"
        )
        self._console.print(f"{msg}\n")
