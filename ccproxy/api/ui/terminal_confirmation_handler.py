"""Terminal UI handler for confirmation requests using Rich."""

import asyncio
import contextlib
import sys


# Platform-specific imports for terminal handling
try:
    import termios
    import tty

    HAS_TERMIOS = True
except ImportError:
    # Windows doesn't have termios/tty
    HAS_TERMIOS = False

from rich.console import Console
from rich.panel import Panel
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

    def _format_value(self, value: str) -> str:
        """Format a value for display."""
        return value if len(value) <= 60 else value[:57] + "..."

    async def _get_user_confirmation_with_cancellation(
        self, request: ConfirmationRequest, cancellation_event: asyncio.Event
    ) -> bool:
        """Get user confirmation with timeout and cancellation handling."""
        self._console.print(
            "\n[dim]Note: This prompt will auto-cancel if another handler responds[/dim]"
        )
        self._console.print("\n[bold]Allow this operation? (y/N):[/bold] ", end="")

        # Create async tasks for different completion conditions
        input_task = asyncio.create_task(self._get_async_input())
        timeout_task = asyncio.create_task(asyncio.sleep(request.time_remaining()))
        cancellation_task = asyncio.create_task(cancellation_event.wait())

        try:
            # Wait for the first to complete
            done, pending = await asyncio.wait(
                [input_task, timeout_task, cancellation_task],
                return_when=asyncio.FIRST_COMPLETED,
            )

            # Cancel all pending tasks
            for task in pending:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

            if input_task in done:
                user_input = input_task.result()
                result = user_input.lower().startswith("y")
                self._console.print()  # Add newline after input
                return result
            elif cancellation_task in done:
                # Request was cancelled by another handler
                self._console.print("[yellow]cancelled by another handler[/yellow]")
                raise asyncio.CancelledError("Request cancelled by another handler")
            else:
                # Timeout occurred
                self._console.print("\n[red]Timeout - request denied[/red]")
                return False

        except Exception:
            # Clean up on any exception
            for task in [input_task, timeout_task, cancellation_task]:
                if not task.done():
                    task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await task
            raise

    async def _get_async_input(self) -> str:
        """Get user input asynchronously without blocking the event loop."""
        loop = asyncio.get_event_loop()

        def get_input() -> str:
            try:
                # Use platform-appropriate single character input
                if HAS_TERMIOS and sys.stdin.isatty():
                    # Unix-like systems with termios support
                    old_settings = termios.tcgetattr(sys.stdin.fileno())
                    tty.setraw(sys.stdin.fileno())

                    try:
                        # Read a single character
                        char = sys.stdin.read(1)
                        # Print the character so user sees what they typed
                        sys.stdout.write(char)
                        sys.stdout.flush()
                        return char
                    finally:
                        # Restore terminal settings
                        termios.tcsetattr(
                            sys.stdin.fileno(), termios.TCSADRAIN, old_settings
                        )
                elif sys.platform.startswith("win"):
                    # Windows-specific single character input
                    import msvcrt

                    char = msvcrt.getch().decode("utf-8", errors="ignore")
                    # Print the character so user sees what they typed
                    sys.stdout.write(char)
                    sys.stdout.flush()
                    return char
                else:
                    # Fallback for other environments or non-TTY
                    return input().strip()
            except (KeyboardInterrupt, EOFError):
                return "n"
            except Exception:
                return "n"

        return await loop.run_in_executor(None, get_input)

    def _display_result(self, allowed: bool) -> None:
        """Display the confirmation result."""
        msg = (
            "[green]√ Request allowed[/green]"
            if allowed
            else "[red]× Request denied[/red]"
        )
        self._console.print(f"{msg}\n")
