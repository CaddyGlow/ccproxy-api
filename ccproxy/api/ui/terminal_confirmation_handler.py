"""Terminal UI handler for confirmation requests using Rich and Live display."""

import asyncio
import contextlib
import sys
import time


# Platform-specific imports for single-key input
try:
    import termios
    import tty

    HAS_TERMIOS = True
except ImportError:
    HAS_TERMIOS = False

from rich.align import Align
from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from structlog import get_logger

from ccproxy.api.services.confirmation_service import ConfirmationRequest
from ccproxy.api.ui.confirmation_handler_protocol import ConfirmationHandlerProtocol


logger = get_logger(__name__)


class TerminalConfirmationHandler:
    """Handles confirmation requests in the terminal using a Rich Live display.

    Implements ConfirmationHandlerProtocol for type safety and interoperability.
    """

    def __init__(self) -> None:
        self._console = Console()
        self._cancellation_events: dict[str, asyncio.Event] = {}

    def cancel_confirmation(self, request_id: str, reason: str = "cancelled") -> None:
        """Cancel an ongoing confirmation request."""
        if request_id in self._cancellation_events:
            self._cancellation_events[request_id].set()

    async def handle_confirmation(self, request: ConfirmationRequest) -> bool:
        """Handle a confirmation request by prompting the user with a live display."""
        cancellation_event = asyncio.Event()
        self._cancellation_events[request.id] = cancellation_event

        try:
            return await self._get_user_confirmation_live(request, cancellation_event)
        except asyncio.CancelledError:
            # A cancellation message is displayed within the live handler
            raise
        except Exception as e:
            logger.error(
                "terminal_confirmation_error",
                request_id=request.id,
                error=str(e),
                exc_info=True,
            )
            self._console.print("\n[red]Error handling confirmation request[/red]\n")
            return False  # Deny on error
        finally:
            self._cancellation_events.pop(request.id, None)

    def _build_live_display(
        self,
        request: ConfirmationRequest,
        time_left: float,
        status: str = "Waiting for input...",
        result: bool | None = None,
    ) -> Panel:
        """Builds the rich renderable for the live display using a Group."""
        # Create a more compact table with better column sizing
        table = Table(show_header=False, box=None, padding=0, width=70)
        table.add_column(style="bold cyan", width=12, no_wrap=True)
        table.add_column(style="white", overflow="fold")
        table.add_row("Tool", request.tool_name)
        table.add_row("Request ID", f"{request.id[:8]}...")

        # Format input parameters more compactly
        for key, value in request.input.items():
            value_str = self._format_value(str(value))
            table.add_row(f"  {key}", value_str)

        # Create status components with better alignment
        if time_left > 0:
            timeout_text = f"[yellow]Timeout in {int(time_left)}s[/yellow]"
        else:
            timeout_text = "[red]Timeout![/red]"

        status_text = f"[dim]{status}[/dim]"

        # Create the prompt text based on result
        if result is True:
            prompt_text = Text("√ Allowed", style="bold green")
        elif result is False:
            prompt_text = Text("× Denied", style="bold red")
        else:
            prompt_text = Text("Allow this operation? (y/N): ", style="bold white")

        # Use a more compact group layout
        ui_group = Group(
            table,
            "",  # Empty line for spacing
            Align.center(timeout_text),
            Align.center(status_text),
            "",  # Empty line for spacing
            Align.center(prompt_text),
        )

        # Create panel with better styling and fixed width
        return Panel(
            ui_group,
            title="[bold red]! Permission Request[/bold red]",
            border_style="red" if result is None else "green" if result else "red",
            width=76,  # Fixed width to prevent scrolling issues
            padding=(1, 2),  # Add some internal padding
        )

    async def _get_user_confirmation_live(
        self, request: ConfirmationRequest, cancellation_event: asyncio.Event
    ) -> bool:
        """Get user confirmation using a live display with timeout and cancellation."""
        start_time = time.monotonic()
        timeout = request.time_remaining()

        status = "Waiting for input..."
        result = None

        live_display = self._build_live_display(request, timeout, status)

        with Live(
            live_display,
            console=self._console,
            auto_refresh=False,
            transient=False,  # Keep content visible
            refresh_per_second=10,  # Higher refresh rate for responsiveness
        ) as live:
            while result is None:
                time_elapsed = time.monotonic() - start_time
                time_left = timeout - time_elapsed

                # Check for cancellation
                if cancellation_event.is_set():
                    status = "Cancelled by another handler"
                    live.update(
                        self._build_live_display(request, 0, status), refresh=True
                    )
                    await asyncio.sleep(0.5)
                    raise asyncio.CancelledError("Request cancelled by another handler")

                # Check for timeout
                if time_left <= 0:
                    status = "Timeout - request denied"
                    result = False
                    break

                # Update display
                live.update(
                    self._build_live_display(request, time_left, status), refresh=True
                )

                # Check for user input (single key press)
                key = self._get_single_key()
                if key:
                    key_lower = key.lower()
                    if key_lower in ("y", "n", "\x03", "\x04"):  # y, n, Ctrl+C, Ctrl+D
                        result = key_lower == "y"
                        status = "Request allowed" if result else "Request denied"
                        break

                # Small sleep to prevent busy waiting
                await asyncio.sleep(0.1)

            # Final update to show the result clearly
            final_display = self._build_live_display(request, 0, status, result)
            live.update(final_display, refresh=True)

        # Give user a moment to see the final result
        await asyncio.sleep(1.5)

        # A final newline to separate from subsequent output
        self._console.print()

        return result if result is not None else False

    def _format_value(self, value: str) -> str:
        """Format a value, truncating if too long."""
        if len(value) > 57:  # 60 - 3 for "..."
            return value[:57] + "..."
        return value

    def _display_request(self, request: ConfirmationRequest) -> None:
        """Display a confirmation request using Rich Panel."""
        # Create a table for the request details
        table = Table(show_header=False, box=None, padding=0, width=70)
        table.add_column(style="bold cyan", width=12, no_wrap=True)
        table.add_column(style="white", overflow="fold")
        table.add_row("Tool", request.tool_name)
        table.add_row("Request ID", f"{request.id[:8]}...")

        # Format input parameters
        for key, value in request.input.items():
            formatted_value = self._format_value(str(value))
            table.add_row(f"  {key}", formatted_value)

        # Create panel with the table
        panel = Panel(
            table,
            title="[bold red]! Permission Request[/bold red]",
            border_style="red",
            width=76,
            padding=(1, 2),
        )

        self._console.print(panel)

    def _display_result(self, result: bool) -> None:
        """Display the confirmation result."""
        if result:
            self._console.print("[green]Request allowed[/green]")
        else:
            self._console.print("[red]Request denied[/red]")

    def _get_single_key(self) -> str | None:
        """Get a single key press without waiting for enter. Non-blocking."""
        try:
            if not sys.stdin.isatty():
                return None

            if HAS_TERMIOS:
                # Unix-like systems
                import select

                if not select.select([sys.stdin], [], [], 0)[0]:
                    return None

                old_settings = termios.tcgetattr(sys.stdin.fileno())
                try:
                    tty.setraw(sys.stdin.fileno())
                    char = sys.stdin.read(1)
                    return char
                finally:
                    termios.tcsetattr(
                        sys.stdin.fileno(), termios.TCSADRAIN, old_settings
                    )
            elif sys.platform.startswith("win"):
                # Windows
                import msvcrt

                if msvcrt.kbhit():
                    return msvcrt.getch().decode("utf-8", errors="ignore")
                return None
            else:
                return None
        except Exception:
            return None
