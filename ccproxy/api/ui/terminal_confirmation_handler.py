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
        """Initialize the terminal handler."""
        self._console = Console()

    async def handle_confirmation(self, request: ConfirmationRequest) -> bool:
        """Handle a confirmation request by prompting the user.

        Args:
            request: The confirmation request to handle

        Returns:
            True if approved, False if denied
        """
        logger.debug(
            "terminal_handler_started",
            request_id=request.id,
            tool_name=request.tool_name,
        )

        try:
            # Display the request
            self._display_request(request)

            # Get user confirmation with timeout handling
            result = await self._get_user_confirmation(request)

            # Display result
            self._display_result(result)

            logger.debug(
                "terminal_handler_completed",
                request_id=request.id,
                result=result,
            )

            return result

        except Exception as e:
            logger.error(
                "terminal_confirmation_error",
                request_id=request.id,
                error=str(e),
                exc_info=True,
            )
            self._console.print("[red]Error handling confirmation request[/red]\n")
            return False  # Deny on error

    def _display_request(self, request: ConfirmationRequest) -> None:
        """Display the confirmation request to the user."""
        # Create a formatted table
        table = Table(show_header=False, box=None)
        table.add_column(style="bold cyan", width=20)
        table.add_column()

        table.add_row("Tool", request.tool_name)
        table.add_row("Request ID", request.id[:8] + "...")

        # Format tool input
        for key, value in request.input.items():
            value_str = self._format_value(value)
            table.add_row(f"  {key}", value_str)

        # Create panel with warning
        panel = Panel(
            table,
            title="[bold red]⚠️  Permission Request[/bold red]",
            border_style="red",
            expand=False,
        )

        self._console.print("\n")
        self._console.print(panel)

        # Show timeout info
        time_remaining = request.time_remaining()
        self._console.print(
            f"\n[yellow]Timeout in {time_remaining} seconds "
            f"(auto-deny if no response)[/yellow]"
        )

    def _format_value(self, value: Any) -> str:
        """Format a value for display."""
        value_str = str(value)
        if len(value_str) > 60:
            value_str = value_str[:57] + "..."
        return value_str

    async def _get_user_confirmation(self, request: ConfirmationRequest) -> bool:
        """Get user confirmation with timeout handling."""
        loop = asyncio.get_event_loop()

        def prompt_user() -> bool:
            try:
                return Confirm.ask(
                    "\n[bold]Allow this operation?[/bold]",
                    default=False,
                )
            except (KeyboardInterrupt, EOFError):
                return False

        # Run the blocking prompt in a thread executor
        try:
            result = await asyncio.wait_for(
                loop.run_in_executor(None, prompt_user),
                timeout=request.time_remaining(),
            )
            return bool(result)
        except TimeoutError:
            self._console.print("[red]Timeout - request denied[/red]")
            return False

    def _display_result(self, allowed: bool) -> None:
        """Display the confirmation result."""
        if allowed:
            self._console.print("[green]✓ Request allowed[/green]\n")
        else:
            self._console.print("[red]✗ Request denied[/red]\n")
