"""Terminal UI handler for confirmation requests using Textual with request stacking support."""

import asyncio
import contextlib
import time
from dataclasses import dataclass

from textual.app import App, ComposeResult
from textual.containers import Container, Vertical
from textual.events import Key
from textual.reactive import reactive
from textual.screen import ModalScreen
from textual.timer import Timer
from textual.widgets import Label, Static

from ccproxy.core.async_task_manager import (
    create_fire_and_forget_task,
    create_managed_task,
)
from ccproxy.core.logging import get_plugin_logger

from ..models import PermissionRequest


logger = get_plugin_logger()


@dataclass
class PendingRequest:
    """Represents a pending confirmation request with its response future."""

    request: PermissionRequest
    future: asyncio.Future[bool]
    cancelled: bool = False


class ConfirmationScreen(ModalScreen[bool]):
    """Modal screen for displaying a single confirmation request."""

    BINDINGS = [
        ("y", "confirm", "Yes"),
        ("n", "deny", "No"),
        ("enter", "confirm", "Confirm"),
        ("escape", "deny", "Cancel"),
        ("ctrl+c", "cancel", "Cancel"),
    ]

    def __init__(self, request: PermissionRequest) -> None:
        super().__init__()
        self.request = request
        self.start_time = time.time()
        self.countdown_timer: Timer | None = None

    time_remaining = reactive(0.0)

    def compose(self) -> ComposeResult:
        """Compose the confirmation dialog."""
        with Container(id="confirmation-dialog"):
            yield Vertical(
                Label("[bold red]Permission Request[/bold red]", id="title"),
                self._create_info_display(),
                Label("Calculating timeout...", id="countdown", classes="countdown"),
                Label(
                    "[bold white]Allow this operation? (y/N):[/bold white]",
                    id="question",
                ),
                id="content",
            )

    def _create_info_display(self) -> Static:
        """Create the information display widget."""
        info_lines = [
            f"[bold cyan]Tool:[/bold cyan] {self.request.tool_name}",
            f"[bold cyan]Request ID:[/bold cyan] {self.request.id[:8]}...",
        ]

        # Add input parameters
        for key, value in self.request.input.items():
            display_value = value if len(value) <= 50 else f"{value[:47]}..."
            info_lines.append(f"[bold cyan]{key}:[/bold cyan] {display_value}")

        return Static("\n".join(info_lines), id="info")

    def on_mount(self) -> None:
        """Start the countdown timer when mounted."""
        self.update_countdown()
        self.countdown_timer = self.set_interval(0.1, self.update_countdown)

    def update_countdown(self) -> None:
        """Update the countdown display."""
        elapsed = time.time() - self.start_time
        remaining = max(0, self.request.time_remaining() - elapsed)
        self.time_remaining = remaining

        if remaining <= 0:
            self._timeout()
        else:
            countdown_widget = self.query_one("#countdown", Label)
            if remaining > 10:
                style = "yellow"
            elif remaining > 5:
                style = "orange1"
            else:
                style = "red"
            countdown_widget.update(f"[{style}]Timeout in {remaining:.1f}s[/{style}]")

    def _timeout(self) -> None:
        """Handle timeout."""
        if self.countdown_timer:
            self.countdown_timer.stop()
            self.countdown_timer = None
        # Schedule the async result display
        self.call_later(self._show_result, False, "TIMEOUT - DENIED")

    async def _show_result(self, allowed: bool, message: str) -> None:
        """Show the result with visual feedback before dismissing.

        Args:
            allowed: Whether the request was allowed
            message: Message to display
        """
        # Update the question to show the result
        question_widget = self.query_one("#question", Label)
        if allowed:
            question_widget.update(f"[bold green]✓ {message}[/bold green]")
        else:
            question_widget.update(f"[bold red]✗ {message}[/bold red]")

        # Update the dialog border color
        dialog = self.query_one("#confirmation-dialog", Container)
        if allowed:
            dialog.styles.border = ("solid", "green")
        else:
            dialog.styles.border = ("solid", "red")

        # Give user time to see the result
        await asyncio.sleep(1.5)
        self.dismiss(allowed)

    def action_confirm(self) -> None:
        """Confirm the request."""
        if self.countdown_timer:
            self.countdown_timer.stop()
            self.countdown_timer = None
        self.call_later(self._show_result, True, "ALLOWED")

    def action_deny(self) -> None:
        """Deny the request."""
        if self.countdown_timer:
            self.countdown_timer.stop()
            self.countdown_timer = None
        self.call_later(self._show_result, False, "DENIED")

    def action_cancel(self) -> None:
        """Cancel the request (Ctrl+C)."""
        if self.countdown_timer:
            self.countdown_timer.stop()
            self.countdown_timer = None
        self.call_later(self._show_result, False, "CANCELLED")
        # Raise KeyboardInterrupt to forward it up
        raise KeyboardInterrupt("User cancelled confirmation")


class ConfirmationApp(App[bool]):
    """Simple Textual app for a single confirmation request."""

    CSS = """

    Screen {
        border: none;
    }

    Static {
        background: $surface;
    }

    #confirmation-dialog {
        width: 60;
        height: 18;
        border: round solid $accent;
        background: $surface;
        padding: 1;
        box-sizing: border-box;
    }

    #title {
        text-align: center;
        margin-bottom: 1;
    }

    #info {
        border: solid $primary;
        margin: 1;
        padding: 1;
        background: $surface;
        height: auto;
    }

    #countdown {
        text-align: center;
        margin: 1;
        background: $surface;
        text-style: bold;
        height: 1;
    }

    #question {
        text-align: center;
        margin: 1;
        background: $surface;
    }


    .countdown {
        text-style: bold;
    }
    """

    BINDINGS = [
        ("y", "confirm", "Yes"),
        ("n", "deny", "No"),
        ("enter", "confirm", "Confirm"),
        ("escape", "deny", "Cancel"),
        ("ctrl+c", "cancel", "Cancel"),
    ]

    def __init__(self, request: PermissionRequest) -> None:
        super().__init__()
        self.theme = "textual-ansi"
        self.request = request
        self.result = False
        self.start_time = time.time()
        self.countdown_timer: Timer | None = None

    time_remaining = reactive(0.0)

    def compose(self) -> ComposeResult:
        """Compose the confirmation dialog directly."""
        with Container(id="confirmation-dialog"):
            yield Vertical(
                Label("[bold red]Permission Request[/bold red]", id="title"),
                self._create_info_display(),
                Label("Calculating timeout...", id="countdown", classes="countdown"),
                Label(
                    "[bold white]Allow this operation? (y/N):[/bold white]",
                    id="question",
                ),
                id="content",
            )

    def _create_info_display(self) -> Static:
        """Create the information display widget."""
        info_lines = [
            f"[bold cyan]Tool:[/bold cyan] {self.request.tool_name}",
            f"[bold cyan]Request ID:[/bold cyan] {self.request.id[:8]}...",
        ]

        # Add input parameters
        for key, value in self.request.input.items():
            display_value = value if len(value) <= 50 else f"{value[:47]}..."
            info_lines.append(f"[bold cyan]{key}:[/bold cyan] {display_value}")

        return Static("\n".join(info_lines), id="info")

    def on_mount(self) -> None:
        """Start the countdown timer when mounted."""
        self.update_countdown()
        self.countdown_timer = self.set_interval(0.1, self.update_countdown)

    def update_countdown(self) -> None:
        """Update the countdown display."""
        elapsed = time.time() - self.start_time
        remaining = max(0, self.request.time_remaining() - elapsed)
        self.time_remaining = remaining

        if remaining <= 0:
            self._timeout()
        else:
            countdown_widget = self.query_one("#countdown", Label)
            if remaining > 10:
                style = "yellow"
            elif remaining > 5:
                style = "orange1"
            else:
                style = "red"
            countdown_widget.update(f"[{style}]Timeout in {remaining:.1f}s[/{style}]")

    def _timeout(self) -> None:
        """Handle timeout."""
        if self.countdown_timer:
            self.countdown_timer.stop()
            self.countdown_timer = None
        # Schedule the async result display
        self.call_later(self._show_result, False, "TIMEOUT - DENIED")

    async def _show_result(self, allowed: bool, message: str) -> None:
        """Show the result with visual feedback before exiting.

        Args:
            allowed: Whether the request was allowed
            message: Message to display
        """
        # Update the question to show the result
        question_widget = self.query_one("#question", Label)
        if allowed:
            question_widget.update(f"[bold green]✓ {message}[/bold green]")
        else:
            question_widget.update(f"[bold red]✗ {message}[/bold red]")

        # Update the dialog border color
        dialog = self.query_one("#confirmation-dialog", Container)
        if allowed:
            dialog.styles.border = ("solid", "green")
        else:
            dialog.styles.border = ("solid", "red")

        # Give user time to see the result
        await asyncio.sleep(1.5)
        self.exit(allowed)

    def action_confirm(self) -> None:
        """Confirm the request."""
        if self.countdown_timer:
            self.countdown_timer.stop()
            self.countdown_timer = None
        self.call_later(self._show_result, True, "ALLOWED")

    def action_deny(self) -> None:
        """Deny the request."""
        if self.countdown_timer:
            self.countdown_timer.stop()
            self.countdown_timer = None
        self.call_later(self._show_result, False, "DENIED")

    def action_cancel(self) -> None:
        """Cancel the request (Ctrl+C)."""
        if self.countdown_timer:
            self.countdown_timer.stop()
            self.countdown_timer = None
        self.call_later(self._show_result, False, "CANCELLED")
        # Raise KeyboardInterrupt to forward it up
        raise KeyboardInterrupt("User cancelled confirmation")

    async def on_key(self, event: Key) -> None:
        """Handle global key events, especially Ctrl+C."""
        if event.key == "ctrl+c":
            # Forward the KeyboardInterrupt
            self.exit(False)
            raise KeyboardInterrupt("User cancelled confirmation")


class TerminalPermissionHandler:
    """Handles confirmation requests in the terminal using Textual with request stacking.

    Implements ConfirmationHandlerProtocol for type safety and interoperability.
    """

    def __init__(self) -> None:
        """Initialize the terminal confirmation handler."""
        self._request_queue: (
            asyncio.Queue[tuple[PermissionRequest, asyncio.Future[bool]]] | None
        ) = None
        self._cancelled_requests: set[str] = set()
        self._processing_task: asyncio.Task[None] | None = None
        self._active_apps: dict[str, ConfirmationApp] = {}

    def _get_request_queue(
        self,
    ) -> asyncio.Queue[tuple[PermissionRequest, asyncio.Future[bool]]]:
        """Lazily initialize and return the request queue."""
        if self._request_queue is None:
            self._request_queue = asyncio.Queue()
        return self._request_queue

    def _safe_set_future_result(
        self, future: asyncio.Future[bool], result: bool
    ) -> bool:
        """Safely set a future result, handling already cancelled futures.

        Args:
            future: The future to set the result on
            result: The result to set

        Returns:
            bool: True if result was set successfully, False if future was cancelled
        """
        if future.cancelled():
            return False
        try:
            future.set_result(result)
            return True
        except asyncio.InvalidStateError:
            # Future was already resolved or cancelled
            return False

    def _safe_set_future_exception(
        self, future: asyncio.Future[bool], exception: BaseException
    ) -> bool:
        """Safely set a future exception, handling already cancelled futures.

        Args:
            future: The future to set the exception on
            exception: The exception to set

        Returns:
            bool: True if exception was set successfully, False if future was cancelled
        """
        if future.cancelled():
            return False
        try:
            future.set_exception(exception)
            return True
        except asyncio.InvalidStateError:
            # Future was already resolved or cancelled
            return False

    async def _process_queue(self) -> None:
        """Process requests from the queue one by one."""
        while True:
            try:
                request, future = await self._get_request_queue().get()

                # Check if request is valid for processing
                if not self._is_request_processable(request, future):
                    continue

                # Process the request
                await self._process_single_request(request, future)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("queue_processing_error", error=str(e), exc_info=e)

    def _is_request_processable(
        self, request: PermissionRequest, future: asyncio.Future[bool]
    ) -> bool:
        """Check if a request can be processed."""
        # Check if cancelled before processing
        if request.id in self._cancelled_requests:
            self._safe_set_future_result(future, False)
            self._cancelled_requests.discard(request.id)
            return False

        # Check if expired
        if request.time_remaining() <= 0:
            self._safe_set_future_result(future, False)
            return False

        return True

    async def _process_single_request(
        self, request: PermissionRequest, future: asyncio.Future[bool]
    ) -> None:
        """Process a single permission request."""
        app = None
        try:
            # Create and run a simple app for this request
            app = ConfirmationApp(request)
            self._active_apps[request.id] = app

            app_result = await app.run_async(inline=True, inline_no_clear=True)
            result = bool(app_result) if app_result is not None else False

            # Apply cancellation if it occurred during processing
            if request.id in self._cancelled_requests:
                result = False
                self._cancelled_requests.discard(request.id)

            self._safe_set_future_result(future, result)

        except KeyboardInterrupt:
            self._safe_set_future_exception(
                future, KeyboardInterrupt("User cancelled confirmation")
            )
        except Exception as e:
            logger.error(
                "confirmation_app_error",
                request_id=request.id,
                error=str(e),
                exc_info=e,
            )
            self._safe_set_future_result(future, False)
        finally:
            # Always cleanup app reference
            if app:
                self._active_apps.pop(request.id, None)

    def _ensure_processing_task_running(self) -> None:
        """Ensure the processing task is running."""
        if self._processing_task is None or self._processing_task.done():
            # Use fire-and-forget since this is called from sync context
            create_fire_and_forget_task(
                self._create_processing_task(),
                name="terminal_handler_processing",
                creator="TerminalHandler",
            )

    async def _create_processing_task(self) -> None:
        """Create the processing task in async context."""
        self._processing_task = await create_managed_task(
            self._process_queue(),
            name="terminal_handler_queue_processor",
            creator="TerminalHandler",
        )

    async def _queue_and_wait_for_result(self, request: PermissionRequest) -> bool:
        """Queue a request and wait for its result."""
        future: asyncio.Future[bool] = asyncio.Future()
        await self._get_request_queue().put((request, future))
        return await future

    async def handle_permission(self, request: PermissionRequest) -> bool:
        """Handle a permission request.

        Args:
            request: The permission request to handle

        Returns:
            bool: True if the user confirmed, False otherwise
        """
        try:
            logger.info(
                "handling_confirmation_request",
                request_id=request.id,
                tool_name=request.tool_name,
                time_remaining=request.time_remaining(),
            )

            # Check if request has already expired
            if request.time_remaining() <= 0:
                logger.info("confirmation_request_expired", request_id=request.id)
                return False

            # Ensure processing task is running
            self._ensure_processing_task_running()

            # Queue request and wait for result
            result = await self._queue_and_wait_for_result(request)

            logger.info(
                "confirmation_request_completed", request_id=request.id, result=result
            )

            return result

        except Exception as e:
            logger.error(
                "confirmation_handling_error",
                request_id=request.id,
                error=str(e),
                exc_info=e,
            )
            return False

    def cancel_confirmation(self, request_id: str, reason: str = "cancelled") -> None:
        """Cancel an ongoing confirmation request.

        Args:
            request_id: The ID of the request to cancel
            reason: The reason for cancellation
        """
        logger.info("cancelling_confirmation", request_id=request_id, reason=reason)
        self._cancelled_requests.add(request_id)

        # If there's an active dialog for this request, close it immediately
        if request_id in self._active_apps:
            app = self._active_apps[request_id]
            # Schedule the cancellation feedback asynchronously
            create_fire_and_forget_task(
                self._cancel_active_dialog(app, reason),
                name="terminal_dialog_cancel",
                creator="TerminalHandler",
            )

    async def _cancel_active_dialog(self, app: ConfirmationApp, reason: str) -> None:
        """Cancel an active dialog with visual feedback.

        Args:
            app: The active ConfirmationApp to cancel
            reason: The reason for cancellation
        """
        try:
            # Determine the message and result based on reason
            if "approved by another handler" in reason.lower():
                message = "APPROVED BY ANOTHER HANDLER"
                allowed = True
            elif "denied by another handler" in reason.lower():
                message = "DENIED BY ANOTHER HANDLER"
                allowed = False
            else:
                message = f"CANCELLED - {reason.upper()}"
                allowed = False

            # Show visual feedback through the app's _show_result method
            await app._show_result(allowed, message)

        except Exception as e:
            logger.error(
                "cancel_dialog_error",
                error=str(e),
                exc_info=e,
            )
            # Fallback: just exit the app without feedback
            with contextlib.suppress(Exception):
                app.exit(False)

    async def shutdown(self) -> None:
        """Shutdown the handler and cleanup resources."""
        if self._processing_task and not self._processing_task.done():
            self._processing_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._processing_task

        self._processing_task = None
