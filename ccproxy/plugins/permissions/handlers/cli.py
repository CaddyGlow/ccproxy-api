"""CLI command for handling confirmation requests via SSE stream."""

import asyncio
import contextlib
import json
import logging
from collections.abc import AsyncIterator
from typing import Any

import httpx
import typer
from pydantic import ValidationError

import ccproxy.core.logging
from ccproxy.config.settings import Settings
from ccproxy.core.async_task_manager import create_managed_task
from ccproxy.core.logging import get_plugin_logger

from ..models import PermissionRequest
from .protocol import ConfirmationHandlerProtocol
from .terminal import TerminalPermissionHandler as TextualPermissionHandler


logger = get_plugin_logger()

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
        terminal_handler: ConfirmationHandlerProtocol,
        ui: bool = True,
        auth_token: str | None = None,
        auto_reconnect: bool = True,
    ):
        self.api_url = api_url.rstrip("/")
        self.terminal_handler = terminal_handler
        self.client: httpx.AsyncClient | None = None
        self.max_retries = 5
        self.base_delay = 1.0
        self.max_delay = 60.0
        self.ui = ui
        self.auth_token = auth_token
        self.auto_reconnect = auto_reconnect

        self._ongoing_requests: dict[str, asyncio.Task[bool]] = {}
        self._resolved_requests: dict[str, tuple[bool, str]] = {}
        self._resolved_by_us: set[str] = set()

    async def __aenter__(self) -> "SSEConfirmationHandler":
        """Async context manager entry."""
        headers = {}
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"

        self.client = httpx.AsyncClient(timeout=300.0, headers=headers)  # 5 minutes
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Async context manager exit."""
        if self.client:
            await self.client.aclose()
            self.client = None

    async def handle_event(self, event_type: str, data: dict[str, Any]) -> None:
        """Handle an SSE event by dispatching to specific handlers.

        Args:
            event_type: Type of the event
            data: Event data
        """
        if event_type == "ping":
            return

        from ..models import EventType

        handler_map = {
            EventType.PERMISSION_REQUEST.value: self._handle_permission_request,
            EventType.PERMISSION_RESOLVED.value: self._handle_permission_resolved,
        }

        handler = handler_map.get(event_type)
        if handler:
            await handler(data)
        else:
            logger.warning(
                "unhandled_sse_event", event_type=event_type, category="streaming"
            )

    async def _handle_permission_request(self, data: dict[str, Any]) -> None:
        """Handle a confirmation request event.

        Args:
            data: Event data containing request details
        """
        request_id = data.get("request_id")
        if not request_id:
            logger.warning("permission_request_missing_id", data=data)
            return

        # Check if this request was already resolved by another handler
        if request_id in self._resolved_requests:
            allowed, reason = self._resolved_requests[request_id]
            logger.info(
                "permission_already_resolved_by_other_handler",
                request_id=request_id,
                allowed=allowed,
                reason=reason,
            )
            logger.info(
                "permission_already_handled",
                request_id=request_id[:8],
                reason=reason,
            )
            return

        logger.info(
            "permission_request_received",
            request_id=request_id,
            tool_name=data.get("tool_name"),
        )

        try:
            # Map request_id to id field for PermissionRequest model
            request_data = dict(data)
            if "request_id" in request_data:
                request_data["id"] = request_data.pop("request_id")
            request = PermissionRequest.model_validate(request_data)
        except ValidationError as e:
            logger.error(
                "permission_request_validation_failed",
                data=data,
                error=str(e),
                exc_info=e,
            )
            return
        except Exception as e:
            logger.error(
                "permission_request_parsing_error", data=data, error=str(e), exc_info=e
            )
            return

        if self.ui and request_id is not None:
            task = await create_managed_task(
                self._handle_permission_with_cancellation(request),
                name=f"cli_permission_handler_{request_id}",
                creator="CLIConfirmationHandler",
            )
            self._ongoing_requests[request_id] = task

    async def _handle_permission_resolved(self, data: dict[str, Any]) -> None:
        """Handle a confirmation resolved event.

        Args:
            data: Event data containing resolution details
        """
        request_id = data.get("request_id")
        allowed = data.get("allowed", False)

        if request_id is not None and allowed is not None:
            reason = (
                "approved by another handler"
                if allowed
                else "denied by another handler"
            )
            self._resolved_requests[request_id] = (allowed, reason)

        was_resolved_by_us = (
            request_id is not None and request_id in self._resolved_by_us
        )

        if request_id is not None and request_id in self._ongoing_requests:
            task = self._ongoing_requests[request_id]
            if not task.done() and not was_resolved_by_us:
                logger.info(
                    "cancelling_ongoing_confirmation",
                    request_id=request_id,
                    allowed=allowed,
                )

                status_text = "approved" if allowed else "denied"
                self.terminal_handler.cancel_confirmation(
                    request_id, f"{status_text} by another handler"
                )

                task.cancel()

                with contextlib.suppress(TimeoutError, asyncio.CancelledError):
                    await asyncio.wait_for(task, timeout=0.1)

                logger.info(
                    "permission_cancelled_by_other_handler",
                    request_id=request_id[:8],
                    status=status_text,
                )

            if request_id is not None:
                self._ongoing_requests.pop(request_id, None)

        if request_id is not None:
            self._resolved_by_us.discard(request_id)

    async def _handle_permission_with_cancellation(
        self, request: PermissionRequest
    ) -> bool:
        """Handle permission with cancellation support.

        Args:
            request: The permission request to handle
        """
        try:
            allowed = await self.terminal_handler.handle_permission(request)

            if request.id in self._resolved_requests:
                logger.info(
                    "permission_resolved_while_processing",
                    request_id=request.id,
                    our_result=allowed,
                )
                return False

            self._resolved_by_us.add(request.id)

            await self.send_response(request.id, allowed)

            await asyncio.sleep(0.5)

            return allowed

        except asyncio.CancelledError:
            logger.info(
                "permission_cancelled",
                request_id=request.id,
            )
            raise

        except Exception as e:
            logger.error(
                "permission_handling_error",
                request_id=request.id,
                error=str(e),
                exc_info=e,
            )
            # Only send response if not already resolved
            if request.id not in self._resolved_requests:
                # If response fails, it might already be resolved
                with contextlib.suppress(Exception):
                    await self.send_response(request.id, False)
            return False

    async def send_response(self, request_id: str, allowed: bool) -> None:
        """Send a confirmation response to the API.

        Args:
            request_id: ID of the confirmation request
            allowed: Whether to allow or deny
        """
        if not self.client:
            logger.error("send_response_no_client", request_id=request_id)
            return

        try:
            response = await self.client.post(
                f"{self.api_url}/permissions/{request_id}/respond",
                json={"allowed": allowed},
            )

            if response.status_code == 200:
                logger.info(
                    "permission_response_sent",
                    request_id=request_id,
                    allowed=allowed,
                )
            elif response.status_code == 409:
                # Already resolved by another handler
                logger.info(
                    "permission_already_resolved",
                    request_id=request_id,
                    status_code=response.status_code,
                )
            else:
                logger.error(
                    "permission_response_failed",
                    request_id=request_id,
                    status_code=response.status_code,
                    response=response.text,
                )

        except httpx.RequestError as e:
            logger.error(
                "permission_response_network_error",
                request_id=request_id,
                error=str(e),
                exc_info=e,
            )
        except Exception as e:
            logger.error(
                "permission_response_error",
                request_id=request_id,
                error=str(e),
                exc_info=e,
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

            buffer = buffer.replace("\r\n", "\n")

            while "\n\n" in buffer:
                event_text, buffer = buffer.split("\n\n", 1)

                if not event_text.strip():
                    continue

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
        if not self.client:
            logger.error("run_no_client")
            return

        stream_url = f"{self.api_url}/permissions/stream"
        retry_count = 0

        logger.info(
            "connecting_to_permission_stream",
            url=stream_url,
        )
        print(f"Connecting to confirmation stream at {stream_url}...")

        while retry_count <= self.max_retries:
            try:
                await self._connect_and_handle_stream(stream_url)
                # If we get here, connection ended gracefully
                if self.auto_reconnect:
                    # Reset retry count and reconnect
                    retry_count = 0
                    print("Connection closed. Reconnecting...")
                    await asyncio.sleep(1.0)  # Brief pause before reconnecting
                    continue
                else:
                    print("Connection closed. Exiting (auto-reconnect disabled).")
                    break

            except KeyboardInterrupt:
                logger.info("permission_handler_shutdown_requested")
                break

            except (
                httpx.ConnectError,
                httpx.TimeoutException,
                httpx.ReadTimeout,
            ) as e:
                retry_count += 1
                if retry_count > self.max_retries:
                    logger.error(
                        "connection_failed_max_retries",
                        max_retries=self.max_retries,
                    )
                    raise typer.Exit(1) from None

                # Exponential backoff with jitter
                delay = min(self.base_delay * (2 ** (retry_count - 1)), self.max_delay)

                logger.warning(
                    "connection_failed_retrying",
                    attempt=retry_count,
                    max_retries=self.max_retries,
                    retry_delay=delay,
                    error=str(e),
                )

                print(
                    f"Connection failed (attempt {retry_count}/{self.max_retries}). Retrying in {delay}s..."
                )

                await asyncio.sleep(delay)
                continue

            except Exception as e:
                logger.error(
                    "sse_client_error", error=str(e), exc_info=e, category="streaming"
                )
                raise typer.Exit(1) from e

    async def _connect_and_handle_stream(self, stream_url: str) -> None:
        """Connect to the stream and handle events."""
        if not self.client:
            logger.error("connect_no_client")
            return

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
                    logger.error(
                        "sse_connection_client_error",
                        status_code=response.status_code,
                        response=error_text,
                    )
                    raise typer.Exit(1)

            logger.info(
                "sse_connection_established",
                url=stream_url,
                message="Connected to confirmation stream. Waiting for requests...",
            )
            print("✓ Connected to confirmation stream. Waiting for requests...")

            async for event_type, data in self.parse_sse_stream(response):
                try:
                    await self.handle_event(event_type, data)
                except Exception as e:
                    logger.error(
                        "sse_event_error",
                        event_type=event_type,
                        error=str(e),
                        exc_info=e,
                    )


@app.command()
def connect(
    api_url: str | None = typer.Option(
        None,
        "--api-url",
        "-u",
        help="API server URL (defaults to settings)",
    ),
    no_ui: bool = typer.Option(False, "--no-ui", help="Disable UI mode"),
    verbose: int = typer.Option(
        0,
        "-v",
        "--verbose",
        count=True,
        help="Increase verbosity (-v for INFO, -vv for DEBUG)",
    ),
    auth_token: str | None = typer.Option(
        None,
        "--auth-token",
        "-t",
        help="Bearer token for API authentication (overrides config)",
        envvar="CCPROXY_AUTH_TOKEN",
    ),
    no_reconnect: bool = typer.Option(
        False,
        "--no-reconnect",
        help="Disable automatic reconnection when connection is lost",
    ),
) -> None:
    """Connect to the API server and handle confirmation requests.

    This command connects to the CCProxy API server via Server-Sent Events
    and handles permission confirmation requests in the terminal.

    """
    # Configure logging level based on verbosity
    # Handle case where verbose might be OptionInfo (in tests) or int (runtime)
    verbose_count = verbose if isinstance(verbose, int) else 0

    if verbose_count >= 2:
        log_level = logging.DEBUG
    elif verbose_count >= 1:
        log_level = logging.INFO
    else:
        log_level = logging.WARNING

    # Configure root logger level
    logging.basicConfig(level=log_level)

    # Configure ccproxy.core.logging to respect the same log level
    ccproxy.core.logging.configure_structlog(log_level)

    settings = Settings.from_config()

    # Use provided URL or default from settings
    if not api_url:
        api_url = f"http://{settings.server.host}:{settings.server.port}"

    # Determine auth token: CLI arg > config setting > None
    token = auth_token or (
        settings.security.auth_token.get_secret_value()
        if settings.security.auth_token
        else None
    )

    # Create handlers based on UI mode selection
    terminal_handler: ConfirmationHandlerProtocol = TextualPermissionHandler()

    async def run_handler() -> None:
        """Run the handler with proper resource management."""
        async with SSEConfirmationHandler(
            api_url,
            terminal_handler,
            not no_ui,
            auth_token=token,
            auto_reconnect=not no_reconnect,
        ) as sse_handler:
            await sse_handler.run()

    # Run the async handler
    try:
        asyncio.run(run_handler())
    except KeyboardInterrupt:
        logger.info("permission_handler_stopped")
    except Exception as e:
        logger.error("permission_handler_error", error=str(e), exc_info=e)
        raise typer.Exit(1) from e


if __name__ == "__main__":
    app()
