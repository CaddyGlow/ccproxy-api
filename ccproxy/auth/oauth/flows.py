"""OAuth flow engines for CLI authentication."""

import asyncio
import base64
import secrets
import sys
import webbrowser
from typing import Any

import typer
from rich.console import Console

from ccproxy.auth.oauth.cli_errors import AuthProviderError, PortBindError
from ccproxy.auth.oauth.registry import OAuthProviderProtocol
from ccproxy.core.logging import get_logger


logger = get_logger(__name__)
console = Console()


class CLICallbackServer:
    """Temporary HTTP server for handling OAuth callbacks in CLI flows."""

    def __init__(self, port: int, callback_path: str = "/callback") -> None:
        """Initialize the callback server.

        Args:
            port: Port to bind the server to
            callback_path: Path to handle OAuth callbacks
        """
        self.port = port
        self.callback_path = callback_path
        self.server: Any = None
        self.callback_received = False
        self.callback_data: dict[str, Any] = {}
        self.callback_future: asyncio.Future[dict[str, Any]] | None = None

    async def start(self) -> None:
        """Start the callback server."""
        import aiohttp.web

        app = aiohttp.web.Application()
        app.router.add_get(self.callback_path, self._handle_callback)

        # Create server on specified port
        try:
            runner = aiohttp.web.AppRunner(app)
            await runner.setup()

            site = aiohttp.web.TCPSite(runner, "localhost", self.port)
            await site.start()

            self.server = runner
            logger.debug(
                "cli_callback_server_started", port=self.port, path=self.callback_path
            )
        except OSError as e:
            if e.errno == 48:  # Address already in use
                raise PortBindError(
                    f"Port {self.port} is already in use. Please close other applications using this port."
                ) from e
            else:
                raise PortBindError(
                    f"Failed to start callback server on port {self.port}: {e}"
                ) from e

    async def stop(self) -> None:
        """Stop the callback server."""
        if self.server:
            await self.server.cleanup()
            self.server = None
            logger.debug("cli_callback_server_stopped", port=self.port)

    async def _handle_callback(self, request: Any) -> Any:
        """Handle OAuth callback requests."""
        import aiohttp.web

        # Extract callback parameters
        query_params = dict(request.query)

        # Store callback data
        self.callback_data = query_params
        self.callback_received = True

        # Signal that callback was received
        if self.callback_future and not self.callback_future.done():
            self.callback_future.set_result(query_params)

        logger.debug("cli_callback_received", params=list(query_params.keys()))

        # Return success page
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Authentication Complete</title>
            <style>
                body { font-family: Arial, sans-serif; text-align: center; margin-top: 50px; }
                .success { color: #4CAF50; }
                .info { color: #2196F3; }
            </style>
        </head>
        <body>
            <h1 class="success">✓ Authentication Successful</h1>
            <p class="info">You can close this window and return to the command line.</p>
        </body>
        </html>
        """

        return aiohttp.web.Response(text=html_content, content_type="text/html")

    async def wait_for_callback(
        self, expected_state: str | None = None, timeout: int = 300
    ) -> dict[str, Any]:
        """Wait for OAuth callback with optional state validation.

        Args:
            expected_state: Expected OAuth state parameter for validation
            timeout: Timeout in seconds

        Returns:
            Callback data dictionary

        Raises:
            asyncio.TimeoutError: If callback is not received within timeout
            ValueError: If state validation fails
        """
        self.callback_future = asyncio.Future()

        try:
            # Wait for callback with timeout
            callback_data = await asyncio.wait_for(
                self.callback_future, timeout=timeout
            )

            # Validate state if provided
            if expected_state and expected_state != "manual":
                received_state = callback_data.get("state")
                if received_state != expected_state:
                    raise ValueError(
                        f"OAuth state mismatch: expected {expected_state}, got {received_state}"
                    )

            # Check for OAuth errors
            if "error" in callback_data:
                error = callback_data.get("error")
                error_description = callback_data.get(
                    "error_description", "No description provided"
                )
                raise ValueError(f"OAuth error: {error} - {error_description}")

            # Ensure we have an authorization code
            if "code" not in callback_data:
                raise ValueError("No authorization code received in callback")

            return callback_data

        except TimeoutError:
            logger.error("cli_callback_timeout", timeout=timeout, port=self.port)
            raise TimeoutError(f"No OAuth callback received within {timeout} seconds")


def render_qr_code(url: str) -> None:
    """Render QR code for URL when TTY supports it."""
    if not sys.stdout.isatty():
        return

    try:
        import qrcode  # type: ignore[import-untyped]

        qr = qrcode.QRCode(border=1)
        qr.add_data(url)
        qr.print_ascii(invert=True)
        console.print("[dim]Scan QR code with mobile device[/dim]")
    except ImportError:
        # QR code library not available - graceful degradation
        pass


class BrowserFlow:
    """Browser-based OAuth flow with callback server."""

    async def run(self, provider: OAuthProviderProtocol, no_browser: bool) -> Any:
        """Execute browser OAuth flow with fallback handling."""
        cli_config = provider.cli

        # Try provider's fixed port
        try:
            callback_server = CLICallbackServer(
                cli_config.callback_port, cli_config.callback_path
            )
            await callback_server.start()
        except PortBindError as e:
            # Offer manual fallback for fixed-port providers
            if cli_config.fixed_redirect_uri:
                console.print(
                    f"[yellow]Port {cli_config.callback_port} unavailable. Try --manual mode.[/yellow]"
                )
                raise AuthProviderError(
                    f"Required port {cli_config.callback_port} unavailable"
                ) from e
            raise

        try:
            # Generate OAuth parameters with PKCE if supported
            state = secrets.token_urlsafe(32)
            code_verifier = None
            if provider.supports_pkce:
                code_verifier = (
                    base64.urlsafe_b64encode(secrets.token_bytes(32))
                    .decode("utf-8")
                    .rstrip("=")
                )

            # Use fixed redirect URI or construct from config
            redirect_uri = (
                cli_config.fixed_redirect_uri
                or f"http://localhost:{cli_config.callback_port}{cli_config.callback_path}"
            )

            # Get authorization URL
            auth_url = await provider.get_authorization_url(
                state, code_verifier, redirect_uri
            )

            # Always show URL and QR code for fallback
            console.print(f"[bold]Visit: {auth_url}[/bold]")
            render_qr_code(auth_url)

            # Try to open browser unless explicitly disabled
            if not no_browser:
                try:
                    webbrowser.open(auth_url)
                    console.print("[dim]Opening browser...[/dim]")
                except Exception:
                    console.print(
                        "[yellow]Could not open browser automatically[/yellow]"
                    )

            # Wait for callback with timeout and state validation
            try:
                callback_data = await callback_server.wait_for_callback(
                    state, timeout=300
                )
                credentials = await provider.handle_callback(
                    callback_data["code"], state, code_verifier, redirect_uri
                )
                return await provider.save_credentials(credentials)
            except TimeoutError:
                # Fallback to manual code entry if callback times out
                console.print(
                    "[yellow]Callback timed out. You can enter the code manually.[/yellow]"
                )
                if cli_config.supports_manual_code:
                    # Use provider-specific manual redirect URI or fallback to OOB
                    manual_redirect_uri = (
                        cli_config.manual_redirect_uri or "urn:ietf:wg:oauth:2.0:oob"
                    )
                    manual_auth_url = await provider.get_authorization_url(
                        state, code_verifier, manual_redirect_uri
                    )
                    console.print(f"[bold]Manual URL: {manual_auth_url}[/bold]")

                    import typer

                    raw_code = typer.prompt("Enter the authorization code")

                    # Parse the code - some providers (like Claude) return code#state format
                    # Extract the code and state parts
                    code_parts = raw_code.split("#")
                    code = code_parts[0].strip()

                    # If there's a state in the input (Claude format), use it instead of our generated state
                    if len(code_parts) > 1 and code_parts[1].strip():
                        actual_state = code_parts[1].strip()
                    else:
                        actual_state = state

                    credentials = await provider.handle_callback(
                        code, actual_state, code_verifier, manual_redirect_uri
                    )
                    return await provider.save_credentials(credentials)
                else:
                    raise
        finally:
            await callback_server.stop()


class DeviceCodeFlow:
    """OAuth device code flow for headless environments."""

    async def run(self, provider: OAuthProviderProtocol) -> Any:
        """Execute device code flow with polling."""
        (
            device_code,
            user_code,
            verification_uri,
            expires_in,
        ) = await provider.start_device_flow()

        console.print(f"[bold green]Visit: {verification_uri}[/bold green]")
        console.print(f"[bold green]Enter code: {user_code}[/bold green]")
        render_qr_code(verification_uri)  # QR code for mobile

        # Poll for completion with timeout
        with console.status("Waiting for authorization..."):
            credentials = await provider.complete_device_flow(
                device_code, 5, expires_in
            )

        return await provider.save_credentials(credentials)


class ManualCodeFlow:
    """Manual authorization code entry for restricted environments."""

    async def run(self, provider: OAuthProviderProtocol) -> Any:
        """Execute manual code entry flow."""
        # Generate state for manual flow
        state = secrets.token_urlsafe(32)
        code_verifier = None
        if provider.supports_pkce:
            code_verifier = (
                base64.urlsafe_b64encode(secrets.token_bytes(32))
                .decode("utf-8")
                .rstrip("=")
            )

        # Get provider-specific manual redirect URI or fallback to OOB
        manual_redirect_uri = (
            provider.cli.manual_redirect_uri or "urn:ietf:wg:oauth:2.0:oob"
        )

        # Get authorization URL for manual entry
        auth_url = await provider.get_authorization_url(
            state, code_verifier, manual_redirect_uri
        )

        console.print(f"[bold green]Visit: {auth_url}[/bold green]")
        render_qr_code(auth_url)

        # Prompt for manual code entry
        raw_code = typer.prompt("[bold]Enter the authorization code[/bold]").strip()

        # Parse the code - some providers (like Claude) return code#state format
        # Extract the code and state parts
        code_parts = raw_code.split("#")
        code = code_parts[0].strip()

        # If there's a state in the input (Claude format), use it instead of our generated state
        if len(code_parts) > 1 and code_parts[1].strip():
            actual_state = code_parts[1].strip()
        else:
            actual_state = state

        # Use the provider's handle_callback method instead of exchange_manual_code
        # to properly handle state validation
        credentials = await provider.handle_callback(
            code, actual_state, code_verifier, manual_redirect_uri
        )
        return await provider.save_credentials(credentials)
