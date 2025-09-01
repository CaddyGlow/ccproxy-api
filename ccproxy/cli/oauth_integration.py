"""OAuth integration for CLI commands."""

import asyncio
import contextlib
import webbrowser
from typing import Any

from rich.console import Console

from ccproxy.auth.oauth.registry import OAuthRegistry
from ccproxy.auth.oauth.session import OAuthSessionManager
from ccproxy.core.logging import get_logger


logger = get_logger(__name__)
console = Console()


class CLIOAuthHandler:
    """Handles OAuth flows for CLI commands."""

    def __init__(self, port: int = 9999, registry: OAuthRegistry | None = None):
        """Initialize OAuth handler.

        Args:
            port: Local port for OAuth callback server
        """
        self.port = port
        self.redirect_uri = f"http://localhost:{port}/callback"
        self.session_manager = OAuthSessionManager()
        self.registry = registry or OAuthRegistry()

    async def list_providers(self) -> dict[str, Any]:
        """List all available OAuth providers.

        Returns:
            Dictionary of provider info
        """
        return self.registry.list_providers()

    async def login(
        self,
        provider_name: str,
        open_browser: bool = True,
        timeout: int = 300,
    ) -> Any:
        """Perform OAuth login for a provider.

        Args:
            provider_name: Name of the OAuth provider
            open_browser: Whether to automatically open browser
            timeout: Timeout in seconds for OAuth flow

        Returns:
            Authentication credentials

        Raises:
            ValueError: If provider not found
            TimeoutError: If OAuth flow times out
        """
        # Get provider from registry
        provider = self.registry.get_provider(provider_name)

        if not provider:
            available = list(self.registry.list_providers().keys())
            raise ValueError(
                f"OAuth provider '{provider_name}' not found. "
                f"Available providers: {', '.join(available)}"
            )

        # Update port and redirect_uri from provider config if available
        provider_config = provider.get_config()
        # Prefer explicit resolver on config if present
        if hasattr(provider_config, "get_redirect_uri"):
            import urllib.parse

            ru = provider_config.get_redirect_uri()
            self.redirect_uri = ru
            parsed = urllib.parse.urlparse(ru)
            if parsed.hostname in ["localhost", "127.0.0.1"] and parsed.port:
                self.port = parsed.port
        else:
            # Fallback to callback_port + default path
            if hasattr(provider_config, "callback_port"):
                self.port = provider_config.callback_port
                self.redirect_uri = f"http://localhost:{self.port}/callback"
            # If provider has a specific redirect_uri configured, use it for validation
            # But we still need to use our local callback server
            if (
                hasattr(provider_config, "redirect_uri")
                and provider_config.redirect_uri
            ):
                # Extract port from provider's redirect_uri if it's a localhost URL
                import urllib.parse

                parsed = urllib.parse.urlparse(provider_config.redirect_uri)
                if parsed.hostname in ["localhost", "127.0.0.1"] and parsed.port:
                    self.port = parsed.port
                    self.redirect_uri = f"http://localhost:{self.port}/callback"

        # Generate PKCE parameters if provider supports it
        code_verifier = None
        if provider.supports_pkce:
            import base64
            import secrets

            code_verifier = (
                base64.urlsafe_b64encode(secrets.token_bytes(32)).decode().rstrip("=")
            )

        # Generate state for CSRF protection
        import secrets

        state = secrets.token_urlsafe(32)

        # Store session data
        await self.session_manager.create_session(
            state,
            {
                "provider": provider_name,
                "code_verifier": code_verifier,
                "redirect_uri": self.redirect_uri,
            },
        )

        # Get authorization URL
        auth_url = await provider.get_authorization_url(state, code_verifier)

        console.print(f"\n[cyan]Starting OAuth login for {provider_name}...[/cyan]")
        console.print(f"Authorization URL: {auth_url}")

        # Open browser if requested
        if open_browser:
            console.print("\n[yellow]Opening browser for authentication...[/yellow]")
            webbrowser.open(auth_url)
        else:
            console.print(
                "\n[yellow]Please visit the URL above to authenticate[/yellow]"
            )

        # Start callback server and wait for response
        console.print(f"\n[dim]Waiting for OAuth callback on port {self.port}...[/dim]")
        console.print(
            "[dim]Or if you see the authorization code in your browser, press Enter to input it manually[/dim]"
        )

        try:
            # Start temporary HTTP server to handle callback
            code = await self._wait_for_callback(state, timeout)

            # Exchange code for tokens
            console.print("\n[cyan]Exchanging authorization code for tokens...[/cyan]")

            # Get session data
            session_data = await self.session_manager.get_session(state)
            if not session_data:
                raise ValueError("Session expired or not found")

            # Handle callback through provider
            credentials = await provider.handle_callback(
                code, state, session_data.get("code_verifier")
            )

            # Clean up session
            await self.session_manager.delete_session(state)

            console.print(
                f"\n[green]✓ Successfully authenticated with {provider_name}![/green]"
            )

            return credentials

        except TimeoutError:
            await self.session_manager.delete_session(state)
            raise TimeoutError(
                f"OAuth flow timed out after {timeout} seconds. Please try again."
            ) from None
        except Exception as e:
            await self.session_manager.delete_session(state)
            logger.error(
                "oauth_login_error",
                provider=provider_name,
                error=str(e),
                exc_info=e,
            )
            raise

    async def _wait_for_callback(self, state: str, timeout: int) -> str:
        """Wait for OAuth callback with authorization code.

        Args:
            state: Expected state parameter
            timeout: Timeout in seconds

        Returns:
            Authorization code from callback

        Raises:
            TimeoutError: If no callback received within timeout
            ValueError: If callback contains error or invalid state
        """
        from aiohttp import web

        code_future: asyncio.Future[str] = asyncio.Future()
        manual_entry_requested = asyncio.Event()

        # Allow immediate manual entry
        manual_entry_requested.set()

        async def handle_callback(request: web.Request) -> web.Response:
            """Handle OAuth callback request."""
            # Extract parameters
            params = request.rel_url.query

            # Check for error
            if "error" in params:
                error = params.get("error")
                error_desc = params.get("error_description", "No description")
                code_future.set_exception(
                    ValueError(f"OAuth error: {error} - {error_desc}")
                )
                return web.Response(
                    text=f"Authentication failed: {error_desc}",
                    status=400,
                )

            # Validate state
            callback_state = params.get("state")
            if callback_state != state:
                code_future.set_exception(
                    ValueError("Invalid state parameter - possible CSRF attack")
                )
                return web.Response(
                    text="Invalid state parameter",
                    status=400,
                )

            # Extract code
            code = params.get("code")
            if not code:
                code_future.set_exception(
                    ValueError("No authorization code in callback")
                )
                return web.Response(
                    text="No authorization code received",
                    status=400,
                )

            # Set the result
            code_future.set_result(code)

            # Return success page
            return web.Response(
                text="""
                <html>
                <head><title>Authentication Successful</title></head>
                <body style="font-family: sans-serif; text-align: center; padding: 50px;">
                    <h1>✓ Authentication Successful!</h1>
                    <p>You can now close this window and return to the terminal.</p>
                    <script>window.setTimeout(function(){window.close()}, 2000);</script>
                </body>
                </html>
                """,
                content_type="text/html",
            )

        # Create web app
        app = web.Application()
        app.router.add_get("/callback", handle_callback)

        # Start server
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "localhost", self.port)

        # Try to start the server, but handle port conflicts gracefully
        try:
            await site.start()
        except OSError as e:
            # Port might be in use, allow manual entry
            console.print(
                f"\n[yellow]Warning: Could not start callback server on port {self.port}: {e}[/yellow]"
            )
            console.print(
                "[yellow]You can still enter the authorization code manually.[/yellow]"
            )
            manual_entry_requested.set()

        try:
            # Create task to wait for automatic callback
            auto_callback_task = asyncio.create_task(
                self._wait_for_auto_callback(
                    code_future, timeout, manual_entry_requested
                )
            )

            # Create task to handle manual input
            manual_input_task = asyncio.create_task(
                self._wait_for_manual_input(manual_entry_requested)
            )

            # Wait for either automatic callback or manual input
            done, pending = await asyncio.wait(
                [auto_callback_task, manual_input_task],
                return_when=asyncio.FIRST_COMPLETED,
            )

            # Cancel the other task
            for task in pending:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

            # Get the result from the completed task
            for task in done:
                if task.exception():
                    raise task.exception()
                return task.result()

        finally:
            # Clean up server
            await runner.cleanup()

    async def _wait_for_auto_callback(
        self,
        code_future: asyncio.Future[str],
        timeout: int,
        manual_entry_requested: asyncio.Event,
    ) -> str:
        """Wait for automatic OAuth callback.

        Args:
            code_future: Future to receive the authorization code
            timeout: Timeout in seconds
            manual_entry_requested: Event signaling manual entry was requested

        Returns:
            Authorization code from callback
        """
        try:
            # Wait for callback with timeout
            code: str = await asyncio.wait_for(code_future, timeout=timeout)
            return code
        except TimeoutError:
            # If we timeout and haven't requested manual entry yet, do so now
            if not manual_entry_requested.is_set():
                console.print(
                    "\n[yellow]Callback timeout. You can enter the authorization code manually.[/yellow]"
                )
                manual_entry_requested.set()
            # Re-raise to be handled by the caller
            raise

    async def _wait_for_manual_input(
        self, manual_entry_requested: asyncio.Event
    ) -> str:
        """Wait for manual authorization code input.

        Args:
            manual_entry_requested: Event to wait for before prompting

        Returns:
            Authorization code entered by user
        """
        # Wait for signal that manual entry is needed
        await manual_entry_requested.wait()

        # Prompt user for manual code entry
        console.print("\n[cyan]Manual Code Entry Available[/cyan]")
        console.print(
            "After authorizing in your browser, you can enter the authorization code here."
        )
        console.print("[dim]Look for the 'code' parameter in the redirect URL.[/dim]")
        console.print(
            "[dim]The URL will look like: http://localhost:XXXX/callback?code=YOUR_CODE&state=...[/dim]"
        )
        console.print(
            "\n[dim]Press Enter to input code manually, or wait for automatic callback[/dim]"
        )
        console.print("[dim]Press Ctrl+C to cancel at any time[/dim]")

        # Use asyncio-compatible input
        from aioconsole import ainput

        # Wait for user to press Enter to start manual entry
        await ainput("")

        while True:
            try:
                code = await ainput(
                    "\nEnter the authorization code (or 'cancel' to abort): "
                )
                code = code.strip()

                if code.lower() == "cancel":
                    raise ValueError("OAuth flow cancelled by user")

                if not code:
                    console.print("[red]Please enter a valid authorization code.[/red]")
                    continue

                # Clean the code - remove any URL fragment (part after #)
                if "#" in code:
                    code = code.split("#")[0]
                    console.print("[dim]Note: Removed URL fragment from code[/dim]")

                # Basic validation - codes are typically alphanumeric with hyphens/underscores
                if len(code) < 10:
                    console.print(
                        "[red]Authorization code seems too short. Please check and try again.[/red]"
                    )
                    continue

                return code

            except (EOFError, KeyboardInterrupt):
                raise ValueError("OAuth flow cancelled") from None

    async def refresh_token(self, provider_name: str, refresh_token: str) -> Any:
        """Refresh access token for a provider.

        Args:
            provider_name: Name of the OAuth provider
            refresh_token: Refresh token to use

        Returns:
            New credentials

        Raises:
            ValueError: If provider not found or doesn't support refresh
        """
        provider = self.registry.get_provider(provider_name)

        if not provider:
            raise ValueError(f"OAuth provider '{provider_name}' not found")

        if not provider.supports_refresh:
            raise ValueError(
                f"Provider '{provider_name}' does not support token refresh"
            )

        return await provider.refresh_access_token(refresh_token)

    async def revoke_token(self, provider_name: str, token: str) -> None:
        """Revoke a token for a provider.

        Args:
            provider_name: Name of the OAuth provider
            token: Token to revoke

        Raises:
            ValueError: If provider not found
        """
        provider = self.registry.get_provider(provider_name)

        if not provider:
            raise ValueError(f"OAuth provider '{provider_name}' not found")

        await provider.revoke_token(token)

    async def login_manual(self, provider_name: str) -> Any:
        """Perform OAuth login with manual code entry.

        Args:
            provider_name: Name of the OAuth provider

        Returns:
            Authentication credentials

        Raises:
            ValueError: If provider not found
        """
        # Get provider from handler's registry
        provider = self.registry.get_provider(provider_name)

        if not provider:
            available = list(self.registry.list_providers().keys())
            raise ValueError(
                f"OAuth provider '{provider_name}' not found. "
                f"Available providers: {', '.join(available)}"
            )

        # Update port and redirect_uri from provider config if available
        provider_config = provider.get_config()
        # Prefer explicit resolver on config if present
        if hasattr(provider_config, "get_redirect_uri"):
            import urllib.parse

            ru = provider_config.get_redirect_uri()
            self.redirect_uri = ru
            parsed = urllib.parse.urlparse(ru)
            if parsed.hostname in ["localhost", "127.0.0.1"] and parsed.port:
                self.port = parsed.port
        else:
            if hasattr(provider_config, "callback_port"):
                self.port = provider_config.callback_port
                self.redirect_uri = f"http://localhost:{self.port}/callback"
            # If provider has a specific redirect_uri configured, use it
            if (
                hasattr(provider_config, "redirect_uri")
                and provider_config.redirect_uri
            ):
                # Extract port from provider's redirect_uri if it's a localhost URL
                import urllib.parse

                parsed = urllib.parse.urlparse(provider_config.redirect_uri)
                if parsed.hostname in ["localhost", "127.0.0.1"] and parsed.port:
                    self.port = parsed.port
                    self.redirect_uri = f"http://localhost:{self.port}/callback"

        # Generate PKCE parameters if provider supports it
        code_verifier = None
        if provider.supports_pkce:
            import base64
            import secrets

            code_verifier = (
                base64.urlsafe_b64encode(secrets.token_bytes(32)).decode().rstrip("=")
            )

        # Generate state for CSRF protection
        import secrets

        state = secrets.token_urlsafe(32)

        # Get authorization URL
        auth_url = await provider.get_authorization_url(state, code_verifier)

        console.print(f"\n[cyan]Manual OAuth Login for {provider_name}[/cyan]")
        console.print("\n[bold]Step 1:[/bold] Open this URL in your browser:")
        console.print(f"[blue]{auth_url}[/blue]")

        console.print(
            "\n[bold]Step 2:[/bold] After authorizing, you'll be redirected to a callback URL."
        )

        # Special instructions for Claude API with console callback
        if provider_name == "claude-api" and "console.anthropic.com" in auth_url:
            console.print(
                "[yellow]Note: You'll be redirected to the Anthropic Console.[/yellow]"
            )
            console.print(
                "[dim]The URL will look like: https://console.anthropic.com/oauth/code/callback?code=AUTH_CODE[/dim]"
            )
        else:
            console.print(
                f"[dim]The URL will look like: http://localhost:{self.port}/callback?code=AUTH_CODE&state=STATE_VALUE[/dim]"
            )

        console.print("\n[bold]Step 3:[/bold] Copy the authorization code from the URL")
        console.print(
            "[dim]The code is the value after 'code=' in the redirect URL[/dim]"
        )
        console.print(
            "[dim]Example: If the URL contains 'code=abc123def456', copy 'abc123def456'[/dim]"
        )

        # Get manual code input
        from aioconsole import ainput

        while True:
            try:
                code = await ainput(
                    "\nEnter the authorization code (or 'cancel' to abort): "
                )
                code = code.strip()

                if code.lower() == "cancel":
                    raise ValueError("OAuth flow cancelled by user")

                if not code:
                    console.print("[red]Please enter a valid authorization code.[/red]")
                    continue

                # Clean the code - remove any URL fragment (part after #)
                if "#" in code:
                    code = code.split("#")[0]
                    console.print("[dim]Note: Removed URL fragment from code[/dim]")

                # Basic validation - codes are typically alphanumeric with hyphens/underscores
                if len(code) < 10:
                    console.print(
                        "[red]Authorization code seems too short. Please check and try again.[/red]"
                    )
                    continue

                break

            except EOFError:
                raise ValueError("OAuth flow cancelled - no input available") from None

        # Exchange code for tokens
        console.print("\n[cyan]Exchanging authorization code for tokens...[/cyan]")

        # Handle callback through provider
        credentials = await provider.handle_callback(code, state, code_verifier)

        console.print(
            f"\n[green]✓ Successfully authenticated with {provider_name}![/green]"
        )

        return credentials

    async def check_status(self, provider_name: str) -> dict[str, Any]:
        """Check authentication status for a provider.

        Args:
            provider_name: Name of the OAuth provider

        Returns:
            Status information including whether authenticated

        Raises:
            ValueError: If provider not found
        """
        provider = self.registry.get_provider(provider_name)

        if not provider:
            raise ValueError(f"OAuth provider '{provider_name}' not found")

        # Try to load stored credentials
        storage = provider.get_storage()
        if not storage:
            return {
                "authenticated": False,
                "provider": provider_name,
                "message": "No storage configured",
            }

        credentials = await storage.load()
        if not credentials:
            return {
                "authenticated": False,
                "provider": provider_name,
                "message": "No stored credentials",
            }

        # Let the provider determine expiration status
        # This keeps provider-specific logic in the provider
        is_expired = False
        has_refresh = False

        # Use generic checks that work for any credential type
        if hasattr(credentials, "is_expired"):
            is_expired = credentials.is_expired

        # Check for refresh token generically
        if hasattr(credentials, "refresh_token"):
            has_refresh = bool(credentials.refresh_token)

        return {
            "authenticated": True,
            "provider": provider_name,
            "expired": is_expired,
            "has_refresh_token": has_refresh,
            "storage_location": storage.get_location()
            if hasattr(storage, "get_location")
            else None,
        }
