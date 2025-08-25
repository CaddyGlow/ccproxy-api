"""OAuth integration for CLI commands."""

import asyncio
import webbrowser
from typing import Any

from rich.console import Console

from ccproxy.auth.oauth.registry import get_oauth_registry
from ccproxy.auth.oauth.session import OAuthSessionManager
from ccproxy.core.logging import get_logger


logger = get_logger(__name__)
console = Console()


class CLIOAuthHandler:
    """Handles OAuth flows for CLI commands."""

    def __init__(self, port: int = 9999):
        """Initialize OAuth handler.

        Args:
            port: Local port for OAuth callback server
        """
        self.port = port
        self.redirect_uri = f"http://localhost:{port}/callback"
        self.session_manager = OAuthSessionManager()

    async def list_providers(self) -> dict[str, Any]:
        """List all available OAuth providers.

        Returns:
            Dictionary of provider info
        """
        registry = get_oauth_registry()
        return registry.list_providers()

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
        registry = get_oauth_registry()
        provider = registry.get_provider(provider_name)

        if not provider:
            available = list(registry.list_providers().keys())
            raise ValueError(
                f"OAuth provider '{provider_name}' not found. "
                f"Available providers: {', '.join(available)}"
            )

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
        await site.start()

        try:
            # Wait for callback with timeout
            code: str = await asyncio.wait_for(code_future, timeout=timeout)
            return code
        finally:
            # Clean up server
            await runner.cleanup()

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
        registry = get_oauth_registry()
        provider = registry.get_provider(provider_name)

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
        registry = get_oauth_registry()
        provider = registry.get_provider(provider_name)

        if not provider:
            raise ValueError(f"OAuth provider '{provider_name}' not found")

        await provider.revoke_token(token)

    async def check_status(self, provider_name: str) -> dict[str, Any]:
        """Check authentication status for a provider.

        Args:
            provider_name: Name of the OAuth provider

        Returns:
            Status information including whether authenticated

        Raises:
            ValueError: If provider not found
        """
        registry = get_oauth_registry()
        provider = registry.get_provider(provider_name)

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
