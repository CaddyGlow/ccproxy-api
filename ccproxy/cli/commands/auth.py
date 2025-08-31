"""Authentication and credential management commands."""

import asyncio
import contextlib
from typing import Annotated, Any

import typer
from rich import box
from rich.console import Console
from rich.table import Table
from structlog import get_logger

from ccproxy.cli.helpers import get_rich_toolkit
from ccproxy.config.settings import get_settings


app = typer.Typer(name="auth", help="Authentication and credential management")

console = Console()
logger = get_logger(__name__)


def _suppress_debug_logging() -> None:
    pass


async def initialize_plugins_for_oauth() -> None:
    """Initialize plugins to register their OAuth providers."""
    from ccproxy.auth.oauth.registry import get_oauth_registry
    from ccproxy.config.settings import get_settings
    from ccproxy.plugins.discovery import discover_and_load_plugins
    from ccproxy.plugins.factory import AuthProviderPluginFactory

    settings = get_settings()
    plugins = discover_and_load_plugins(settings)

    # Register OAuth providers from auth provider plugins
    registry = get_oauth_registry()
    for plugin_name, factory in plugins.items():
        # Check if this is an auth provider plugin
        if isinstance(factory, AuthProviderPluginFactory):
            try:
                provider = factory.create_auth_provider()
                registry.register_provider(provider)
                logger.debug(
                    f"Registered OAuth provider from plugin: {plugin_name}",
                    category="auth",
                )
            except Exception as e:
                logger.warning(
                    f"Failed to register OAuth provider from {plugin_name}: {e}"
                )


async def discover_oauth_providers() -> dict[str, tuple[str, str]]:
    """Discover available OAuth-enabled providers.

    Returns:
        Dictionary mapping provider names to (auth_type, description) tuples
    """
    from ccproxy.auth.oauth.registry import get_oauth_registry

    # Initialize plugins first to register providers
    await initialize_plugins_for_oauth()

    registry = get_oauth_registry()
    providers = registry.list_providers()

    result = {}
    for name, info in providers.items():
        result[name] = ("oauth", info.description)

    return result


def get_oauth_provider_choices() -> list[str]:
    """Get list of available OAuth provider names for CLI choices."""
    providers = asyncio.run(discover_oauth_providers())
    return list(providers.keys())


async def get_plugin_for_provider(provider: str) -> Any:
    """Get the plugin instance for the specified provider.

    Args:
        provider: Provider name (e.g., 'claude_api', 'codex')

    Returns:
        Plugin instance for the provider

    Raises:
        ValueError: If provider not found or doesn't support OAuth
    """
    # TODO: Integrate with v2 plugin system
    # For now, this is a stub that will be replaced when v2 integration is complete
    raise ValueError(
        f"OAuth provider '{provider}' integration pending v2 plugin system update"
    )


async def get_oauth_client_for_provider(provider: str) -> Any:
    """Get OAuth client for the specified provider.

    Args:
        provider: Provider name (e.g., 'claude_api', 'codex')

    Returns:
        OAuth client instance for the provider

    Raises:
        ValueError: If provider not found or doesn't support OAuth
    """
    plugin = await get_plugin_for_provider(provider)

    # Initialize plugin with minimal CoreServices for CLI context
    import httpx
    import structlog

    from ccproxy.core.services import CoreServices

    settings = get_settings()

    # Create minimal services for CLI usage
    async with httpx.AsyncClient() as client:
        services = CoreServices(
            settings=settings,
            http_client=client,
            logger=structlog.get_logger(),
        )

        # Initialize the plugin
        await plugin.initialize(services)

        # Now get the OAuth client
        oauth_client = await plugin.get_oauth_client()
        if not oauth_client:
            raise ValueError(f"Provider '{provider}' does not implement OAuth client")
        return oauth_client


async def check_provider_credentials(provider: str) -> dict[str, Any]:
    """Check if provider has valid stored credentials.

    Args:
        provider: Provider name

    Returns:
        Dictionary with credential status information
    """
    try:
        # Get plugin for provider and use it to check credentials
        plugin = await get_plugin_for_provider(provider)
        oauth_client = await plugin.get_oauth_client()

        if not oauth_client:
            return {
                "has_credentials": False,
                "expired": True,
                "path": None,
                "credentials": None,
            }

        # Try to get profile info to test if credentials are valid
        # This will use the plugin's internal credential checking logic
        profile_info = await plugin.get_profile_info()

        # Basic credential status based on whether we can get profile info
        has_credentials = profile_info is not None

        return {
            "has_credentials": has_credentials,
            "expired": not has_credentials,
            "path": None,  # Plugin-specific, would need to be added to protocol if needed
            "credentials": None,  # Plugin-specific, would need to be added to protocol if needed
        }

    except AttributeError as e:
        logger.debug(
            "credentials_check_missing_attribute",
            provider=provider,
            error=str(e),
            exc_info=e,
        )
        # If we can't check credentials, assume none exist
        return {
            "has_credentials": False,
            "expired": True,
            "path": None,
            "credentials": None,
        }
    except FileNotFoundError as e:
        logger.debug(
            "credentials_file_not_found", provider=provider, error=str(e), exc_info=e
        )
        # If we can't check credentials, assume none exist
        return {
            "has_credentials": False,
            "expired": True,
            "path": None,
            "credentials": None,
        }
    except Exception as e:
        logger.debug(
            "credentials_check_failed", provider=provider, error=str(e), exc_info=e
        )
        # If we can't check credentials, assume none exist
        return {
            "has_credentials": False,
            "expired": True,
            "path": None,
            "credentials": None,
        }


@app.command(name="providers")
def list_providers() -> None:
    """List all available OAuth providers."""
    _suppress_debug_logging()
    toolkit = get_rich_toolkit()
    toolkit.print("[bold cyan]Available OAuth Providers[/bold cyan]", centered=True)
    toolkit.print_line()

    try:
        providers = asyncio.run(discover_oauth_providers())

        if not providers:
            toolkit.print("No OAuth providers found", tag="warning")
            return

        table = Table(
            show_header=True,
            header_style="bold cyan",
            box=box.ROUNDED,
            title="OAuth Providers",
            title_style="bold white",
        )
        table.add_column("Provider", style="cyan")
        table.add_column("Auth Type", style="white")
        table.add_column("Description", style="dim")

        for name, (auth_type, description) in providers.items():
            table.add_row(name, auth_type, description)

        console.print(table)

    except ImportError as e:
        toolkit.print(f"Plugin import error: {e}", tag="error")
        raise typer.Exit(1) from e
    except AttributeError as e:
        toolkit.print(f"Plugin configuration error: {e}", tag="error")
        raise typer.Exit(1) from e
    except Exception as e:
        toolkit.print(f"Error listing providers: {e}", tag="error")
        raise typer.Exit(1) from e


@app.command(name="login")
def login_command(
    provider: Annotated[
        str,
        typer.Argument(
            help="Provider to authenticate with (claude-api, codex, openai)"
        ),
    ],
    no_browser: Annotated[
        bool,
        typer.Option("--no-browser", help="Don't automatically open browser for OAuth"),
    ] = False,
    manual: Annotated[
        bool,
        typer.Option(
            "--manual", "-m", help="Skip callback server and enter code manually"
        ),
    ] = False,
) -> None:
    """Login to a provider using OAuth authentication.

    Examples:
        ccproxy auth login claude-api     # Claude API OAuth login
        ccproxy auth login codex          # Codex/OpenAI OAuth login
        ccproxy auth login openai         # Alias for codex
        ccproxy auth login claude-api --manual  # Manual code entry
    """
    _suppress_debug_logging()
    toolkit = get_rich_toolkit()

    # Normalize provider names
    provider = provider.lower()
    if provider == "openai":
        provider = "codex"  # OpenAI is an alias for codex

    # Handle OAuth providers (claude-api, codex)
    toolkit.print(
        f"[bold cyan]OAuth Login - {provider.replace('_', '-').title()}[/bold cyan]",
        centered=True,
    )
    toolkit.print_line()

    try:
        # Validate provider exists
        providers = asyncio.run(discover_oauth_providers())
        if provider not in providers:
            available = ", ".join(providers.keys()) if providers else "none"
            toolkit.print(
                f"Provider '{provider}' not found. Available OAuth providers: {available}",
                tag="error",
            )
            raise typer.Exit(1)

        # Use the new OAuth integration handler
        from ccproxy.cli.oauth_integration import CLIOAuthHandler

        # Use different ports for different providers to avoid conflicts
        port = 9999 if provider == "claude-api" else 1455
        handler = CLIOAuthHandler(port=port)

        try:
            # If manual flag is set, use a different approach
            if manual:
                credentials = asyncio.run(handler.login_manual(provider))
            else:
                credentials = asyncio.run(
                    handler.login(provider, open_browser=not no_browser)
                )

            toolkit.print(f"Successfully logged in to {provider}!", tag="success")

            # Show credential summary using provider's method
            console.print(f"\n[dim]Authentication successful for {provider}[/dim]")

            from ccproxy.auth.oauth.registry import get_oauth_registry

            registry = get_oauth_registry()
            oauth_provider = registry.get_provider(provider)

            if oauth_provider and hasattr(oauth_provider, "get_credential_summary"):
                try:
                    summary = oauth_provider.get_credential_summary(credentials)

                    # Display summary info
                    if summary.get("subscription_type"):
                        console.print(f"  Subscription: {summary['subscription_type']}")
                    if summary.get("account_id"):
                        console.print(f"  Account ID: {summary['account_id']}")
                    if summary.get("scopes"):
                        console.print(f"  Scopes: {', '.join(summary['scopes'])}")
                    if summary.get("expires_at"):
                        console.print(f"  Token expires: {summary['expires_at']}")

                    # Show storage location
                    storage = oauth_provider.get_storage()
                    if storage and hasattr(storage, "get_location"):
                        console.print(f"  Stored at: {storage.get_location()}")
                except Exception as e:
                    logger.debug(f"Could not get credential summary: {e}")

        except TimeoutError as e:
            toolkit.print(f"OAuth login timed out: {e}", tag="error")
            raise typer.Exit(1) from e
        except ValueError as e:
            toolkit.print(f"OAuth configuration error: {e}", tag="error")
            raise typer.Exit(1) from e

    except ValueError as e:
        toolkit.print(f"Configuration error during {provider} login: {e}", tag="error")
        raise typer.Exit(1) from e
    except KeyboardInterrupt:
        console.print("\n[yellow]Login cancelled by user.[/yellow]")
        raise typer.Exit(1) from None
    except OSError as e:
        toolkit.print(f"OAuth server error during {provider} login: {e}", tag="error")
        raise typer.Exit(1) from e
    except ImportError as e:
        toolkit.print(
            f"Failed to import required modules for {provider}: {e}", tag="error"
        )
        raise typer.Exit(1) from e
    except Exception as e:
        toolkit.print(f"Error during {provider} login: {e}", tag="error")
        raise typer.Exit(1) from e


@app.command(name="status")
def status_command(
    provider: Annotated[
        str,
        typer.Argument(help="Provider to check status (claude-api, codex, openai)"),
    ],
    detailed: Annotated[
        bool,
        typer.Option("--detailed", "-d", help="Show detailed credential information"),
    ] = False,
) -> None:
    """Check authentication status and info for specified provider.

    Shows authentication status, credential validity, and account information.

    Examples:
        ccproxy auth status claude-api   # Claude API OAuth status
        ccproxy auth status codex        # Codex/OpenAI status
        ccproxy auth status -d codex     # Detailed info with tokens
    """
    _suppress_debug_logging()
    toolkit = get_rich_toolkit()

    # Store original provider name for display
    display_provider = provider.lower()

    # Normalize provider names for internal use
    provider = provider.lower()
    if provider == "openai":
        provider = "codex"

    # Use proper display names
    display_name = {
        "openai": "OpenAI",
        "codex": "Codex",
        "claude-api": "Claude-Api",
    }.get(display_provider, display_provider.replace("_", "-").title())

    toolkit.print(
        f"[bold cyan]{display_name} Authentication Status[/bold cyan]",
        centered=True,
    )
    toolkit.print_line()

    try:
        # Get the OAuth provider for this provider name
        oauth_provider = asyncio.run(get_oauth_provider_for_name(provider))

        profile_info = None
        credentials = None

        if oauth_provider:
            # Try to load credentials using the provider's storage
            try:
                credentials = asyncio.run(oauth_provider.load_credentials())

                if credentials:
                    # Get credential summary from the provider
                    if hasattr(oauth_provider, "get_credential_summary"):
                        summary = oauth_provider.get_credential_summary(credentials)

                        # Build profile info from summary
                        profile_info = {}

                        # Map common fields
                        if "email" in summary:
                            profile_info["email"] = summary["email"]
                        if "account_id" in summary:
                            profile_info["account_id"] = summary["account_id"]
                        if "subscription_type" in summary:
                            profile_info["subscription_type"] = summary[
                                "subscription_type"
                            ]
                        if "expires_at" in summary:
                            profile_info["expires_at"] = summary["expires_at"]
                        if "scopes" in summary:
                            profile_info["scopes"] = summary["scopes"]

                        # Add any other fields from summary
                        for key, value in summary.items():
                            if key not in profile_info:
                                profile_info[key] = value
                    else:
                        # Fallback to basic info if no summary method
                        profile_info = {
                            "provider": provider,
                            "authenticated": True,
                        }

            except Exception as e:
                logger.debug(f"{provider}_status_error", error=str(e), exc_info=e)

        if profile_info:
            console.print("[green]✓[/green] Authenticated with valid credentials")

            # Display profile information generically
            console.print("\n[bold]Account Information[/bold]")

            # Define field display names
            field_display_names = {
                "email": "Email",
                "organization_name": "Organization",
                "organization_type": "Organization Type",
                "subscription_type": "Subscription",
                "plan_type": "Plan",
                "user_id": "User ID",
                "account_id": "Account ID",
                "full_name": "Full Name",
                "display_name": "Display Name",
                "has_claude_pro": "Claude Pro",
                "has_claude_max": "Claude Max",
                "rate_limit_tier": "Rate Limit Tier",
                "billing_type": "Billing Type",
                "expires_at": "Expires At",
                "scopes": "Scopes",
                "email_verified": "Email Verified",
                "subscription_start": "Subscription Start",
                "subscription_until": "Subscription Until",
                "organization_role": "Organization Role",
                "organization_id": "Organization ID",
            }

            # Display fields in a sensible order
            priority_fields = [
                "email",
                "organization_name",
                "subscription_type",
                "plan_type",
                "expires_at",
            ]

            # Show priority fields first
            for field in priority_fields:
                if field in profile_info:
                    display_name = field_display_names.get(
                        field, field.replace("_", " ").title()
                    )
                    value = profile_info[field]

                    # Format special values
                    if field == "scopes" and isinstance(value, list):
                        value = ", ".join(value)
                    elif field == "user_id" and len(str(value)) > 20:
                        value = f"{str(value)[:12]}..."
                    elif isinstance(value, bool):
                        value = "Yes" if value else "No"

                    console.print(f"  {display_name}: {value}")

            # Show remaining fields
            for field, value in profile_info.items():
                if field not in priority_fields and field not in [
                    "provider",
                    "authenticated",
                ]:
                    display_name = field_display_names.get(
                        field, field.replace("_", " ").title()
                    )

                    # Format special values
                    if field == "scopes" and isinstance(value, list):
                        value = ", ".join(value)
                    elif field == "user_id" and len(str(value)) > 20:
                        value = f"{str(value)[:12]}..."
                    elif isinstance(value, bool):
                        value = "Yes" if value else "No"

                    console.print(f"  {display_name}: {value}")

            # For detailed mode, try to show token preview if available
            if detailed and credentials:
                # Try to extract token for preview
                token_str = None

                if hasattr(credentials, "access_token"):
                    # Direct access token
                    token_str = str(credentials.access_token)
                elif hasattr(credentials, "claude_ai_oauth"):
                    # Claude OAuth structure
                    oauth = credentials.claude_ai_oauth
                    if hasattr(oauth, "access_token"):
                        if hasattr(oauth.access_token, "get_secret_value"):
                            token_str = oauth.access_token.get_secret_value()
                        else:
                            token_str = str(oauth.access_token)

                if token_str and len(token_str) > 20:
                    token_preview = f"{token_str[:8]}...{token_str[-8:]}"
                    console.print(f"\n  Token: [dim]{token_preview}[/dim]")
        else:
            # No profile info means not authenticated or provider doesn't exist
            console.print("[red]✗[/red] Not authenticated or provider not found")
            console.print(f"  Run 'ccproxy auth login {provider}' to authenticate")

    except ImportError as e:
        console.print(f"[red]✗[/red] Failed to import required modules: {e}")
        raise typer.Exit(1) from e
    except AttributeError as e:
        console.print(f"[red]✗[/red] Configuration or plugin error: {e}")
        raise typer.Exit(1) from e
    except Exception as e:
        console.print(f"[red]✗[/red] Error checking status: {e}")
        raise typer.Exit(1) from e


@app.command(name="logout")
def logout_command(
    provider: Annotated[
        str, typer.Argument(help="Provider to logout from (codex, openai)")
    ],
) -> None:
    """Logout and remove stored credentials for specified provider.

    Examples:
        ccproxy auth logout codex
        ccproxy auth logout openai
    """
    _suppress_debug_logging()
    toolkit = get_rich_toolkit()

    # Normalize provider names
    provider = provider.lower()
    if provider == "openai":
        provider = "codex"

    toolkit.print(f"[bold cyan]{provider.title()} Logout[/bold cyan]", centered=True)
    toolkit.print_line()

    try:
        # Get the OAuth provider for this provider name
        oauth_provider = asyncio.run(get_oauth_provider_for_name(provider))

        if not oauth_provider:
            toolkit.print(f"Provider '{provider}' not found", tag="error")
            raise typer.Exit(1)

        # Check if credentials exist
        existing_creds = None
        with contextlib.suppress(Exception):
            existing_creds = asyncio.run(oauth_provider.load_credentials())

        if not existing_creds:
            console.print("[yellow]No credentials found. Already logged out.[/yellow]")
            return

        # Confirm logout
        confirm = typer.confirm(
            "Are you sure you want to logout and remove credentials?"
        )
        if not confirm:
            console.print("Logout cancelled.")
            return

        # Delete credentials using provider's storage
        success = False
        try:
            storage = oauth_provider.get_storage()
            if storage and hasattr(storage, "delete"):
                success = asyncio.run(storage.delete())
            elif storage and hasattr(storage, "clear"):
                success = asyncio.run(storage.clear())
            else:
                # Try to delete through save with None
                success = asyncio.run(oauth_provider.save_credentials(None))
        except Exception as e:
            logger.debug("logout_error", error=str(e), exc_info=e)

        if success:
            toolkit.print(f"Successfully logged out from {provider}!", tag="success")
            console.print("Credentials have been removed.")
        else:
            toolkit.print("Failed to remove credentials", tag="error")
            raise typer.Exit(1)

    except FileNotFoundError as e:
        toolkit.print("No credentials found to remove.", tag="warning")
        # Don't exit with error for this case
    except OSError as e:
        toolkit.print(f"Failed to remove credential files: {e}", tag="error")
        raise typer.Exit(1) from e
    except ImportError as e:
        toolkit.print(f"Failed to import required modules: {e}", tag="error")
        raise typer.Exit(1) from e
    except Exception as e:
        toolkit.print(f"Error during logout: {e}", tag="error")
        raise typer.Exit(1) from e


# OpenAI Codex Authentication Commands


async def get_oauth_provider_for_name(provider: str) -> Any:
    """Get OAuth provider instance for the specified provider name.

    Args:
        provider: Provider name (e.g., 'claude-api', 'codex')

    Returns:
        OAuth provider instance or None if not found
    """
    from ccproxy.auth.oauth.registry import get_oauth_registry
    from ccproxy.config.settings import get_settings
    from ccproxy.plugins.discovery import discover_and_load_plugins
    from ccproxy.plugins.factory import AuthProviderPluginFactory

    # First ensure plugins are loaded to register providers
    settings = get_settings()
    plugins = discover_and_load_plugins(settings)

    # Register OAuth providers from auth provider plugins
    registry = get_oauth_registry()
    for _plugin_name, factory in plugins.items():
        if isinstance(factory, AuthProviderPluginFactory):
            try:
                oauth_provider = factory.create_auth_provider()
                # Register if not already registered
                if not registry.has_provider(oauth_provider.provider_name):
                    registry.register_provider(oauth_provider)
            except Exception:
                pass  # Already logged in initialize_plugins_for_oauth

    # Now get the provider
    return registry.get_provider(provider)
