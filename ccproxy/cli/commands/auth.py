"""Authentication and credential management commands."""

import asyncio
import contextlib
import logging
import os
from typing import Annotated, Any

import structlog
import typer
from rich import box
from rich.console import Console
from rich.table import Table

from ccproxy.auth.oauth.registry import OAuthRegistry
from ccproxy.cli.helpers import get_rich_toolkit
from ccproxy.config.settings import get_settings
from ccproxy.core.logging import bootstrap_cli_logging, get_logger, setup_logging
from ccproxy.hooks.manager import HookManager
from ccproxy.hooks.registry import HookRegistry


app = typer.Typer(name="auth", help="Authentication and credential management")

console = Console()
logger = get_logger(__name__)


def _apply_auth_logger_level() -> None:
    """Set logger level from settings without configuring handlers.

    This ensures the auth command respects the configured log level while
    relying on the global logging configuration for handlers/formatters.
    """
    try:
        level_name = get_settings().logging.level
        level = getattr(logging, level_name.upper(), logging.INFO)
    except Exception:
        level = logging.INFO

    # Only adjust levels; do not add or modify handlers here.
    # Apply to root CCProxy logger and this module's logger.
    logging.getLogger("ccproxy").setLevel(level)
    logging.getLogger(__name__).setLevel(level)


def _ensure_logging_configured() -> None:
    """Ensure global logging is configured with the standard format.

    - If structlog is already configured, do nothing (respect global handlers).
    - Otherwise, configure using Settings.logging (same path as `serve`).
    """
    if structlog.is_configured():
        return

    # First try early bootstrap from env/argv without touching settings
    with contextlib.suppress(Exception):
        bootstrap_cli_logging()

    if structlog.is_configured():
        return

    # If still not configured, apply a safe default console setup.
    # Avoid reading settings here to prevent early logs in default format.
    level_name = os.getenv("LOGGING__LEVEL", "INFO")
    log_file = os.getenv("LOGGING__FILE")
    try:
        setup_logging(json_logs=False, log_level_name=level_name, log_file=log_file)
    except Exception:
        # Last resort: level-only
        _apply_auth_logger_level()


def _expected_plugin_class_name(provider: str) -> str:
    """Return the expected plugin class name from provider input for messaging.

    Example: 'claude_api' -> 'OauthClaudeApiPlugin', 'codex' -> 'OauthCodexPlugin'.
    """
    import re

    base = re.sub(r"[^a-zA-Z0-9]+", "_", provider.strip()).strip("_")
    parts = [p for p in base.split("_") if p]
    camel = "".join(s[:1].upper() + s[1:] for s in parts)
    return f"Oauth{camel}Plugin"


def _provider_module_path(provider: str) -> tuple[str | None, str]:
    """Map provider name to plugin module path and canonical provider key.

    Only explicit providers are supported; aliases are not.
    Returns (module_path or None, canonical_provider_key).
    """
    key = provider.strip().lower()
    # No aliasing (e.g., 'openai' will NOT map to 'codex')
    mapping: dict[str, str] = {
        "codex": "plugins.oauth_codex.plugin",
        "claude-api": "plugins.oauth_claude.plugin",
        # tolerate underscore variant for convenience
        "claude_api": "plugins.oauth_claude.plugin",
    }
    return mapping.get(key), key


async def _lazy_register_oauth_provider(
    provider: str, registry: OAuthRegistry
) -> Any | None:
    """Lazily import and register just the requested provider's plugin.

    Imports only the plugin module matching the provider, builds context, and
    registers the provider in the OAuth registry. Returns the provider instance
    or None if not found/import failed.
    """
    import importlib

    from ccproxy.config.settings import get_settings
    from ccproxy.plugins import PluginContext
    from ccproxy.plugins.factory import AuthProviderPluginFactory
    from ccproxy.services.cli_detection import CLIDetectionService

    module_path, key = _provider_module_path(provider)
    if not module_path:
        return None

    try:
        module = importlib.import_module(module_path)
    except Exception as e:
        logger.debug(
            "oauth_plugin_import_failed",
            provider=provider,
            module=module_path,
            error=str(e),
            exc_info=e,
        )
        return None

    factory = getattr(module, "factory", None)
    if not isinstance(factory, AuthProviderPluginFactory):
        logger.debug(
            "oauth_plugin_factory_missing_or_invalid",
            provider=provider,
            module=module_path,
        )
        return None

    # Prepare minimal context for the auth provider
    settings = get_settings()
    detection_service = CLIDetectionService(settings)

    hook_manager = None
    try:
        if settings.plugins.get("request_tracer", {}).get("enabled", False):
            from plugins.request_tracer.config import RequestTracerConfig
            from plugins.request_tracer.hooks.http import HTTPTracerHook

            hook_registry = HookRegistry()
            hook_manager = HookManager(hook_registry)
            tracer_config = RequestTracerConfig(
                **settings.plugins.get("request_tracer", {})
            )
            # Register HTTP tracer hook for generic HTTP interception
            http_hook = HTTPTracerHook(tracer_config)
            hook_registry.register(http_hook)
            # logger.debug("http_tracer_hook_registered", category="auth")
    except Exception as e:
        # Tracing is best-effort; continue without it
        logger.debug(
            "hook_registration_failed", error=str(e), category="auth", exc_info=e
        )
        pass

    # Create HTTP client with hook support if enabled
    from ccproxy.core.http_client import HTTPClientFactory

    http_client = HTTPClientFactory.create_client(
        settings=settings,
        hook_manager=hook_manager,  # Will use HookableHTTPClient if hook_manager is provided
    )

    context_data: dict[str, Any] = {
        "detection_service": detection_service,
        "http_client": http_client,
    }
    if hook_manager:
        context_data["hook_manager"] = hook_manager
    context = PluginContext(context_data)

    try:
        oauth_provider = factory.create_auth_provider(context)
    except Exception as e:
        logger.debug(
            "oauth_provider_create_failed", provider=provider, error=str(e), exc_info=e
        )
        return None

    try:
        # Avoid double registration if already present
        if not registry.has_provider(oauth_provider.provider_name):
            registry.register_provider(oauth_provider)
    except Exception:
        # Even if registration fails, return the instance if it matches
        pass

    return oauth_provider


async def discover_oauth_providers() -> dict[str, tuple[str, str]]:
    """Return known OAuth providers without importing all plugins.

    Keeps this lightweight: lists supported provider keys and generic descriptions.
    """
    return {
        "codex": ("oauth", "OpenAI Codex OAuth"),
        "claude-api": ("oauth", "Claude API OAuth"),
    }


def get_oauth_provider_choices() -> list[str]:
    """Get list of available OAuth provider names for CLI choices."""
    providers = asyncio.run(discover_oauth_providers())
    return list(providers.keys())


async def get_plugin_for_provider(provider: str) -> Any:
    """Deprecated: use OAuth providers directly. Retained for compatibility."""
    raise ValueError("Direct plugin access is no longer supported; use OAuth providers")


async def get_oauth_client_for_provider(provider: str, registry: OAuthRegistry) -> Any:
    """Get OAuth client for the specified provider.

    Args:
        provider: Provider name (e.g., 'claude_api', 'codex')

    Returns:
        OAuth client instance for the provider

    Raises:
        ValueError: If provider not found or doesn't support OAuth
    """
    # Load provider lazily and return its client
    oauth_provider = await get_oauth_provider_for_name(provider, registry)
    if not oauth_provider:
        raise ValueError(f"Provider '{provider}' not found")
    oauth_client = getattr(oauth_provider, "client", None)
    if not oauth_client:
        raise ValueError(f"Provider '{provider}' does not implement OAuth client")
    return oauth_client


async def check_provider_credentials(
    provider: str, registry: OAuthRegistry
) -> dict[str, Any]:
    """Check if provider has valid stored credentials.

    Args:
        provider: Provider name

    Returns:
        Dictionary with credential status information
    """
    try:
        # Lazily load provider and inspect storage
        oauth_provider = await get_oauth_provider_for_name(provider, registry)
        if not oauth_provider:
            return {
                "has_credentials": False,
                "expired": True,
                "path": None,
                "credentials": None,
            }

        # Try to load credentials via provider storage
        creds = await oauth_provider.load_credentials()
        has_credentials = creds is not None

        return {
            "has_credentials": has_credentials,
            "expired": not has_credentials,
            "path": None,  # Provider-specific; not exposed in protocol
            "credentials": None,  # Redacted
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
    _ensure_logging_configured()
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
        typer.Argument(help="Provider to authenticate with (claude-api, codex)"),
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
        ccproxy auth login claude-api --manual  # Manual code entry
    """
    _ensure_logging_configured()
    # Load settings early so configuration logs appear before other inits
    with contextlib.suppress(Exception):
        _ = get_settings()
    toolkit = get_rich_toolkit()

    # Normalize provider name (no aliasing)
    provider = provider.strip().lower()

    # Handle OAuth providers (claude-api, codex)
    toolkit.print(
        f"[bold cyan]OAuth Login - {provider.replace('_', '-').title()}[/bold cyan]",
        centered=True,
    )
    toolkit.print_line()

    try:
        # Create a local registry for this CLI session
        registry = OAuthRegistry()
        # Lazily load provider plugin and register in registry
        oauth_provider = asyncio.run(get_oauth_provider_for_name(provider, registry))
        if not oauth_provider:
            providers = asyncio.run(discover_oauth_providers())
            available = ", ".join(providers.keys()) if providers else "none"
            expected = _expected_plugin_class_name(provider)
            # Heuristic suggestion for delimiter differences
            suggestion = None
            if provider.replace("_", "-") in providers:
                suggestion = provider.replace("_", "-")
            elif provider.replace("-", "_") in providers:
                suggestion = provider.replace("-", "_")
            msg = (
                f"Provider '{provider}' not found. Available: {available}. "
                f"Expected plugin class '{expected}'."
            )
            if suggestion:
                msg += f" Did you mean '{suggestion}'?"
            toolkit.print(msg, tag="error")
            raise typer.Exit(1)

        # Use the new OAuth integration handler
        from ccproxy.cli.oauth_integration import CLIOAuthHandler

        # Get port from provider config if available
        port = 9999  # Default port
        provider_config = oauth_provider.get_config()
        if hasattr(provider_config, "callback_port"):
            port = provider_config.callback_port
        elif provider == "codex":
            port = 1455  # Keep different port for codex to avoid conflicts

        handler = CLIOAuthHandler(port=port, registry=registry)

        try:
            # If manual flag is set, use a different approach
            if manual:
                asyncio.run(_lazy_register_oauth_provider(provider, registry))
                credentials = asyncio.run(handler.login_manual(provider))
            else:
                # Ensure provider is registered in the same registry used by handler
                asyncio.run(_lazy_register_oauth_provider(provider, registry))
                credentials = asyncio.run(
                    handler.login(provider, open_browser=not no_browser)
                )

            toolkit.print(f"Successfully logged in to {provider}!", tag="success")

            # Show credential summary using provider's method
            console.print(f"\n[dim]Authentication successful for {provider}[/dim]")

            oauth_provider = handler.registry.get_provider(provider)

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
        typer.Argument(help="Provider to check status (claude-api, codex)"),
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
    _ensure_logging_configured()
    # Load settings early so configuration logs appear before other inits
    with contextlib.suppress(Exception):
        _ = get_settings()
    toolkit = get_rich_toolkit()

    # Normalize provider (no aliasing) and derive display name
    provider = provider.strip().lower()
    display_name = provider.replace("_", "-").title()

    toolkit.print(
        f"[bold cyan]{display_name} Authentication Status[/bold cyan]",
        centered=True,
    )
    toolkit.print_line()

    try:
        registry = OAuthRegistry()
        # Get the OAuth provider for this provider name
        oauth_provider = asyncio.run(get_oauth_provider_for_name(provider, registry))
        if not oauth_provider:
            providers = asyncio.run(discover_oauth_providers())
            available = ", ".join(providers.keys()) if providers else "none"
            expected = _expected_plugin_class_name(provider)
            toolkit.print(
                f"Provider '{provider}' not found. Available: {available}. Expected plugin class '{expected}'.",
                tag="error",
            )
            raise typer.Exit(1)

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
        str, typer.Argument(help="Provider to logout from (claude-api, codex)")
    ],
) -> None:
    """Logout and remove stored credentials for specified provider.

    Examples:
        ccproxy auth logout codex
        ccproxy auth logout claude-api
    """
    _ensure_logging_configured()
    # Load settings early so configuration logs appear before other inits
    with contextlib.suppress(Exception):
        _ = get_settings()
    toolkit = get_rich_toolkit()

    # Normalize provider (no aliasing)
    provider = provider.strip().lower()

    toolkit.print(f"[bold cyan]{provider.title()} Logout[/bold cyan]", centered=True)
    toolkit.print_line()

    try:
        registry = OAuthRegistry()
        # Get the OAuth provider for this provider name (lazy load)
        oauth_provider = asyncio.run(get_oauth_provider_for_name(provider, registry))

        if not oauth_provider:
            providers = asyncio.run(discover_oauth_providers())
            available = ", ".join(providers.keys()) if providers else "none"
            expected = _expected_plugin_class_name(provider)
            toolkit.print(
                f"Provider '{provider}' not found. Available: {available}. Expected plugin class '{expected}'.",
                tag="error",
            )
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


async def get_oauth_provider_for_name(provider: str, registry: OAuthRegistry) -> Any:
    """Get OAuth provider instance for the specified provider name.

    Args:
        provider: Provider name (e.g., 'claude-api', 'codex')

    Returns:
        OAuth provider instance or None if not found
    """
    # Already registered?
    existing = registry.get_provider(provider)
    if existing:
        return existing

    # Lazily import and register only this provider
    provider_instance = await _lazy_register_oauth_provider(provider, registry)
    if provider_instance:
        return provider_instance

    return None
