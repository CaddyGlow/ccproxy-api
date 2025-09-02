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
from ccproxy.config.settings import Settings
from ccproxy.core.logging import bootstrap_cli_logging, get_logger, setup_logging
from ccproxy.core.plugins.loader import load_plugin_system
from ccproxy.hooks.manager import HookManager
from ccproxy.hooks.registry import HookRegistry
from ccproxy.services.container import ServiceContainer


app = typer.Typer(name="auth", help="Authentication and credential management")

console = Console()
logger = get_logger(__name__)


def _get_service_container() -> ServiceContainer:
    """Create a service container for the auth commands."""
    settings = Settings.from_config()
    return ServiceContainer(settings)


def _apply_auth_logger_level() -> None:
    """Set logger level from settings without configuring handlers."""
    try:
        level_name = _get_service_container().get_service(Settings).logging.level
        level = getattr(logging, level_name.upper(), logging.INFO)
    except Exception:
        level = logging.INFO

    logging.getLogger("ccproxy").setLevel(level)
    logging.getLogger(__name__).setLevel(level)


def _ensure_logging_configured() -> None:
    """Ensure global logging is configured with the standard format."""
    if structlog.is_configured():
        return

    with contextlib.suppress(Exception):
        bootstrap_cli_logging()

    if structlog.is_configured():
        return

    level_name = os.getenv("LOGGING__LEVEL", "INFO")
    log_file = os.getenv("LOGGING__FILE")
    try:
        setup_logging(json_logs=False, log_level_name=level_name, log_file=log_file)
    except Exception:
        _apply_auth_logger_level()


def _expected_plugin_class_name(provider: str) -> str:
    """Return the expected plugin class name from provider input for messaging."""
    import re

    base = re.sub(r"[^a-zA-Z0-9]+", "_", provider.strip()).strip("_")
    parts = [p for p in base.split("_") if p]
    camel = "".join(s[:1].upper() + s[1:] for s in parts)
    return f"Oauth{camel}Plugin"


def _render_profile_table(
    profile: dict[str, Any],
    title: str = "Account Information",
) -> None:
    """Render a clean, two-column table of profile data using Rich."""
    table = Table(show_header=False, box=box.SIMPLE, title=title)
    table.add_column("Field", style="bold")
    table.add_column("Value")

    def _val(v: Any) -> str:
        if v is None:
            return ""
        if hasattr(v, "isoformat"):
            try:
                return str(v)
            except Exception:
                return str(v)
        if isinstance(v, bool):
            return "Yes" if v else "No"
        if isinstance(v, list):
            return ", ".join(str(x) for x in v)
        s = str(v)
        return s

    def _row(label: str, key: str) -> None:
        if key in profile and profile[key] not in (None, "", []):
            table.add_row(label, _val(profile[key]))

    _row("Provider", "provider_type")
    _row("Account ID", "account_id")
    _row("Email", "email")
    _row("Display Name", "display_name")

    _row("Subscription", "subscription_type")
    _row("Subscription Status", "subscription_status")
    _row("Subscription Expires", "subscription_expires_at")

    _row("Organization", "organization_name")
    _row("Organization Role", "organization_role")

    _row("Has Refresh Token", "has_refresh_token")
    _row("Has ID Token", "has_id_token")
    _row("Token Expires", "token_expires_at")

    _row("Email Verified", "email_verified")

    if len(table.rows) > 0:
        console.print(table)


def _render_profile_features(profile: dict[str, Any]) -> None:
    """Render provider-specific features if present."""
    features = profile.get("features")
    if isinstance(features, dict) and features:
        table = Table(show_header=False, box=box.SIMPLE, title="Features")
        table.add_column("Feature", style="bold")
        table.add_column("Value")
        for k, v in features.items():
            name = k.replace("_", " ").title()
            val = (
                "Yes"
                if isinstance(v, bool) and v
                else ("No" if isinstance(v, bool) else str(v))
            )
            if val and val != "No":
                table.add_row(name, val)
        if len(table.rows) > 0:
            console.print(table)


def _provider_plugin_name(provider: str) -> str | None:
    """Map CLI provider name to plugin manifest name."""
    key = provider.strip().lower()
    mapping: dict[str, str] = {
        "codex": "oauth_codex",
        "claude-api": "oauth_claude",
        "claude_api": "oauth_claude",
    }
    return mapping.get(key)


async def _lazy_register_oauth_provider(
    provider: str,
    registry: OAuthRegistry,
    container: ServiceContainer,
) -> Any | None:
    """Lazily import and register just the requested provider's plugin."""
    from ccproxy.core.plugins.factory import AuthProviderPluginFactory

    plugin_name = _provider_plugin_name(provider)
    if not plugin_name:
        return None

    settings = container.get_service(Settings)
    plugin_registry, _middleware_mgr = load_plugin_system(settings)
    factory = plugin_registry.get_factory(plugin_name)
    if not isinstance(factory, AuthProviderPluginFactory):
        logger.debug(
            "oauth_plugin_factory_missing_or_invalid",
            provider=provider,
            plugin=plugin_name,
        )
        return None

    detection_service = container.get_cli_detection_service()

    hook_manager = None
    try:
        if settings.plugins.get("request_tracer", {}).get("enabled", False):
            from ccproxy.plugins.request_tracer.config import RequestTracerConfig
            from ccproxy.plugins.request_tracer.hooks.http import HTTPTracerHook

            hook_registry = HookRegistry()
            hook_manager = HookManager(hook_registry)
            tracer_config = RequestTracerConfig(
                **settings.plugins.get("request_tracer", {})
            )
            http_hook = HTTPTracerHook(tracer_config)
            hook_registry.register(http_hook)
    except Exception as e:
        logger.debug(
            "hook_registration_failed", error=str(e), category="auth", exc_info=e
        )
        pass

    from ccproxy.core.http_client import HTTPClientFactory

    http_client = HTTPClientFactory.create_client(
        settings=settings,
        hook_manager=hook_manager,
    )

    context_data: dict[str, Any] = {
        "detection_service": detection_service,
        "http_client": http_client,
    }

    try:
        oauth_provider = factory.create_auth_provider(context_data)  # type: ignore[arg-type]
    except Exception as e:
        logger.debug(
            "oauth_provider_create_failed", provider=provider, error=str(e), exc_info=e
        )
        return None

    try:
        if not registry.has_provider(oauth_provider.provider_name):
            registry.register_provider(oauth_provider)
    except Exception:
        pass

    return oauth_provider


async def discover_oauth_providers(
    container: ServiceContainer,
) -> dict[str, tuple[str, str]]:
    """Return available OAuth providers discovered via the plugin loader."""
    providers: dict[str, tuple[str, str]] = {}
    try:
        settings = container.get_service(Settings)
        registry, _ = load_plugin_system(settings)
        for name, factory in registry.factories.items():
            from ccproxy.core.plugins.factory import AuthProviderPluginFactory

            if isinstance(factory, AuthProviderPluginFactory):
                if name == "oauth_claude":
                    providers["claude-api"] = ("oauth", "Claude API OAuth")
                elif name == "oauth_codex":
                    providers["codex"] = ("oauth", "OpenAI Codex OAuth")
    except Exception as e:
        logger.debug("discover_oauth_providers_failed", error=str(e), exc_info=e)
    return providers


def get_oauth_provider_choices() -> list[str]:
    """Get list of available OAuth provider names for CLI choices."""
    container = _get_service_container()
    providers = asyncio.run(discover_oauth_providers(container))
    return list(providers.keys())


async def get_oauth_client_for_provider(
    provider: str,
    registry: OAuthRegistry,
    container: ServiceContainer,
) -> Any:
    """Get OAuth client for the specified provider."""
    oauth_provider = await get_oauth_provider_for_name(provider, registry, container)
    if not oauth_provider:
        raise ValueError(f"Provider '{provider}' not found")
    oauth_client = getattr(oauth_provider, "client", None)
    if not oauth_client:
        raise ValueError(f"Provider '{provider}' does not implement OAuth client")
    return oauth_client


async def check_provider_credentials(
    provider: str,
    registry: OAuthRegistry,
    container: ServiceContainer,
) -> dict[str, Any]:
    """Check if provider has valid stored credentials."""
    try:
        oauth_provider = await get_oauth_provider_for_name(
            provider, registry, container
        )
        if not oauth_provider:
            return {
                "has_credentials": False,
                "expired": True,
                "path": None,
                "credentials": None,
            }

        creds = await oauth_provider.load_credentials()
        has_credentials = creds is not None

        return {
            "has_credentials": has_credentials,
            "expired": not has_credentials,
            "path": None,
            "credentials": None,
        }

    except AttributeError as e:
        logger.debug(
            "credentials_check_missing_attribute",
            provider=provider,
            error=str(e),
            exc_info=e,
        )
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
        container = _get_service_container()
        providers = asyncio.run(discover_oauth_providers(container))

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
    """Login to a provider using OAuth authentication."""
    _ensure_logging_configured()
    toolkit = get_rich_toolkit()

    provider = provider.strip().lower()

    toolkit.print(
        f"[bold cyan]OAuth Login - {provider.replace('_', '-').title()}[/bold cyan]",
        centered=True,
    )
    toolkit.print_line()

    toolkit.print(
        "CLI OAuth flow has been removed. Use provider-specific tooling or API tokens.",
        tag="error",
    )
    raise typer.Exit(1)


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
    """Check authentication status and info for specified provider."""
    _ensure_logging_configured()
    toolkit = get_rich_toolkit()

    provider = provider.strip().lower()
    display_name = provider.replace("_", "-").title()

    toolkit.print(
        f"[bold cyan]{display_name} Authentication Status[/bold cyan]",
        centered=True,
    )
    toolkit.print_line()

    try:
        container = _get_service_container()
        registry = OAuthRegistry()
        oauth_provider = asyncio.run(
            get_oauth_provider_for_name(provider, registry, container)
        )
        if not oauth_provider:
            providers = asyncio.run(discover_oauth_providers(container))
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
            try:
                storage = None
                if hasattr(oauth_provider, "get_storage"):
                    with contextlib.suppress(Exception):
                        storage = oauth_provider.get_storage()

                from typing import Any as _Any

                manager: _Any | None = None
                if provider in ("claude-api", "claude_api"):
                    try:
                        from ccproxy.plugins.oauth_claude.manager import (
                            ClaudeApiTokenManager,
                        )

                        if storage is not None:
                            manager = asyncio.run(
                                ClaudeApiTokenManager.create(storage=storage)
                            )
                        else:
                            manager = asyncio.run(ClaudeApiTokenManager.create())
                    except Exception as e:
                        logger.debug("claude_manager_init_failed", error=str(e))
                elif provider == "codex":
                    try:
                        from ccproxy.plugins.oauth_codex.manager import (
                            CodexTokenManager,
                        )

                        if storage is not None:
                            manager = asyncio.run(
                                CodexTokenManager.create(storage=storage)
                            )
                        else:
                            manager = asyncio.run(CodexTokenManager.create())
                    except Exception as e:
                        logger.debug("codex_manager_init_failed", error=str(e))

                if manager is not None:
                    credentials = asyncio.run(manager.load_credentials())
                else:
                    credentials = asyncio.run(oauth_provider.load_credentials())

                if credentials:
                    if provider == "codex":
                        standard_profile = None
                        if hasattr(oauth_provider, "get_standard_profile"):
                            with contextlib.suppress(Exception):
                                standard_profile = asyncio.run(
                                    oauth_provider.get_standard_profile(credentials)
                                )
                        if not standard_profile and hasattr(
                            oauth_provider,
                            "_extract_standard_profile",
                        ):
                            with contextlib.suppress(Exception):
                                standard_profile = (
                                    oauth_provider._extract_standard_profile(
                                        credentials
                                    )
                                )
                        if standard_profile is not None:
                            try:
                                profile_info = standard_profile.model_dump(
                                    exclude={"raw_profile_data"}
                                )
                            except Exception:
                                profile_info = {
                                    "provider": provider,
                                    "authenticated": True,
                                }
                        else:
                            profile_info = {"provider": provider, "authenticated": True}
                    else:
                        quick = None
                        if manager is not None and hasattr(
                            manager,
                            "get_unified_profile_quick",
                        ):
                            with contextlib.suppress(Exception):
                                quick = asyncio.run(manager.get_unified_profile_quick())
                        if (
                            (not quick or quick == {})
                            and detailed
                            and manager is not None
                        ):
                            with contextlib.suppress(Exception):
                                quick = asyncio.run(manager.get_unified_profile())
                        if quick and isinstance(quick, dict) and quick != {}:
                            profile_info = quick
                            try:
                                prov = (
                                    profile_info.get("provider_type")
                                    or profile_info.get("provider")
                                    or ""
                                ).lower()
                                extras = (
                                    profile_info.get("extras")
                                    if isinstance(profile_info.get("extras"), dict)
                                    else None
                                )
                                if (
                                    prov in {"claude-api", "claude_api", "claude"}
                                    and extras
                                ):
                                    account = (
                                        extras.get("account", {})
                                        if isinstance(extras.get("account"), dict)
                                        else {}
                                    )
                                    org = (
                                        extras.get("organization", {})
                                        if isinstance(extras.get("organization"), dict)
                                        else {}
                                    )
                                    if account.get("has_claude_max") is True:
                                        profile_info["subscription_type"] = "max"
                                        profile_info["subscription_status"] = "active"
                                    elif account.get("has_claude_pro") is True:
                                        profile_info["subscription_type"] = "pro"
                                        profile_info["subscription_status"] = "active"
                                    features = {}
                                    if isinstance(account.get("has_claude_max"), bool):
                                        features["claude_max"] = account.get(
                                            "has_claude_max"
                                        )
                                    if isinstance(account.get("has_claude_pro"), bool):
                                        features["claude_pro"] = account.get(
                                            "has_claude_pro"
                                        )
                                    if features:
                                        profile_info["features"] = {
                                            **features,
                                            **(profile_info.get("features") or {}),
                                        }
                                    if org.get("name") and not profile_info.get(
                                        "organization_name"
                                    ):
                                        profile_info["organization_name"] = org.get(
                                            "name"
                                        )
                                    if not profile_info.get("organization_role"):
                                        profile_info["organization_role"] = "member"
                            except Exception:
                                pass
                        else:
                            standard_profile = None
                            if hasattr(oauth_provider, "get_standard_profile"):
                                with contextlib.suppress(Exception):
                                    standard_profile = asyncio.run(
                                        oauth_provider.get_standard_profile(credentials)
                                    )
                            if standard_profile is not None:
                                try:
                                    profile_info = standard_profile.model_dump(
                                        exclude={"raw_profile_data"}
                                    )
                                except Exception:
                                    profile_info = {
                                        "provider": provider,
                                        "authenticated": True,
                                    }
                            else:
                                profile_info = {
                                    "provider": provider,
                                    "authenticated": True,
                                }

                    if profile_info is not None and "provider" not in profile_info:
                        profile_info["provider"] = provider

                    try:
                        prov_dbg = (
                            profile_info.get("provider_type")
                            or profile_info.get("provider")
                            or ""
                        ).lower()
                        missing = []
                        for f in (
                            "subscription_type",
                            "organization_name",
                            "display_name",
                        ):
                            if not profile_info.get(f):
                                missing.append(f)
                        if missing:
                            reasons: list[str] = []
                            qextra = (
                                quick.get("extras") if isinstance(quick, dict) else None
                            )
                            if prov_dbg in {"codex", "openai"}:
                                auth_claims = None
                                if isinstance(qextra, dict):
                                    auth_claims = qextra.get(
                                        "https://api.openai.com/auth"
                                    )
                                if not auth_claims:
                                    reasons.append("missing_openai_auth_claims")
                                else:
                                    if "chatgpt_plan_type" not in auth_claims:
                                        reasons.append("plan_type_not_in_claims")
                                    orgs = (
                                        auth_claims.get("organizations")
                                        if isinstance(auth_claims, dict)
                                        else None
                                    )
                                    if not orgs:
                                        reasons.append("no_organizations_in_claims")
                                if (
                                    hasattr(credentials, "id_token")
                                    and not credentials.id_token
                                ):
                                    reasons.append("no_id_token_available")
                            elif prov_dbg in {"claude", "claude-api", "claude_api"}:
                                if not (
                                    isinstance(qextra, dict) and qextra.get("account")
                                ):
                                    reasons.append("missing_claude_account_extras")
                            if reasons:
                                logger.debug(
                                    "profile_fields_missing",
                                    provider=prov_dbg,
                                    missing_fields=missing,
                                    reasons=reasons,
                                )
                    except Exception:
                        pass

            except Exception as e:
                logger.debug(f"{provider}_status_error", error=str(e), exc_info=e)

        if profile_info:
            console.print("[green]✓[/green] Authenticated with valid credentials")

            if "provider_type" not in profile_info and "provider" in profile_info:
                try:
                    profile_info["provider_type"] = str(
                        profile_info["provider"]
                    ).replace("_", "-")
                except Exception:
                    profile_info["provider_type"] = (
                        str(profile_info["provider"])
                        if profile_info.get("provider")
                        else None
                    )

            _render_profile_table(profile_info, title="Account Information")
            _render_profile_features(profile_info)

            if detailed and credentials:
                token_str = None

                if hasattr(credentials, "access_token"):
                    token_str = str(credentials.access_token)
                elif hasattr(credentials, "claude_ai_oauth"):
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
    """Logout and remove stored credentials for specified provider."""
    _ensure_logging_configured()
    toolkit = get_rich_toolkit()

    provider = provider.strip().lower()

    toolkit.print(f"[bold cyan]{provider.title()} Logout[/bold cyan]", centered=True)
    toolkit.print_line()

    try:
        container = _get_service_container()
        registry = OAuthRegistry()
        oauth_provider = asyncio.run(
            get_oauth_provider_for_name(provider, registry, container)
        )

        if not oauth_provider:
            providers = asyncio.run(discover_oauth_providers(container))
            available = ", ".join(providers.keys()) if providers else "none"
            expected = _expected_plugin_class_name(provider)
            toolkit.print(
                f"Provider '{provider}' not found. Available: {available}. Expected plugin class '{expected}'.",
                tag="error",
            )
            raise typer.Exit(1)

        existing_creds = None
        with contextlib.suppress(Exception):
            existing_creds = asyncio.run(oauth_provider.load_credentials())

        if not existing_creds:
            console.print("[yellow]No credentials found. Already logged out.[/yellow]")
            return

        confirm = typer.confirm(
            "Are you sure you want to logout and remove credentials?"
        )
        if not confirm:
            console.print("Logout cancelled.")
            return

        success = False
        try:
            storage = oauth_provider.get_storage()
            if storage and hasattr(storage, "delete"):
                success = asyncio.run(storage.delete())
            elif storage and hasattr(storage, "clear"):
                success = asyncio.run(storage.clear())
            else:
                success = asyncio.run(oauth_provider.save_credentials(None))
        except Exception as e:
            logger.debug("logout_error", error=str(e), exc_info=e)

        if success:
            toolkit.print(f"Successfully logged out from {provider}!", tag="success")
            console.print("Credentials have been removed.")
        else:
            toolkit.print("Failed to remove credentials", tag="error")
            raise typer.Exit(1)

    except FileNotFoundError:
        toolkit.print("No credentials found to remove.", tag="warning")
    except OSError as e:
        toolkit.print(f"Failed to remove credential files: {e}", tag="error")
        raise typer.Exit(1) from e
    except ImportError as e:
        toolkit.print(f"Failed to import required modules: {e}", tag="error")
        raise typer.Exit(1) from e
    except Exception as e:
        toolkit.print(f"Error during logout: {e}", tag="error")
        raise typer.Exit(1) from e


async def get_oauth_provider_for_name(
    provider: str,
    registry: OAuthRegistry,
    container: ServiceContainer,
) -> Any:
    """Get OAuth provider instance for the specified provider name."""
    existing = registry.get_provider(provider)
    if existing:
        return existing

    provider_instance = await _lazy_register_oauth_provider(
        provider, registry, container
    )
    if provider_instance:
        return provider_instance

    return None
