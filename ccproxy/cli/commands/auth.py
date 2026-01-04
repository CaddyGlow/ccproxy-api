"""Authentication and credential management commands."""

import asyncio
import contextlib
import inspect
import logging
import os
from collections.abc import Coroutine
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, cast

import structlog
import typer
from rich import box
from rich.console import Console
from rich.table import Table

from ccproxy.auth.managers.token_snapshot import TokenSnapshot
from ccproxy.auth.oauth.cli_errors import (
    AuthProviderError,
    AuthTimedOutError,
    AuthUserAbortedError,
    NetworkError,
    PortBindError,
)
from ccproxy.auth.oauth.flows import BrowserFlow, DeviceCodeFlow, ManualCodeFlow
from ccproxy.auth.oauth.registry import FlowType, OAuthRegistry
from ccproxy.cli.helpers import get_rich_toolkit
from ccproxy.config.settings import Settings
from ccproxy.core.logging import bootstrap_cli_logging, get_logger, setup_logging
from ccproxy.core.plugins import load_cli_plugins
from ccproxy.core.plugins.hooks.manager import HookManager
from ccproxy.core.plugins.hooks.registry import HookRegistry
from ccproxy.services.container import ServiceContainer


app = typer.Typer(
    name="auth", help="Authentication and credential management", no_args_is_help=True
)

console = Console()
logger = get_logger(__name__)


# Cache settings and container to avoid repeated config file loading
_cached_settings: Settings | None = None
_cached_container: ServiceContainer | None = None


@contextlib.contextmanager
def _temporary_disable_provider_storage(provider: Any, *, disable: bool) -> Any:
    """Temporarily disable provider/client storage (used for custom credential paths)."""

    if not disable:
        yield
        return

    original_provider_storage = getattr(provider, "storage", None)
    client = getattr(provider, "client", None)
    original_client_storage = getattr(client, "storage", None) if client else None

    try:
        if hasattr(provider, "storage"):
            provider.storage = None
        if client is not None and hasattr(client, "storage"):
            client.storage = None
        yield
    finally:
        if hasattr(provider, "storage"):
            provider.storage = original_provider_storage
        if client is not None and hasattr(client, "storage"):
            client.storage = original_client_storage


def _normalize_credentials_file_option(
    toolkit: Any,
    file_option: Path | None,
    *,
    require_exists: bool,
    create_parent: bool = False,
) -> Path | None:
    """Resolve and validate a user-supplied credential file path."""

    if file_option is None:
        return None

    custom_path = file_option.expanduser()
    try:
        custom_path = custom_path.resolve()
    except FileNotFoundError:
        # If parents do not exist, fall back to absolute path for messaging
        custom_path = custom_path.absolute()

    if custom_path.exists() and custom_path.is_dir():
        toolkit.print(
            f"Target path '{custom_path}' is a directory. Provide a file path.",
            tag="error",
        )
        raise typer.Exit(1)

    if require_exists and not custom_path.exists():
        toolkit.print(
            f"Credential file '{custom_path}' not found.",
            tag="error",
        )
        raise typer.Exit(1)

    if create_parent:
        try:
            custom_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            toolkit.print(
                f"Failed to create directory '{custom_path.parent}': {exc}",
                tag="error",
            )
            raise typer.Exit(1) from exc

    return custom_path


def _get_cached_settings() -> Settings:
    """Get cached settings instance."""
    global _cached_settings
    if _cached_settings is None:
        _cached_settings = Settings.from_config()
    return _cached_settings


def _get_service_container() -> ServiceContainer:
    """Create a service container for the auth commands."""
    global _cached_container
    if _cached_container is None:
        settings = _get_cached_settings()
        _cached_container = ServiceContainer(settings)
    return _cached_container


def _apply_auth_logger_level() -> None:
    """Set logger level from settings without configuring handlers."""
    try:
        settings = _get_cached_settings()
        level_name = settings.logging.level
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


def _token_snapshot_from_credentials(
    credentials: Any, provider: str | None = None
) -> TokenSnapshot | None:
    """Best-effort conversion of provider credentials into a token snapshot.

    Uses the BaseCredentials protocol instead of direct imports to avoid boundary violations.
    """
    from ccproxy.auth.models.credentials import BaseCredentials

    # Check if credentials follow the BaseCredentials protocol
    if not isinstance(credentials, BaseCredentials):
        # If not following the protocol, try to extract basic info using duck typing
        return _extract_token_snapshot_duck_typing(credentials, provider)

    # Use the protocol methods
    try:
        data = credentials.to_dict()
        return _build_token_snapshot_from_dict(data, provider)
    except Exception:
        return None


def _extract_token_snapshot_duck_typing(
    credentials: Any, provider: str | None = None
) -> TokenSnapshot | None:
    """Extract token snapshot using duck typing for non-protocol credentials."""
    if not credentials:
        return None

    # Generic duck-typing approach - look for common attributes
    access_token: str | None = None
    refresh_token: str | None = None
    expires_at: datetime | None = None
    account_id: str | None = None
    extras: dict[str, Any] = {}

    # Try to extract access token from various possible attributes
    for attr in ["access_token", "token"]:
        if hasattr(credentials, attr):
            token_obj = getattr(credentials, attr)
            if token_obj:
                if hasattr(token_obj, "get_secret_value"):
                    access_token = token_obj.get_secret_value()
                elif isinstance(token_obj, str):
                    access_token = token_obj
                break

    # Try to extract refresh token
    if hasattr(credentials, "refresh_token"):
        refresh_obj = credentials.refresh_token
        if refresh_obj and hasattr(refresh_obj, "get_secret_value"):
            refresh_token = refresh_obj.get_secret_value()
        elif isinstance(refresh_obj, str):
            refresh_token = refresh_obj

    # Try to extract expiration
    for attr in ["expires_at", "expires_at_datetime", "expiry"]:
        if hasattr(credentials, attr):
            expires_obj = getattr(credentials, attr)
            if isinstance(expires_obj, datetime):
                expires_at = expires_obj
                break

    # Try to extract account ID
    for attr in ["account_id", "user_id", "id"]:
        if hasattr(credentials, attr):
            id_obj = getattr(credentials, attr)
            if isinstance(id_obj, str):
                account_id = id_obj
                break

    if not access_token:
        return None

    return TokenSnapshot(
        provider=provider or "unknown",
        account_id=account_id,
        access_token=access_token,
        refresh_token=refresh_token,
        expires_at=expires_at,
        extras={},
    )


def _build_token_snapshot_from_dict(
    data: dict[str, Any], provider: str | None = None
) -> TokenSnapshot | None:
    """Build token snapshot from dictionary data."""
    if not data:
        return None

    def _unwrap_secret(value: Any) -> str | None:
        """Return plain string from SecretStr-like values."""
        if value is None:
            return None
        if hasattr(value, "get_secret_value"):
            try:
                result = value.get_secret_value()
                return str(result) if result is not None else None
            except Exception:
                return None
        if isinstance(value, str):
            return value or None
        return None

    def _coerce_datetime(value: Any) -> datetime | None:
        """Convert supported values into timezone-aware datetime objects."""
        if value is None:
            return None
        if isinstance(value, datetime):
            return value if value.tzinfo else value.replace(tzinfo=UTC)
        if isinstance(value, str):
            try:
                parsed = datetime.fromisoformat(value)
            except ValueError:
                return None
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
        if isinstance(value, int | float):
            timestamp = float(value)
            # Treat very large integers as millisecond timestamps
            if timestamp > 1e11:
                timestamp /= 1000
            try:
                return datetime.fromtimestamp(timestamp, tz=UTC)
            except (OSError, ValueError):
                return None
        return None

    provider_value = provider or data.get("provider") or "unknown"
    provider_normalized = provider_value.replace("_", "-")

    extras: dict[str, Any] = dict(data.get("extras", {}))
    scopes: tuple[str, ...] = tuple(
        str(scope) for scope in data.get("scopes", []) if scope
    )

    account_id: str | None = _unwrap_secret(data.get("account_id"))
    access_token: str | None = _unwrap_secret(data.get("access_token"))
    refresh_token: str | None = _unwrap_secret(data.get("refresh_token"))
    expires_at: datetime | None = _coerce_datetime(data.get("expires_at"))

    claude_data = data.get("claudeAiOauth") or data.get("claude_ai_oauth")
    if isinstance(claude_data, dict):
        provider_normalized = "claude-api"
        access_token = access_token or _unwrap_secret(
            claude_data.get("accessToken") or claude_data.get("access_token")
        )
        refresh_token = refresh_token or _unwrap_secret(
            claude_data.get("refreshToken") or claude_data.get("refresh_token")
        )
        expires_at = expires_at or _coerce_datetime(
            claude_data.get("expiresAt") or claude_data.get("expires_at")
        )
        scopes = tuple(str(scope) for scope in claude_data.get("scopes", []) if scope)
        subscription = claude_data.get("subscriptionType") or claude_data.get(
            "subscription_type"
        )
        if subscription:
            extras.setdefault("subscription_type", subscription)

    tokens_data = data.get("tokens")
    if isinstance(tokens_data, dict):
        provider_normalized = "codex"
        access_token = access_token or _unwrap_secret(tokens_data.get("access_token"))
        refresh_token = refresh_token or _unwrap_secret(
            tokens_data.get("refresh_token")
        )
        account_id = account_id or tokens_data.get("account_id")
        if "id_token_present" not in extras:
            extras["id_token_present"] = bool(tokens_data.get("id_token"))

    oauth_token_data = data.get("oauth_token") or data.get("oauthToken")
    copilot_token_data = data.get("copilot_token") or data.get("copilotToken")
    if isinstance(oauth_token_data, dict) or isinstance(copilot_token_data, dict):
        provider_normalized = "copilot"

    if isinstance(copilot_token_data, dict):
        token_value = _unwrap_secret(copilot_token_data.get("token"))
        if token_value:
            access_token = token_value
        expires_at = (
            _coerce_datetime(copilot_token_data.get("expires_at")) or expires_at
        )
        extras.setdefault("has_copilot_token", True)

    if isinstance(oauth_token_data, dict):
        access_token = access_token or _unwrap_secret(
            oauth_token_data.get("access_token")
        )
        refresh_token = refresh_token or _unwrap_secret(
            oauth_token_data.get("refresh_token")
        )
        scope_field = oauth_token_data.get("scope") or ""
        if scope_field and not scopes:
            scopes = tuple(
                scope
                for scope in (item.strip() for item in str(scope_field).split(" "))
                if scope
            )
        if not extras.get("has_copilot_token"):
            extras["has_copilot_token"] = False
        if not expires_at:
            created_at = oauth_token_data.get("created_at")
            expires_in = oauth_token_data.get("expires_in")
            if isinstance(created_at, int | float) and isinstance(
                expires_in, int | float
            ):
                expires_at = _coerce_datetime(created_at + expires_in)

    if provider_normalized == "copilot":
        if "refresh_token_present" not in extras:
            extras["refresh_token_present"] = bool(refresh_token)
        extras.setdefault("id_token_present", bool(extras.get("has_copilot_token")))

    return TokenSnapshot(
        provider=provider_normalized,
        account_id=account_id,
        access_token=access_token,
        refresh_token=refresh_token,
        expires_at=expires_at,
        scopes=scopes,
        extras=extras,
    )


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


def _await_if_needed(value: Any) -> Any:
    """Await coroutine values in synchronous CLI context."""
    if inspect.isawaitable(value):
        return asyncio.run(cast(Coroutine[Any, Any, Any], value))
    return value


def _resolve_token_manager_from_registry(
    provider: str, oauth_provider: Any, container: ServiceContainer
) -> Any | None:
    """Try fetching an auth manager from the global registry."""
    try:
        registry = container.get_auth_manager_registry()
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("auth_manager_registry_unavailable", error=str(exc))
        return None

    candidates: list[str] = []

    def _push(name: str | None) -> None:
        if not name:
            return
        normalized = name.strip()
        if not normalized:
            return
        for variant in {
            normalized,
            normalized.replace("-", "_"),
        }:  # normalize hyphen/underscore
            if variant not in candidates:
                candidates.append(variant)

    _push(provider)
    _push(_provider_plugin_name(provider))
    _push(getattr(oauth_provider, "provider_name", None))

    try:
        info = oauth_provider.get_provider_info()
        _push(getattr(info, "plugin_name", None))
    except Exception as exc:  # pragma: no cover - defensive logging only
        logger.debug("provider_info_lookup_failed", error=str(exc))

    for candidate in candidates:
        try:
            manager = asyncio.run(registry.get(candidate))
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug(
                "auth_manager_registry_get_failed", name=candidate, error=str(exc)
            )
            continue
        if manager:
            return manager

    return None


def _resolve_token_manager(
    provider: str, oauth_provider: Any, container: ServiceContainer
) -> Any | None:
    """Resolve token manager via registry or provider helpers."""
    manager = _resolve_token_manager_from_registry(provider, oauth_provider, container)
    if manager:
        return manager

    if hasattr(oauth_provider, "get_token_manager"):
        try:
            candidate = oauth_provider.get_token_manager()
            manager = _await_if_needed(candidate)
            if manager:
                return manager
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.debug("get_token_manager_failed", error=str(exc))

    if hasattr(oauth_provider, "create_token_manager"):
        try:
            candidate = oauth_provider.create_token_manager()
            manager = _await_if_needed(candidate)
            if manager:
                return manager
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.debug("create_token_manager_failed", error=str(exc))

    return None


def _format_seconds(seconds: int | None) -> str:
    """Format seconds into a short human-readable duration."""
    if seconds is None:
        return "Unknown"
    if seconds <= 0:
        return "Expired"

    remaining = int(seconds)
    parts: list[str] = []
    for label, divisor in (("d", 86_400), ("h", 3_600), ("m", 60)):
        value, remaining = divmod(remaining, divisor)
        if value:
            parts.append(f"{value}{label}")
        if len(parts) == 2:
            break

    if remaining and len(parts) < 2:
        parts.append(f"{remaining}s")

    return " ".join(parts) if parts else "<1s"


async def _lazy_register_oauth_provider(
    provider: str,
    registry: OAuthRegistry,
    container: ServiceContainer,
) -> Any | None:
    """Initialize filtered CLI plugin system and ensure provider is registered.

    This bootstraps the hook system and initializes only CLI-safe plugins plus
    the specific auth provider needed. This avoids DuckDB locks, task manager
    errors, and other side effects from heavy provider plugins.
    """
    settings = container.get_service(Settings)

    # Respect global plugin enablement flag
    if not getattr(settings, "enable_plugins", True):
        return None

    # Load only CLI-safe plugins + the specific auth provider needed
    plugin_registry = load_cli_plugins(settings, auth_provider=provider)

    # Create hook system for CLI HTTP flows
    hook_registry = HookRegistry()
    hook_manager = HookManager(hook_registry)
    # Make HookManager available to any services resolved from the container
    with contextlib.suppress(Exception):
        container.register_service(HookManager, instance=hook_manager)

    # Provide core services needed by plugins at runtime

    try:
        # Initialize all plugins; auth providers will register to oauth_registry
        import asyncio as _asyncio

        if _asyncio.get_event_loop().is_running():
            # In practice, we're already in async context; just await directly
            await plugin_registry.initialize_all(container)
        else:  # pragma: no cover - defensive path
            _asyncio.run(plugin_registry.initialize_all(container))
    except Exception as e:
        logger.debug(
            "plugin_initialization_failed_cli",
            error=str(e),
            exc_info=e,
            category="auth",
        )

    # Normalize provider key and return the registered provider instance
    def _norm(p: str) -> str:
        key = p.strip().lower().replace("_", "-")
        if key in {"claude", "claude-api"}:
            return "claude-api"
        if key in {"codex", "openai", "openai-api"}:
            return "codex"
        return key

    try:
        return registry.get(_norm(provider))
    except Exception:
        return None


async def discover_oauth_providers(
    container: ServiceContainer,
) -> dict[str, tuple[str, str]]:
    """Return available OAuth providers discovered via the plugin loader."""
    providers: dict[str, tuple[str, str]] = {}
    try:
        settings = container.get_service(Settings)
        # For discovery, we can load all plugins temporarily since we don't initialize them
        from ccproxy.core.plugins import load_plugin_system

        registry, _ = load_plugin_system(settings)
        for name, factory in registry.factories.items():
            from ccproxy.core.plugins import AuthProviderPluginFactory

            if isinstance(factory, AuthProviderPluginFactory):
                if name == "oauth_claude":
                    providers["claude-api"] = ("oauth", "Claude API OAuth")
                elif name == "oauth_codex":
                    providers["codex"] = ("oauth", "OpenAI Codex OAuth")
                elif name == "copilot":
                    providers["copilot"] = ("oauth", "GitHub Copilot OAuth")
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
        typer.Argument(
            help="Provider to authenticate with (claude-api, codex, copilot)"
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
    output_file: Annotated[
        Path | None,
        typer.Option(
            "--file",
            help="Write credentials to this path instead of the default storage",
        ),
    ] = None,
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            help="Overwrite existing credential file when using --file",
        ),
    ] = False,
) -> None:
    """Login to a provider using OAuth authentication."""
    _ensure_logging_configured()
    # Capture plugin-injected CLI args for potential use by auth providers
    try:
        from ccproxy.cli.helpers import get_plugin_cli_args

        _ = get_plugin_cli_args()
        # Currently not used directly here, but available to providers
    except Exception:
        pass
    toolkit = get_rich_toolkit()

    if force and output_file is None:
        toolkit.print("--force can only be used together with --file", tag="error")
        raise typer.Exit(1)

    custom_path: Path | None = None
    if output_file is not None:
        custom_path = output_file.expanduser()
        try:
            custom_path = custom_path.resolve()
        except FileNotFoundError:
            # Path.resolve() on some platforms raises when parents missing; fallback to absolute()
            custom_path = custom_path.absolute()

        if custom_path.exists() and custom_path.is_dir():
            toolkit.print(
                f"Target path '{custom_path}' is a directory. Provide a file path.",
                tag="error",
            )
            raise typer.Exit(1)

        if custom_path.exists() and not force:
            toolkit.print(
                f"Credential file '{custom_path}' already exists. Use --force to overwrite.",
                tag="error",
            )
            raise typer.Exit(1)

        try:
            custom_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            toolkit.print(
                f"Failed to create directory '{custom_path.parent}': {exc}",
                tag="error",
            )
            raise typer.Exit(1)

    provider = provider.strip().lower()
    display_name = provider.replace("_", "-").title()

    toolkit.print(
        f"[bold cyan]OAuth Login - {display_name}[/bold cyan]",
        centered=True,
    )
    toolkit.print_line()

    custom_path_str = str(custom_path) if custom_path else None

    try:
        container = _get_service_container()
        registry = container.get_oauth_registry()
        oauth_provider = asyncio.run(
            get_oauth_provider_for_name(provider, registry, container)
        )

        if not oauth_provider:
            providers = asyncio.run(discover_oauth_providers(container))
            available = ", ".join(providers.keys()) if providers else "none"
            toolkit.print(
                f"Provider '{provider}' not found. Available: {available}",
                tag="error",
            )
            raise typer.Exit(1)

        # Get CLI configuration from provider
        cli_config = oauth_provider.cli

        # Flow engine selection with fallback logic
        flow_engine: ManualCodeFlow | DeviceCodeFlow | BrowserFlow
        try:
            with _temporary_disable_provider_storage(
                oauth_provider, disable=custom_path is not None
            ):
                if manual:
                    # Manual mode requested
                    if not cli_config.supports_manual_code:
                        raise AuthProviderError(
                            f"Provider '{provider}' doesn't support manual code entry"
                        )
                    flow_engine = ManualCodeFlow()
                    success = asyncio.run(
                        flow_engine.run(oauth_provider, save_path=custom_path_str)
                    )

                elif (
                    cli_config.preferred_flow == FlowType.device
                    and cli_config.supports_device_flow
                ):
                    # Device flow preferred and supported
                    flow_engine = DeviceCodeFlow()
                    success = asyncio.run(
                        flow_engine.run(oauth_provider, save_path=custom_path_str)
                    )

                else:
                    # Browser flow (default)
                    flow_engine = BrowserFlow()
                    success = asyncio.run(
                        flow_engine.run(
                            oauth_provider,
                            no_browser=no_browser,
                            save_path=custom_path_str,
                        )
                    )

        except PortBindError as e:
            # Port binding failed - offer manual fallback
            if cli_config.supports_manual_code:
                console.print(
                    "[yellow]Port binding failed. Falling back to manual mode.[/yellow]"
                )
                with _temporary_disable_provider_storage(
                    oauth_provider, disable=custom_path is not None
                ):
                    flow_engine = ManualCodeFlow()
                    success = asyncio.run(
                        flow_engine.run(oauth_provider, save_path=custom_path_str)
                    )
            else:
                console.print(
                    f"[red]Port {cli_config.callback_port} unavailable and manual mode not supported[/red]"
                )
                raise typer.Exit(1) from e

        except AuthTimedOutError:
            console.print("[red]Authentication timed out[/red]")
            raise typer.Exit(1)

        except AuthUserAbortedError:
            console.print("[yellow]Authentication cancelled by user[/yellow]")
            raise typer.Exit(1)

        except AuthProviderError as e:
            console.print(f"[red]Authentication failed: {e}[/red]")
            raise typer.Exit(1) from e

        except NetworkError as e:
            console.print(f"[red]Network error: {e}[/red]")
            raise typer.Exit(1) from e

        if success:
            console.print("[green]✓[/green] Authentication successful!")
            if custom_path:
                console.print(
                    f"[dim]Credentials saved to {custom_path}[/dim]",
                )
        else:
            console.print("[red]✗[/red] Authentication failed")
            raise typer.Exit(1)

    except KeyboardInterrupt:
        console.print("\n[yellow]Login cancelled by user.[/yellow]")
        raise typer.Exit(2) from None
    except ImportError as e:
        toolkit.print(f"Plugin import error: {e}", tag="error")
        raise typer.Exit(1) from e
    except typer.Exit:
        # Re-raise typer exits
        raise
    except Exception as e:
        toolkit.print(f"Error during login: {e}", tag="error")
        logger.error("login_command_error", error=str(e), exc_info=e)
        raise typer.Exit(1) from e


def _refresh_provider_tokens(provider: str, custom_path: Path | None = None) -> None:
    """Shared implementation for refresh/renew commands."""
    toolkit = get_rich_toolkit()
    provider_key = provider.strip().lower()
    display_name = provider_key.replace("_", "-").title()

    toolkit.print(
        f"[bold cyan]{display_name} Token Refresh[/bold cyan]",
        centered=True,
    )
    toolkit.print_line()

    credential_path = _normalize_credentials_file_option(
        toolkit, custom_path, require_exists=True
    )
    load_kwargs: dict[str, Any] = {}
    save_kwargs: dict[str, Any] = {}
    if credential_path is not None:
        load_kwargs["custom_path"] = credential_path
        save_kwargs["custom_path"] = credential_path

    try:
        container = _get_service_container()
        registry = container.get_oauth_registry()
        oauth_provider = asyncio.run(
            get_oauth_provider_for_name(provider_key, registry, container)
        )

        if not oauth_provider:
            providers = asyncio.run(discover_oauth_providers(container))
            available = ", ".join(providers.keys()) if providers else "none"
            toolkit.print(
                f"Provider '{provider_key}' not found. Available: {available}",
                tag="error",
            )
            raise typer.Exit(1)

        if not bool(getattr(oauth_provider, "supports_refresh", False)):
            toolkit.print(
                f"Provider '{provider_key}' does not support token refresh.",
                tag="warning",
            )
            raise typer.Exit(1)

        credentials = asyncio.run(oauth_provider.load_credentials(**load_kwargs))
        if not credentials:
            toolkit.print(
                (
                    f"No credentials found at '{credential_path}'."
                    if credential_path
                    else "No credentials found. Run 'ccproxy auth login' first."
                ),
                tag="warning",
            )
            raise typer.Exit(1)

        snapshot = _token_snapshot_from_credentials(credentials, provider_key)

        manager = None
        if credential_path is None:
            manager = _resolve_token_manager(provider_key, oauth_provider, container)

        refreshed_credentials: Any | None = None
        try:
            if (
                credential_path is None
                and manager
                and hasattr(manager, "refresh_token")
            ):
                refreshed_credentials = asyncio.run(manager.refresh_token())
            else:
                refresh_token = snapshot.refresh_token if snapshot else None
                if not refresh_token:
                    toolkit.print(
                        "Stored credentials do not include a refresh token; "
                        "re-authentication is required.",
                        tag="warning",
                    )
                    raise typer.Exit(1)

                with _temporary_disable_provider_storage(
                    oauth_provider, disable=credential_path is not None
                ):
                    refreshed_credentials = asyncio.run(
                        oauth_provider.refresh_access_token(refresh_token)
                    )
                if credential_path and refreshed_credentials:
                    saved = asyncio.run(
                        oauth_provider.save_credentials(
                            refreshed_credentials, **save_kwargs
                        )
                    )
                    if not saved:
                        toolkit.print(
                            f"Refreshed credentials could not be saved to '{credential_path}'.",
                            tag="warning",
                        )
        except Exception as exc:
            toolkit.print(f"Token refresh failed: {exc}", tag="error")
            logger.error(
                "token_refresh_failed",
                provider=provider_key,
                error=str(exc),
                exc_info=exc,
            )
            raise typer.Exit(1) from exc

        if refreshed_credentials is None:
            with contextlib.suppress(Exception):
                refreshed_credentials = asyncio.run(
                    oauth_provider.load_credentials(**load_kwargs)
                )
            if (
                not refreshed_credentials
                and manager
                and hasattr(manager, "load_credentials")
            ):
                with contextlib.suppress(Exception):
                    refreshed_credentials = asyncio.run(manager.load_credentials())

        refreshed_snapshot = None
        if refreshed_credentials:
            refreshed_snapshot = _token_snapshot_from_credentials(
                refreshed_credentials, provider_key
            )

        if not refreshed_snapshot and snapshot:
            refreshed_snapshot = snapshot

        if not refreshed_snapshot:
            toolkit.print(
                "Token refresh completed but updated credentials could not be loaded. "
                "Check logs for details.",
                tag="warning",
            )
            return

        account_display = refreshed_snapshot.account_id or "—"
        expires_at = (
            refreshed_snapshot.expires_at.isoformat()
            if refreshed_snapshot.expires_at
            else "Unknown"
        )
        expires_in = _format_seconds(refreshed_snapshot.expires_in_seconds())
        access_preview = refreshed_snapshot.access_token_preview() or "(hidden)"
        refresh_preview = (
            refreshed_snapshot.refresh_token_preview()
            if refreshed_snapshot.refresh_token
            else None
        )

        toolkit.print("Tokens refreshed successfully", tag="success")

        summary = Table(show_header=False, box=box.SIMPLE)
        summary.add_column("Field", style="bold")
        summary.add_column("Value")
        summary.add_row("Account", account_display)
        summary.add_row("Expires At", expires_at)
        summary.add_row("Expires In", expires_in)
        summary.add_row("Access Token", access_preview)
        if refresh_preview:
            summary.add_row("Refresh Token", refresh_preview)
        if refreshed_snapshot.scopes:
            summary.add_row("Scopes", ", ".join(refreshed_snapshot.scopes))

        console.print(summary)

    except typer.Exit:
        raise
    except Exception as exc:
        toolkit.print(f"Unexpected error during refresh: {exc}", tag="error")
        logger.error(
            "refresh_command_error", provider=provider_key, error=str(exc), exc_info=exc
        )
        raise typer.Exit(1) from exc


@app.command(name="refresh")
def refresh_command(
    provider: Annotated[
        str,
        typer.Argument(help="Provider to refresh (claude-api, codex, copilot)"),
    ],
    credential_file: Annotated[
        Path | None,
        typer.Option(
            "--file",
            help=(
                "Refresh credentials stored at this path instead of the default storage"
            ),
        ),
    ] = None,
) -> None:
    """Refresh stored credentials using the provider's refresh token."""
    _ensure_logging_configured()
    _refresh_provider_tokens(provider, credential_file)


@app.command(name="renew")
def renew_command(
    provider: Annotated[
        str,
        typer.Argument(help="Alias for refresh command"),
    ],
    credential_file: Annotated[
        Path | None,
        typer.Option(
            "--file",
            help=(
                "Refresh credentials stored at this path instead of the default storage"
            ),
        ),
    ] = None,
) -> None:
    """Alias for refresh."""
    _ensure_logging_configured()
    _refresh_provider_tokens(provider, credential_file)


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
    credential_file: Annotated[
        Path | None,
        typer.Option(
            "--file",
            help=("Read credentials from this path instead of the default storage"),
        ),
    ] = None,
) -> None:
    """Check authentication status and info for specified provider."""
    _ensure_logging_configured()
    toolkit = get_rich_toolkit()

    credential_path = _normalize_credentials_file_option(
        toolkit, credential_file, require_exists=False
    )
    credential_missing = bool(credential_path and not credential_path.exists())
    load_kwargs: dict[str, Any] = {}
    if credential_path is not None:
        load_kwargs["custom_path"] = credential_path

    provider = provider.strip().lower()
    display_name = provider.replace("_", "-").title()

    toolkit.print(
        f"[bold cyan]{display_name} Authentication Status[/bold cyan]",
        centered=True,
    )
    toolkit.print_line()

    try:
        container = _get_service_container()
        registry = container.get_oauth_registry()
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
        snapshot: TokenSnapshot | None = None

        if oauth_provider:
            try:
                # Delegate to provider; providers may internally use their managers
                credentials = asyncio.run(
                    oauth_provider.load_credentials(**load_kwargs)
                )

                if credential_missing and not credentials:
                    toolkit.print(
                        f"Credential file '{credential_path}' not found.",
                        tag="warning",
                    )

                # Optionally obtain a token manager via provider API (if exposed)
                manager = None
                if credential_path is None:
                    try:
                        if hasattr(oauth_provider, "create_token_manager"):
                            manager = asyncio.run(oauth_provider.create_token_manager())
                        elif hasattr(oauth_provider, "get_token_manager"):
                            mgr = oauth_provider.get_token_manager()  # may be sync
                            # If coroutine, run it; else use directly
                            if hasattr(mgr, "__await__"):
                                manager = asyncio.run(mgr)
                            else:
                                manager = mgr
                    except Exception as e:
                        logger.debug("token_manager_unavailable", error=str(e))

                if manager and hasattr(manager, "get_token_snapshot"):
                    with contextlib.suppress(Exception):
                        result = manager.get_token_snapshot()
                        if asyncio.iscoroutine(result):
                            snapshot = asyncio.run(result)
                        else:
                            snapshot = cast(TokenSnapshot | None, result)

                if not snapshot and credentials:
                    snapshot = _token_snapshot_from_credentials(credentials, provider)

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
                        # Prefer provider-supplied quick profile methods if available
                        if credential_path is None and hasattr(
                            oauth_provider, "get_unified_profile_quick"
                        ):
                            with contextlib.suppress(Exception):
                                quick = asyncio.run(
                                    oauth_provider.get_unified_profile_quick()
                                )
                        if (
                            credential_path is None
                            and (not quick or quick == {})
                            and hasattr(oauth_provider, "get_unified_profile")
                        ):
                            with contextlib.suppress(Exception):
                                quick = asyncio.run(
                                    oauth_provider.get_unified_profile()
                                )
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
                                has_id_token = bool(
                                    snapshot and snapshot.extras.get("id_token_present")
                                )
                                if not has_id_token:
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

        token_snapshot = snapshot
        if not token_snapshot and credentials:
            token_snapshot = _token_snapshot_from_credentials(credentials, provider)

        if token_snapshot:
            # Ensure we surface token metadata in the rendered profile table
            if not profile_info:
                profile_info = {
                    "provider_type": token_snapshot.provider or provider,
                    "authenticated": True,
                }

            if token_snapshot.expires_at:
                profile_info["token_expires_at"] = token_snapshot.expires_at

            profile_info["has_refresh_token"] = token_snapshot.has_refresh_token()
            profile_info["has_access_token"] = token_snapshot.has_access_token()

            has_id_token = bool(
                token_snapshot.extras.get("id_token_present")
                or token_snapshot.extras.get("has_id_token")
            )
            if not has_id_token and credentials and hasattr(credentials, "id_token"):
                with contextlib.suppress(Exception):
                    has_id_token = bool(credentials.id_token)
            profile_info["has_id_token"] = has_id_token

            if token_snapshot.scopes and not profile_info.get("scopes"):
                profile_info["scopes"] = list(token_snapshot.scopes)

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

            if detailed and token_snapshot:
                preview = token_snapshot.access_token_preview()
                if preview:
                    console.print(f"\n  Token: [dim]{preview}[/dim]")
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
        registry = container.get_oauth_registry()
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
    existing = registry.get(provider)
    if existing:
        return existing

    provider_instance = await _lazy_register_oauth_provider(
        provider, registry, container
    )
    if provider_instance:
        return provider_instance

    return None
