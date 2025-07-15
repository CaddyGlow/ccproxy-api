"""Authentication and credential management commands."""

import asyncio
import json
from datetime import UTC, datetime, timezone
from pathlib import Path
from typing import Optional

import typer
from rich import box
from rich.console import Console
from rich.table import Table
from structlog import get_logger

from ccproxy.auth.models import ValidationResult
from ccproxy.cli.helpers import get_rich_toolkit
from ccproxy.config.settings import get_settings
from ccproxy.core.async_utils import get_claude_docker_home_dir
from ccproxy.services.credentials import CredentialsManager


app = typer.Typer(name="auth", help="Authentication and credential management")

console = Console()
logger = get_logger(__name__)


def get_credentials_manager(
    custom_paths: list[Path] | None = None,
) -> CredentialsManager:
    """Get a CredentialsManager instance with custom paths if provided."""
    if custom_paths:
        # Get base settings and update storage paths
        settings = get_settings()
        settings.auth.storage.storage_paths = custom_paths
        return CredentialsManager(config=settings.auth)
    else:
        # Use default settings
        settings = get_settings()
        return CredentialsManager(config=settings.auth)


def get_docker_credential_paths() -> list[Path]:
    """Get credential file paths for Docker environment."""
    docker_home = Path(get_claude_docker_home_dir())
    return [
        docker_home / ".claude" / ".credentials.json",
        docker_home / ".config" / "claude" / ".credentials.json",
        Path(".credentials.json"),
    ]


@app.command(name="validate")
def validate_credentials(
    docker: bool = typer.Option(
        False,
        "--docker",
        help="Use Docker credential paths (from get_claude_docker_home_dir())",
    ),
    credential_file: str | None = typer.Option(
        None,
        "--credential-file",
        help="Path to specific credential file to validate",
    ),
) -> None:
    """Validate Claude CLI credentials.

    Checks for valid Claude credentials in standard locations:
    - ~/.claude/credentials.json
    - ~/.config/claude/credentials.json

    With --docker flag, checks Docker credential paths:
    - {docker_home}/.claude/credentials.json
    - {docker_home}/.config/claude/credentials.json

    With --credential-file, validates the specified file directly.

    Examples:
        ccproxy auth validate
        ccproxy auth validate --docker
        ccproxy auth validate --credential-file /path/to/credentials.json
    """
    toolkit = get_rich_toolkit()
    toolkit.print("[bold cyan]Claude Credentials Validation[/bold cyan]", centered=True)
    toolkit.print_line()

    try:
        # Get credential paths based on options
        custom_paths = None
        if credential_file:
            custom_paths = [Path(credential_file)]
        elif docker:
            custom_paths = get_docker_credential_paths()

        # Validate credentials
        manager = get_credentials_manager(custom_paths)
        validation_result = asyncio.run(manager.validate())

        if validation_result.valid:
            # Create a status table
            table = Table(
                show_header=True,
                header_style="bold cyan",
                box=box.ROUNDED,
                title="Credential Status",
                title_style="bold white",
            )
            table.add_column("Property", style="cyan")
            table.add_column("Value", style="white")

            # Status
            status = "Valid" if not validation_result.expired else "Expired"
            status_style = "green" if not validation_result.expired else "red"
            table.add_row("Status", f"[{status_style}]{status}[/{status_style}]")

            # Path
            if validation_result.path:
                table.add_row("Location", f"[dim]{validation_result.path}[/dim]")

            # Subscription type
            if validation_result.credentials:
                sub_type = (
                    validation_result.credentials.claude_ai_oauth.subscription_type
                    or "Unknown"
                )
                table.add_row("Subscription", f"[bold]{sub_type}[/bold]")

                # Expiration
                oauth_token = validation_result.credentials.claude_ai_oauth
                exp_dt = oauth_token.expires_at_datetime
                now = datetime.now(UTC)
                time_diff = exp_dt - now

                if time_diff.total_seconds() > 0:
                    days = time_diff.days
                    hours = time_diff.seconds // 3600
                    exp_str = f"{exp_dt.strftime('%Y-%m-%d %H:%M:%S UTC')} ({days}d {hours}h remaining)"
                else:
                    exp_str = f"{exp_dt.strftime('%Y-%m-%d %H:%M:%S UTC')} [red](Expired)[/red]"

                table.add_row("Expires", exp_str)

                # Scopes
                scopes = oauth_token.scopes
                if scopes:
                    table.add_row("Scopes", ", ".join(str(s) for s in scopes))

            console.print(table)

            # Success message
            if not validation_result.expired:
                toolkit.print(
                    "[green]✓[/green] Valid Claude credentials found", tag="success"
                )
            else:
                toolkit.print(
                    "[yellow]![/yellow] Claude credentials found but expired",
                    tag="warning",
                )
                toolkit.print(
                    "\nPlease refresh your credentials by logging into Claude CLI",
                    tag="info",
                )

        else:
            # No valid credentials
            toolkit.print("[red]✗[/red] No credentials file found", tag="error")

            console.print("\n[dim]To authenticate with Claude CLI, run:[/dim]")
            console.print("[cyan]claude login[/cyan]")

    except Exception as e:
        toolkit.print(f"Error validating credentials: {e}", tag="error")
        raise typer.Exit(1) from e


@app.command(name="info")
def credential_info(
    docker: bool = typer.Option(
        False,
        "--docker",
        help="Use Docker credential paths (from get_claude_docker_home_dir())",
    ),
    credential_file: str | None = typer.Option(
        None,
        "--credential-file",
        help="Path to specific credential file to display info for",
    ),
) -> None:
    """Display detailed credential information.

    Shows all available information about Claude credentials including
    file location, token details, and subscription information.

    Examples:
        ccproxy auth info
        ccproxy auth info --docker
        ccproxy auth info --credential-file /path/to/credentials.json
    """
    toolkit = get_rich_toolkit()
    toolkit.print("[bold cyan]Claude Credential Information[/bold cyan]", centered=True)
    toolkit.print_line()

    try:
        # Get credential paths based on options
        custom_paths = None
        if credential_file:
            custom_paths = [Path(credential_file)]
        elif docker:
            custom_paths = get_docker_credential_paths()

        # Get credentials manager and try to load credentials
        manager = get_credentials_manager(custom_paths)
        credentials = asyncio.run(manager.load())

        if not credentials:
            toolkit.print("No credential file found", tag="error")
            console.print("\n[dim]Expected locations:[/dim]")
            for path in manager.config.storage.storage_paths:
                console.print(f"  - {path}")
            raise typer.Exit(1)

        # Display account section
        console.print("\n[bold]Account[/bold]")
        oauth = credentials.claude_ai_oauth

        # Login method based on subscription type
        login_method = "Claude Account"
        if oauth.subscription_type:
            login_method = f"Claude {oauth.subscription_type.title()} Account"
        console.print(f"  L Login Method: {login_method}")

        # Try to load saved account profile first
        profile = asyncio.run(manager.get_account_profile())

        if profile:
            # Display saved account data
            if profile.organization:
                console.print(f"  L Organization: {profile.organization.name}")
                if profile.organization.organization_type:
                    console.print(
                        f"  L Organization Type: {profile.organization.organization_type}"
                    )
                if profile.organization.billing_type:
                    console.print(
                        f"  L Billing Type: {profile.organization.billing_type}"
                    )
                if profile.organization.rate_limit_tier:
                    console.print(
                        f"  L Rate Limit Tier: {profile.organization.rate_limit_tier}"
                    )
            else:
                console.print("  L Organization: [dim]Not available[/dim]")

            if profile.account:
                console.print(f"  L Email: {profile.account.email}")
                if profile.account.full_name:
                    console.print(f"  L Full Name: {profile.account.full_name}")
                if profile.account.display_name:
                    console.print(f"  L Display Name: {profile.account.display_name}")
                console.print(
                    f"  L Has Claude Pro: {'Yes' if profile.account.has_claude_pro else 'No'}"
                )
                console.print(
                    f"  L Has Claude Max: {'Yes' if profile.account.has_claude_max else 'No'}"
                )
            else:
                console.print("  L Email: [dim]Not available[/dim]")
        else:
            # No saved profile, try to fetch fresh data
            try:
                # First try to get a valid access token (with refresh if needed)
                valid_token = asyncio.run(manager.get_access_token())
                if valid_token:
                    profile = asyncio.run(manager.fetch_user_profile())
                    if profile:
                        # Save the profile for future use
                        asyncio.run(manager._save_account_profile(profile))

                        if profile.organization:
                            console.print(
                                f"  L Organization: {profile.organization.name}"
                            )
                        else:
                            console.print(
                                "  L Organization: [dim]Unable to fetch[/dim]"
                            )

                        if profile.account:
                            console.print(f"  L Email: {profile.account.email}")
                        else:
                            console.print("  L Email: [dim]Unable to fetch[/dim]")
                    else:
                        console.print("  L Organization: [dim]Unable to fetch[/dim]")
                        console.print("  L Email: [dim]Unable to fetch[/dim]")

                    # Reload credentials after potential refresh to show updated token info
                    credentials = asyncio.run(manager.load())
                    if credentials:
                        oauth = credentials.claude_ai_oauth
                else:
                    console.print("  L Organization: [dim]Token refresh failed[/dim]")
                    console.print("  L Email: [dim]Token refresh failed[/dim]")
            except Exception as e:
                logger.debug(f"Could not fetch user profile: {e}")
                console.print("  L Organization: [dim]Unable to fetch[/dim]")
                console.print("  L Email: [dim]Unable to fetch[/dim]")

        # Create details table
        console.print()
        table = Table(
            show_header=True,
            header_style="bold cyan",
            box=box.ROUNDED,
            title="Credential Details",
            title_style="bold white",
        )
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="white")

        # File location - check if there's a credentials file or if using keyring
        cred_file = asyncio.run(manager.find_credentials_file())
        if cred_file:
            table.add_row("File Location", str(cred_file))
        else:
            table.add_row("File Location", "Keyring storage")

        # Token info
        table.add_row("Subscription Type", oauth.subscription_type or "Unknown")
        table.add_row(
            "Token Expired",
            "[red]Yes[/red]" if oauth.is_expired else "[green]No[/green]",
        )

        # Expiration details
        exp_dt = oauth.expires_at_datetime
        table.add_row("Expires At", exp_dt.strftime("%Y-%m-%d %H:%M:%S UTC"))

        # Time until expiration
        now = datetime.now(UTC)
        time_diff = exp_dt - now
        if time_diff.total_seconds() > 0:
            days = time_diff.days
            hours = (time_diff.seconds % 86400) // 3600
            minutes = (time_diff.seconds % 3600) // 60
            table.add_row(
                "Time Remaining", f"{days} days, {hours} hours, {minutes} minutes"
            )
        else:
            table.add_row("Time Remaining", "[red]Expired[/red]")

        # Scopes
        if oauth.scopes:
            table.add_row("OAuth Scopes", ", ".join(oauth.scopes))

        # Token preview (first and last 8 chars)
        if oauth.access_token:
            token_preview = f"{oauth.access_token[:8]}...{oauth.access_token[-8:]}"
            table.add_row("Access Token", f"[dim]{token_preview}[/dim]")

        # Account profile status
        account_profile_exists = profile is not None
        table.add_row(
            "Account Profile",
            "[green]Available[/green]"
            if account_profile_exists
            else "[yellow]Not saved[/yellow]",
        )

        console.print(table)

    except Exception as e:
        toolkit.print(f"Error getting credential info: {e}", tag="error")
        raise typer.Exit(1) from e


@app.command(name="login")
def login_command(
    docker: bool = typer.Option(
        False,
        "--docker",
        help="Use Docker credential paths (from get_claude_docker_home_dir())",
    ),
    credential_file: str | None = typer.Option(
        None,
        "--credential-file",
        help="Path to specific credential file to save to",
    ),
) -> None:
    """Login to Claude using OAuth authentication.

    This command will open your web browser to authenticate with Claude
    and save the credentials locally.

    Examples:
        ccproxy auth login
        ccproxy auth login --docker
        ccproxy auth login --credential-file /path/to/credentials.json
    """
    toolkit = get_rich_toolkit()
    toolkit.print("[bold cyan]Claude OAuth Login[/bold cyan]", centered=True)
    toolkit.print_line()

    try:
        # Get credential paths based on options
        custom_paths = None
        if credential_file:
            custom_paths = [Path(credential_file)]
        elif docker:
            custom_paths = get_docker_credential_paths()

        # Check if already logged in
        manager = get_credentials_manager(custom_paths)
        validation_result = asyncio.run(manager.validate())
        if validation_result.valid and not validation_result.expired:
            console.print(
                "[yellow]You are already logged in with valid credentials.[/yellow]"
            )
            console.print(
                "Use [cyan]ccproxy auth info[/cyan] to view current credentials."
            )

            overwrite = typer.confirm(
                "Do you want to login again and overwrite existing credentials?"
            )
            if not overwrite:
                console.print("Login cancelled.")
                return

        # Perform OAuth login
        console.print("Starting OAuth login process...")
        console.print("Your browser will open for authentication.")
        console.print(
            "A temporary server will start on port 54545 for the OAuth callback..."
        )

        try:
            asyncio.run(manager.login())
            success = True
        except Exception as e:
            logger.error(f"Login failed: {e}")
            success = False

        if success:
            toolkit.print("Successfully logged in to Claude!", tag="success")

            # Show credential info
            console.print("\n[dim]Credential information:[/dim]")
            updated_validation = asyncio.run(manager.validate())
            if updated_validation.valid and updated_validation.credentials:
                oauth_token = updated_validation.credentials.claude_ai_oauth
                console.print(
                    f"  Subscription: {oauth_token.subscription_type or 'Unknown'}"
                )
                if oauth_token.scopes:
                    console.print(f"  Scopes: {', '.join(oauth_token.scopes)}")
                exp_dt = oauth_token.expires_at_datetime
                console.print(f"  Expires: {exp_dt.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        else:
            toolkit.print("Login failed. Please try again.", tag="error")
            raise typer.Exit(1)

    except KeyboardInterrupt:
        console.print("\n[yellow]Login cancelled by user.[/yellow]")
        raise typer.Exit(1) from None
    except Exception as e:
        toolkit.print(f"Error during login: {e}", tag="error")
        raise typer.Exit(1) from e


@app.command()
def renew(
    docker: bool = typer.Option(
        False,
        "--docker",
        "-d",
        help="Renew credentials for Docker environment",
    ),
    credential_file: Path | None = typer.Option(
        None,
        "--credential-file",
        "-f",
        help="Path to custom credential file",
    ),
) -> None:
    """Force renew Claude credentials without checking expiration.

    This command will refresh your access token regardless of whether it's expired.
    Useful for testing or when you want to ensure you have the latest token.

    Examples:
        ccproxy auth renew
        ccproxy auth renew --docker
        ccproxy auth renew --credential-file /path/to/credentials.json
    """
    toolkit = get_rich_toolkit()
    toolkit.print("[bold cyan]Claude Credentials Renewal[/bold cyan]", centered=True)
    toolkit.print_line()

    console = Console()

    try:
        # Get credential paths based on options
        custom_paths = None
        if credential_file:
            custom_paths = [Path(credential_file)]
        elif docker:
            custom_paths = get_docker_credential_paths()

        # Create credentials manager
        manager = get_credentials_manager(custom_paths)

        # Check if credentials exist
        validation_result = asyncio.run(manager.validate())
        if not validation_result.valid:
            toolkit.print("[red]✗[/red] No credentials found to renew", tag="error")
            console.print("\n[dim]Please login first:[/dim]")
            console.print("[cyan]ccproxy auth login[/cyan]")
            raise typer.Exit(1)

        # Force refresh the token
        console.print("[yellow]Refreshing access token...[/yellow]")
        refreshed_credentials = asyncio.run(manager.refresh_token())

        if refreshed_credentials:
            toolkit.print(
                "[green]✓[/green] Successfully renewed credentials!", tag="success"
            )

            # Show updated credential info
            oauth_token = refreshed_credentials.claude_ai_oauth
            console.print("\n[dim]Updated credential information:[/dim]")
            console.print(
                f"  Subscription: {oauth_token.subscription_type or 'Unknown'}"
            )
            if oauth_token.scopes:
                console.print(f"  Scopes: {', '.join(oauth_token.scopes)}")
            exp_dt = oauth_token.expires_at_datetime
            console.print(f"  Expires: {exp_dt.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        else:
            toolkit.print("[red]✗[/red] Failed to renew credentials", tag="error")
            raise typer.Exit(1)

    except KeyboardInterrupt:
        console.print("\n[yellow]Renewal cancelled by user.[/yellow]")
        raise typer.Exit(1) from None
    except Exception as e:
        toolkit.print(f"Error during renewal: {e}", tag="error")
        raise typer.Exit(1) from e


if __name__ == "__main__":
    app()
