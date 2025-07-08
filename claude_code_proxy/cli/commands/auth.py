"""Authentication and credential management commands."""

import asyncio
import json
from datetime import UTC, datetime, timezone
from pathlib import Path

import typer
from rich import box
from rich.console import Console
from rich.table import Table

from claude_code_proxy.services.credentials import CredentialsConfig, CredentialsManager
from claude_code_proxy.utils.cli import get_rich_toolkit
from claude_code_proxy.utils.logging import get_logger
from claude_code_proxy.utils.xdg import get_claude_docker_home_dir


app = typer.Typer(name="auth", help="Authentication and credential management")

console = Console()
logger = get_logger(__name__)


def get_credentials_manager(
    custom_paths: list[Path] | None = None,
) -> CredentialsManager:
    """Get a CredentialsManager instance with custom paths if provided."""
    if custom_paths:
        config = CredentialsConfig(storage_paths=[str(p) for p in custom_paths])
    else:
        config = CredentialsConfig()
    return CredentialsManager(config=config)


def get_docker_credential_paths() -> list[Path]:
    """Get credential file paths for Docker environment."""
    docker_home = get_claude_docker_home_dir()
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

        if validation_result.get("valid"):
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
            status = "Valid" if not validation_result.get("expired") else "Expired"
            status_style = "green" if not validation_result.get("expired") else "red"
            table.add_row("Status", f"[{status_style}]{status}[/{status_style}]")

            # Subscription type
            sub_type = validation_result.get("subscription_type", "Unknown")
            table.add_row("Subscription", f"[bold]{sub_type}[/bold]")

            # Expiration
            expires_at = validation_result.get("expires_at")
            if expires_at and isinstance(expires_at, str):
                exp_dt = datetime.fromisoformat(expires_at)
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
            scopes = validation_result.get("scopes", [])
            if scopes and isinstance(scopes, list):
                table.add_row("Scopes", ", ".join(str(s) for s in scopes))

            console.print(table)

            # Success message
            if not validation_result.get("expired"):
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
            error_msg = validation_result.get("error", "Unknown error")
            toolkit.print(f"[red]✗[/red] {error_msg}", tag="error")

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

        # Get credentials manager and find credential file
        manager = get_credentials_manager(custom_paths)
        cred_file = asyncio.run(manager.find_credentials_file())

        if not cred_file:
            toolkit.print("No credential file found", tag="error")
            console.print("\n[dim]Expected locations:[/dim]")
            for path in manager.config.storage_paths:
                console.print(f"  - {path}")
            raise typer.Exit(1)

        # Load and display credentials
        credentials = asyncio.run(manager.load())
        if not credentials:
            toolkit.print("Failed to load credentials", tag="error")
            raise typer.Exit(1)

        # Display account section
        console.print("\n[bold]Account • /login[/bold]")
        oauth = credentials.claude_ai_oauth

        # Login method based on subscription type
        login_method = "Claude Account"
        if oauth.subscription_type:
            login_method = f"Claude {oauth.subscription_type.title()} Account"
        console.print(f"  L Login Method: {login_method}")

        # Try to fetch user profile for organization and email
        # Use refresh-enabled token method to ensure we have a valid token
        try:
            # First try to get a valid access token (with refresh if needed)
            valid_token = asyncio.run(manager.get_access_token())
            if valid_token:
                profile = asyncio.run(manager.fetch_user_profile())
                if profile and profile.organization:
                    console.print(f"  L Organization: {profile.organization.name}")
                else:
                    console.print("  L Organization: [dim]Unable to fetch[/dim]")

                if profile and profile.account:
                    console.print(f"  L Email: {profile.account.email_address}")
                else:
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

        # File location
        table.add_row("File Location", str(cred_file))

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
        if validation_result.get("valid") and not validation_result.get("expired"):
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
            if updated_validation.get("valid"):
                console.print(
                    f"  Subscription: {updated_validation.get('subscription_type', 'Unknown')}"
                )
                scopes = updated_validation.get("scopes", [])
                if isinstance(scopes, list):
                    console.print(f"  Scopes: {', '.join(scopes)}")
                else:
                    console.print(f"  Scopes: {scopes}")
                expires_at = updated_validation.get("expires_at")
                if expires_at:
                    console.print(f"  Expires: {expires_at}")
        else:
            toolkit.print("Login failed. Please try again.", tag="error")
            raise typer.Exit(1)

    except KeyboardInterrupt:
        console.print("\n[yellow]Login cancelled by user.[/yellow]")
        raise typer.Exit(1) from None
    except Exception as e:
        toolkit.print(f"Error during login: {e}", tag="error")
        raise typer.Exit(1) from e


if __name__ == "__main__":
    app()
