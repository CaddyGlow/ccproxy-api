"""Integration tests for the auth refresh CLI command."""

from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from typer.testing import CliRunner

from ccproxy.cli.commands.auth import app as auth_app


@pytest.fixture
def cli_runner() -> CliRunner:
    """Provide Typer CLI runner."""
    return CliRunner()


def _build_credentials(has_refresh: bool = True) -> SimpleNamespace:
    expires = datetime.now(UTC) + timedelta(hours=1)
    refresh_token = "refresh-token" if has_refresh else None
    return SimpleNamespace(
        access_token="access-token-value",
        refresh_token=refresh_token,
        expires_at=expires,
        account_id="acct-123",
    )


class TestCLIRefreshCommand:
    """Integration-style tests for `ccproxy auth refresh`."""

    @pytest.mark.integration
    def test_refresh_command_uses_token_manager(self, cli_runner: CliRunner) -> None:
        provider = MagicMock()
        provider.supports_refresh = True
        provider.load_credentials = AsyncMock(return_value=_build_credentials())
        provider.refresh_access_token = AsyncMock()

        manager = MagicMock()
        refreshed_creds = _build_credentials()
        refreshed_creds.access_token = "new-access-token"
        manager.refresh_token = AsyncMock(return_value=refreshed_creds)
        manager.load_credentials = AsyncMock(return_value=refreshed_creds)

        with (
            patch(
                "ccproxy.cli.commands.auth.get_oauth_provider_for_name",
                new_callable=AsyncMock,
            ) as mock_get_provider,
            patch(
                "ccproxy.cli.commands.auth.discover_oauth_providers",
                new_callable=AsyncMock,
            ) as mock_discover,
            patch("ccproxy.cli.commands.auth._get_service_container") as mock_container,
            patch(
                "ccproxy.cli.commands.auth._resolve_token_manager",
                return_value=manager,
            ),
        ):
            mock_get_provider.return_value = provider
            mock_discover.return_value = {"test-provider": ("oauth", "Test Provider")}
            mock_container.return_value = MagicMock()

            result = cli_runner.invoke(auth_app, ["refresh", "test-provider"])

        assert result.exit_code == 0
        assert "Tokens refreshed successfully" in result.stdout
        manager.refresh_token.assert_awaited_once()
        provider.refresh_access_token.assert_not_called()

    @pytest.mark.integration
    def test_refresh_command_provider_without_support(
        self, cli_runner: CliRunner
    ) -> None:
        provider = MagicMock()
        provider.supports_refresh = False
        provider.load_credentials = AsyncMock()

        with (
            patch(
                "ccproxy.cli.commands.auth.get_oauth_provider_for_name",
                new_callable=AsyncMock,
            ) as mock_get_provider,
            patch(
                "ccproxy.cli.commands.auth.discover_oauth_providers",
                new_callable=AsyncMock,
            ) as mock_discover,
            patch("ccproxy.cli.commands.auth._get_service_container") as mock_container,
        ):
            mock_get_provider.return_value = provider
            mock_discover.return_value = {"test-provider": ("oauth", "Test Provider")}
            mock_container.return_value = MagicMock()

            result = cli_runner.invoke(auth_app, ["refresh", "test-provider"])

        assert result.exit_code == 1
        assert "does not support token refresh" in result.stdout

    @pytest.mark.integration
    def test_refresh_command_missing_refresh_token(self, cli_runner: CliRunner) -> None:
        provider = MagicMock()
        provider.supports_refresh = True
        provider.load_credentials = AsyncMock(
            return_value=_build_credentials(has_refresh=False)
        )
        provider.refresh_access_token = AsyncMock()

        with (
            patch(
                "ccproxy.cli.commands.auth.get_oauth_provider_for_name",
                new_callable=AsyncMock,
            ) as mock_get_provider,
            patch(
                "ccproxy.cli.commands.auth.discover_oauth_providers",
                new_callable=AsyncMock,
            ) as mock_discover,
            patch("ccproxy.cli.commands.auth._get_service_container") as mock_container,
            patch(
                "ccproxy.cli.commands.auth._resolve_token_manager",
                return_value=None,
            ),
        ):
            mock_get_provider.return_value = provider
            mock_discover.return_value = {"test-provider": ("oauth", "Test Provider")}
            mock_container.return_value = MagicMock()

            result = cli_runner.invoke(auth_app, ["refresh", "test-provider"])

        assert result.exit_code == 1
        assert "re-authentication is required" in result.stdout
        provider.refresh_access_token.assert_not_called()

    @pytest.mark.integration
    def test_refresh_command_with_custom_file(
        self, cli_runner: CliRunner, tmp_path: Path
    ) -> None:
        provider = MagicMock()
        provider.supports_refresh = True
        provider.load_credentials = AsyncMock(return_value=_build_credentials())
        refreshed = _build_credentials()
        refreshed.access_token = "rotated-access"
        provider.refresh_access_token = AsyncMock(return_value=refreshed)
        provider.save_credentials = AsyncMock(return_value=True)

        cred_file = tmp_path / "tokens.json"
        cred_file.write_text("{}", encoding="utf-8")

        with (
            patch(
                "ccproxy.cli.commands.auth.get_oauth_provider_for_name",
                new_callable=AsyncMock,
            ) as mock_get_provider,
            patch(
                "ccproxy.cli.commands.auth.discover_oauth_providers",
                new_callable=AsyncMock,
            ) as mock_discover,
            patch("ccproxy.cli.commands.auth._get_service_container") as mock_container,
            patch("ccproxy.cli.commands.auth._resolve_token_manager") as mock_resolve,
        ):
            mock_get_provider.return_value = provider
            mock_discover.return_value = {"test-provider": ("oauth", "Test Provider")}
            mock_container.return_value = MagicMock()
            mock_resolve.return_value = None

            result = cli_runner.invoke(
                auth_app,
                ["refresh", "test-provider", "--file", str(cred_file)],
            )

        assert result.exit_code == 0
        assert "Tokens refreshed successfully" in result.stdout
        provider.load_credentials.assert_awaited_once_with(custom_path=cred_file)
        provider.refresh_access_token.assert_awaited_once()
        provider.save_credentials.assert_awaited_once_with(
            refreshed, custom_path=cred_file
        )
        mock_resolve.assert_not_called()
