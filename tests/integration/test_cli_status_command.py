"""Integration tests for the auth status CLI command."""

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


def _build_credentials() -> SimpleNamespace:
    expires = datetime.now(UTC) + timedelta(hours=1)
    return SimpleNamespace(
        access_token="access-token-value",
        refresh_token="refresh-token",
        expires_at=expires,
        account_id="acct-123",
    )


class TestCLIStatusCommand:
    """Integration-level tests for `ccproxy auth status`."""

    @pytest.mark.integration
    def test_status_command_with_custom_file(
        self, cli_runner: CliRunner, tmp_path: Path
    ) -> None:
        provider = MagicMock()
        provider.load_credentials = AsyncMock(return_value=_build_credentials())
        provider.get_unified_profile_quick = AsyncMock()
        provider.get_unified_profile = AsyncMock()

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
                ["status", "test-provider", "--file", str(cred_file)],
            )

        assert result.exit_code == 0
        assert "Authenticated with valid credentials" in result.stdout
        provider.load_credentials.assert_awaited_once_with(custom_path=cred_file)
        provider.get_unified_profile_quick.assert_not_called()
        provider.get_unified_profile.assert_not_called()
        mock_resolve.assert_not_called()
