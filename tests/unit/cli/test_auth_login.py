"""Unit-level coverage for `ccproxy auth login`."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from typer.testing import CliRunner

from ccproxy.auth.oauth.cli_errors import AuthProviderError, PortBindError
from ccproxy.auth.oauth.registry import CliAuthConfig, FlowType
from ccproxy.cli.commands.auth import app as auth_app


@pytest.fixture
def cli_runner() -> CliRunner:
    """Provide Typer CLI runner."""

    return CliRunner()


@pytest.fixture
def mock_provider() -> MagicMock:
    """Return a configured OAuth provider double."""

    provider = MagicMock()
    provider.provider_name = "test-provider"
    provider.supports_pkce = True
    provider.supports_refresh = True
    provider.cli = CliAuthConfig(
        preferred_flow=FlowType.browser,
        callback_port=8080,
        callback_path="/callback",
        supports_manual_code=True,
        supports_device_flow=True,
    )

    provider.get_authorization_url = AsyncMock(return_value="https://example.com/auth")
    provider.handle_callback = AsyncMock(return_value={"access_token": "token"})
    provider.save_credentials = AsyncMock(return_value=True)
    provider.start_device_flow = AsyncMock(
        return_value=("device_code", "user_code", "https://example.com/verify", 600)
    )
    provider.complete_device_flow = AsyncMock(return_value={"access_token": "token"})
    provider.exchange_manual_code = AsyncMock(return_value={"access_token": "token"})

    return provider


class TestAuthLoginCLI:
    """Exercise the login command entrypoint under different flows."""

    def test_browser_flow_success(
        self, cli_runner: CliRunner, mock_provider: MagicMock
    ) -> None:
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
            patch("ccproxy.auth.oauth.flows.CLICallbackServer") as mock_server_class,
            patch("ccproxy.auth.oauth.flows.webbrowser"),
        ):
            mock_get_provider.return_value = mock_provider
            mock_discover.return_value = {"test-provider": ("oauth", "Test Provider")}
            mock_container.return_value = MagicMock()

            mock_server = AsyncMock()
            mock_server_class.return_value = mock_server
            mock_server.wait_for_callback.return_value = {
                "code": "test_code",
                "state": "state",
            }

            result = cli_runner.invoke(auth_app, ["login", "test-provider"])

        assert result.exit_code == 0
        assert "Authentication successful!" in result.stdout

    def test_device_flow_success(
        self, cli_runner: CliRunner, mock_provider: MagicMock
    ) -> None:
        mock_provider.cli = CliAuthConfig(
            preferred_flow=FlowType.device,
            supports_device_flow=True,
        )

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
            patch("ccproxy.auth.oauth.flows.render_qr_code"),
        ):
            mock_get_provider.return_value = mock_provider
            mock_discover.return_value = {"test-provider": ("oauth", "Test Provider")}
            mock_container.return_value = MagicMock()

            result = cli_runner.invoke(auth_app, ["login", "test-provider"])

        assert result.exit_code == 0
        assert "Authentication successful!" in result.stdout

    def test_manual_flow_success(
        self, cli_runner: CliRunner, mock_provider: MagicMock
    ) -> None:
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
            patch("ccproxy.auth.oauth.flows.typer.prompt") as mock_prompt,
            patch("ccproxy.auth.oauth.flows.render_qr_code"),
        ):
            mock_get_provider.return_value = mock_provider
            mock_discover.return_value = {"test-provider": ("oauth", "Test Provider")}
            mock_container.return_value = MagicMock()
            mock_prompt.return_value = "test_code"

            result = cli_runner.invoke(auth_app, ["login", "test-provider", "--manual"])

        assert result.exit_code == 0
        assert "Authentication successful!" in result.stdout

    def test_provider_not_found(self, cli_runner: CliRunner) -> None:
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
            mock_get_provider.return_value = None
            mock_discover.return_value = {}
            mock_container.return_value = MagicMock()

            result = cli_runner.invoke(auth_app, ["login", "missing"])

        assert result.exit_code == 1
        assert "not found" in result.stdout

    def test_port_bind_fallback_manual(
        self, cli_runner: CliRunner, mock_provider: MagicMock
    ) -> None:
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
            patch("ccproxy.auth.oauth.flows.CLICallbackServer") as mock_server_class,
            patch("ccproxy.auth.oauth.flows.typer.prompt") as mock_prompt,
            patch("ccproxy.auth.oauth.flows.render_qr_code"),
        ):
            mock_get_provider.return_value = mock_provider
            mock_discover.return_value = {"test-provider": ("oauth", "Test Provider")}
            mock_container.return_value = MagicMock()
            mock_prompt.return_value = "manual_code"

            mock_server = AsyncMock()
            mock_server_class.return_value = mock_server
            mock_server.start.side_effect = PortBindError("busy")

            result = cli_runner.invoke(auth_app, ["login", "test-provider"])

        assert result.exit_code == 0
        assert "Falling back to manual" in result.stdout
        assert "Authentication successful!" in result.stdout

    def test_manual_flow_not_supported(
        self, cli_runner: CliRunner, mock_provider: MagicMock
    ) -> None:
        mock_provider.cli = CliAuthConfig(supports_manual_code=False)

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
            mock_get_provider.return_value = mock_provider
            mock_discover.return_value = {"test-provider": ("oauth", "Test Provider")}
            mock_container.return_value = MagicMock()

            result = cli_runner.invoke(auth_app, ["login", "test-provider", "--manual"])

        assert result.exit_code == 1
        assert "doesn't support manual code entry" in " ".join(result.stdout.split())

    def test_keyboard_interrupt(
        self, cli_runner: CliRunner, mock_provider: MagicMock
    ) -> None:
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
            patch("ccproxy.auth.oauth.flows.BrowserFlow.run") as mock_run,
        ):
            mock_get_provider.return_value = mock_provider
            mock_discover.return_value = {"test-provider": ("oauth", "Test Provider")}
            mock_container.return_value = MagicMock()
            mock_run.side_effect = KeyboardInterrupt()

            result = cli_runner.invoke(auth_app, ["login", "test-provider"])

        assert result.exit_code == 2
        assert "Login cancelled by user" in result.stdout

    def test_auth_provider_error(
        self, cli_runner: CliRunner, mock_provider: MagicMock
    ) -> None:
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
            patch("ccproxy.auth.oauth.flows.BrowserFlow.run") as mock_run,
        ):
            mock_get_provider.return_value = mock_provider
            mock_discover.return_value = {"test-provider": ("oauth", "Test Provider")}
            mock_container.return_value = MagicMock()
            mock_run.side_effect = AuthProviderError("failed")

            result = cli_runner.invoke(auth_app, ["login", "test-provider"])

        assert result.exit_code == 1
        assert "Authentication failed" in result.stdout

    def test_port_bind_error_without_manual(
        self, cli_runner: CliRunner, mock_provider: MagicMock
    ) -> None:
        mock_provider.cli = CliAuthConfig(supports_manual_code=False)

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
            patch("ccproxy.auth.oauth.flows.CLICallbackServer") as mock_server_class,
        ):
            mock_get_provider.return_value = mock_provider
            mock_discover.return_value = {"test-provider": ("oauth", "Test Provider")}
            mock_container.return_value = MagicMock()

            mock_server = AsyncMock()
            mock_server_class.return_value = mock_server
            mock_server.start.side_effect = PortBindError("busy")

            result = cli_runner.invoke(auth_app, ["login", "test-provider"])

        assert result.exit_code == 1
        assert "unavailable and manual mode not supported" in result.stdout
