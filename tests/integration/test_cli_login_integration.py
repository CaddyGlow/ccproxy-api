"""Integration tests for CLI login command with flow engines."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from typer.testing import CliRunner

from ccproxy.auth.oauth.cli_errors import AuthProviderError, PortBindError
from ccproxy.auth.oauth.registry import CliAuthConfig, FlowType
from ccproxy.cli.commands.auth import app as auth_app


@pytest.fixture
def cli_runner() -> CliRunner:
    """Create CLI test runner."""
    return CliRunner()


@pytest.fixture
def mock_provider() -> MagicMock:
    """Mock OAuth provider for integration testing."""
    provider = MagicMock()
    provider.provider_name = "test-provider"
    provider.supports_pkce = True
    provider.cli = CliAuthConfig(
        preferred_flow=FlowType.browser,
        callback_port=8080,
        callback_path="/callback",
        supports_manual_code=True,
        supports_device_flow=True,
    )

    # Mock async methods
    provider.get_authorization_url = AsyncMock(return_value="https://example.com/auth")
    provider.handle_callback = AsyncMock(return_value={"access_token": "test_token"})
    provider.save_credentials = AsyncMock(return_value=True)
    provider.start_device_flow = AsyncMock(
        return_value=("device_code", "user_code", "https://example.com/verify", 600)
    )
    provider.complete_device_flow = AsyncMock(
        return_value={"access_token": "test_token"}
    )
    provider.exchange_manual_code = AsyncMock(
        return_value={"access_token": "test_token"}
    )

    return provider


class TestCLILoginIntegration:
    """Integration tests for CLI login command."""

    @pytest.mark.integration
    def test_login_command_browser_flow(
        self, cli_runner: CliRunner, mock_provider: MagicMock
    ) -> None:
        """Test login command with browser flow."""
        with (
            patch(
                "ccproxy.cli.commands.auth.get_oauth_provider_for_name"
            ) as mock_get_provider,
            patch(
                "ccproxy.cli.commands.auth.discover_oauth_providers"
            ) as mock_discover,
            patch("ccproxy.cli.commands.auth._get_service_container") as mock_container,
            patch("ccproxy.auth.oauth.flows.CLICallbackServer") as mock_server_class,
            patch("ccproxy.auth.oauth.flows.webbrowser"),
        ):
            mock_get_provider.return_value = asyncio.coroutine(lambda: mock_provider)()
            mock_discover.return_value = asyncio.coroutine(
                lambda: {"test-provider": ("oauth", "Test Provider")}
            )()
            mock_container.return_value = MagicMock()

            # Mock callback server
            mock_server = AsyncMock()
            mock_server_class.return_value = mock_server
            mock_server.wait_for_callback.return_value = {
                "code": "test_code",
                "state": "test_state",
            }

            result = cli_runner.invoke(auth_app, ["login", "test-provider"])

            assert result.exit_code == 0
            assert "Authentication successful!" in result.stdout

    @pytest.mark.integration
    def test_login_command_device_flow(
        self, cli_runner: CliRunner, mock_provider: MagicMock
    ) -> None:
        """Test login command with device flow."""
        # Configure provider for device flow
        mock_provider.cli = CliAuthConfig(
            preferred_flow=FlowType.device, supports_device_flow=True
        )

        with (
            patch(
                "ccproxy.cli.commands.auth.get_oauth_provider_for_name"
            ) as mock_get_provider,
            patch(
                "ccproxy.cli.commands.auth.discover_oauth_providers"
            ) as mock_discover,
            patch("ccproxy.cli.commands.auth._get_service_container") as mock_container,
            patch("ccproxy.auth.oauth.flows.render_qr_code"),
        ):
            mock_get_provider.return_value = asyncio.coroutine(lambda: mock_provider)()
            mock_discover.return_value = asyncio.coroutine(
                lambda: {"test-provider": ("oauth", "Test Provider")}
            )()
            mock_container.return_value = MagicMock()

            result = cli_runner.invoke(auth_app, ["login", "test-provider"])

            assert result.exit_code == 0
            assert "Authentication successful!" in result.stdout

    @pytest.mark.integration
    def test_login_command_manual_flow(
        self, cli_runner: CliRunner, mock_provider: MagicMock
    ) -> None:
        """Test login command with manual flow."""
        with (
            patch(
                "ccproxy.cli.commands.auth.get_oauth_provider_for_name"
            ) as mock_get_provider,
            patch(
                "ccproxy.cli.commands.auth.discover_oauth_providers"
            ) as mock_discover,
            patch("ccproxy.cli.commands.auth._get_service_container") as mock_container,
            patch("ccproxy.auth.oauth.flows.typer.prompt") as mock_prompt,
            patch("ccproxy.auth.oauth.flows.render_qr_code"),
        ):
            mock_get_provider.return_value = asyncio.coroutine(lambda: mock_provider)()
            mock_discover.return_value = asyncio.coroutine(
                lambda: {"test-provider": ("oauth", "Test Provider")}
            )()
            mock_container.return_value = MagicMock()
            mock_prompt.return_value = "test_code"

            result = cli_runner.invoke(auth_app, ["login", "test-provider", "--manual"])

            assert result.exit_code == 0
            assert "Authentication successful!" in result.stdout

    @pytest.mark.integration
    def test_login_command_provider_not_found(self, cli_runner: CliRunner) -> None:
        """Test login command with non-existent provider."""
        with (
            patch(
                "ccproxy.cli.commands.auth.get_oauth_provider_for_name"
            ) as mock_get_provider,
            patch(
                "ccproxy.cli.commands.auth.discover_oauth_providers"
            ) as mock_discover,
            patch("ccproxy.cli.commands.auth._get_service_container") as mock_container,
        ):
            mock_get_provider.return_value = asyncio.coroutine(lambda: None)()
            mock_discover.return_value = asyncio.coroutine(lambda: {})()
            mock_container.return_value = MagicMock()

            result = cli_runner.invoke(auth_app, ["login", "nonexistent-provider"])

            assert result.exit_code == 1
            assert "not found" in result.stdout

    @pytest.mark.integration
    def test_login_command_port_bind_fallback(
        self, cli_runner: CliRunner, mock_provider: MagicMock
    ) -> None:
        """Test login command with port bind error fallback to manual."""
        with (
            patch(
                "ccproxy.cli.commands.auth.get_oauth_provider_for_name"
            ) as mock_get_provider,
            patch(
                "ccproxy.cli.commands.auth.discover_oauth_providers"
            ) as mock_discover,
            patch("ccproxy.cli.commands.auth._get_service_container") as mock_container,
            patch("ccproxy.auth.oauth.flows.CLICallbackServer") as mock_server_class,
            patch("ccproxy.auth.oauth.flows.typer.prompt") as mock_prompt,
            patch("ccproxy.auth.oauth.flows.render_qr_code"),
        ):
            mock_get_provider.return_value = asyncio.coroutine(lambda: mock_provider)()
            mock_discover.return_value = asyncio.coroutine(
                lambda: {"test-provider": ("oauth", "Test Provider")}
            )()
            mock_container.return_value = MagicMock()
            mock_prompt.return_value = "test_code"

            # Mock port binding error
            mock_server = AsyncMock()
            mock_server_class.return_value = mock_server
            mock_server.start.side_effect = PortBindError("Port unavailable")

            result = cli_runner.invoke(auth_app, ["login", "test-provider"])

            assert result.exit_code == 0
            assert "Port binding failed. Falling back to manual mode." in result.stdout
            assert "Authentication successful!" in result.stdout

    @pytest.mark.integration
    def test_login_command_manual_not_supported(
        self, cli_runner: CliRunner, mock_provider: MagicMock
    ) -> None:
        """Test login command when manual mode is not supported."""
        # Configure provider to not support manual codes
        mock_provider.cli = CliAuthConfig(supports_manual_code=False)

        with (
            patch(
                "ccproxy.cli.commands.auth.get_oauth_provider_for_name"
            ) as mock_get_provider,
            patch(
                "ccproxy.cli.commands.auth.discover_oauth_providers"
            ) as mock_discover,
            patch("ccproxy.cli.commands.auth._get_service_container") as mock_container,
        ):
            mock_get_provider.return_value = asyncio.coroutine(lambda: mock_provider)()
            mock_discover.return_value = asyncio.coroutine(
                lambda: {"test-provider": ("oauth", "Test Provider")}
            )()
            mock_container.return_value = MagicMock()

            result = cli_runner.invoke(auth_app, ["login", "test-provider", "--manual"])

            assert result.exit_code == 1
            assert "doesn't support manual code entry" in result.stdout

    @pytest.mark.integration
    def test_login_command_keyboard_interrupt(
        self, cli_runner: CliRunner, mock_provider: MagicMock
    ) -> None:
        """Test login command handling keyboard interrupt."""
        with (
            patch(
                "ccproxy.cli.commands.auth.get_oauth_provider_for_name"
            ) as mock_get_provider,
            patch(
                "ccproxy.cli.commands.auth.discover_oauth_providers"
            ) as mock_discover,
            patch("ccproxy.cli.commands.auth._get_service_container") as mock_container,
            patch("ccproxy.auth.oauth.flows.BrowserFlow.run") as mock_flow_run,
        ):
            mock_get_provider.return_value = asyncio.coroutine(lambda: mock_provider)()
            mock_discover.return_value = asyncio.coroutine(
                lambda: {"test-provider": ("oauth", "Test Provider")}
            )()
            mock_container.return_value = MagicMock()
            mock_flow_run.side_effect = KeyboardInterrupt()

            result = cli_runner.invoke(auth_app, ["login", "test-provider"])

            assert result.exit_code == 2
            assert "Login cancelled by user" in result.stdout


class TestCLILoginErrorHandling:
    """Test error handling in CLI login command."""

    @pytest.mark.integration
    def test_auth_provider_error_handling(
        self, cli_runner: CliRunner, mock_provider: MagicMock
    ) -> None:
        """Test handling of AuthProviderError."""
        with (
            patch(
                "ccproxy.cli.commands.auth.get_oauth_provider_for_name"
            ) as mock_get_provider,
            patch(
                "ccproxy.cli.commands.auth.discover_oauth_providers"
            ) as mock_discover,
            patch("ccproxy.cli.commands.auth._get_service_container") as mock_container,
            patch("ccproxy.auth.oauth.flows.BrowserFlow.run") as mock_flow_run,
        ):
            mock_get_provider.return_value = asyncio.coroutine(lambda: mock_provider)()
            mock_discover.return_value = asyncio.coroutine(
                lambda: {"test-provider": ("oauth", "Test Provider")}
            )()
            mock_container.return_value = MagicMock()
            mock_flow_run.side_effect = AuthProviderError(
                "Provider authentication failed"
            )

            result = cli_runner.invoke(auth_app, ["login", "test-provider"])

            assert result.exit_code == 1
            assert (
                "Authentication failed: Provider authentication failed" in result.stdout
            )

    @pytest.mark.integration
    def test_port_bind_error_no_fallback(
        self, cli_runner: CliRunner, mock_provider: MagicMock
    ) -> None:
        """Test port bind error when manual fallback is not supported."""
        # Configure provider to not support manual codes
        mock_provider.cli = CliAuthConfig(supports_manual_code=False)

        with (
            patch(
                "ccproxy.cli.commands.auth.get_oauth_provider_for_name"
            ) as mock_get_provider,
            patch(
                "ccproxy.cli.commands.auth.discover_oauth_providers"
            ) as mock_discover,
            patch("ccproxy.cli.commands.auth._get_service_container") as mock_container,
            patch("ccproxy.auth.oauth.flows.CLICallbackServer") as mock_server_class,
        ):
            mock_get_provider.return_value = asyncio.coroutine(lambda: mock_provider)()
            mock_discover.return_value = asyncio.coroutine(
                lambda: {"test-provider": ("oauth", "Test Provider")}
            )()
            mock_container.return_value = MagicMock()

            # Mock port binding error
            mock_server = AsyncMock()
            mock_server_class.return_value = mock_server
            mock_server.start.side_effect = PortBindError("Port unavailable")

            result = cli_runner.invoke(auth_app, ["login", "test-provider"])

            assert result.exit_code == 1
            assert "unavailable and manual mode not supported" in result.stdout
