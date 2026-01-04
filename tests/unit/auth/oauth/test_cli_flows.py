"""Unit coverage for CLI OAuth flow engines."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ccproxy.auth.oauth.cli_errors import AuthProviderError, PortBindError
from ccproxy.auth.oauth.flows import (
    BrowserFlow,
    CLICallbackServer,
    DeviceCodeFlow,
    ManualCodeFlow,
    render_qr_code,
)
from ccproxy.auth.oauth.registry import CliAuthConfig, FlowType


@pytest.fixture
def mock_provider() -> MagicMock:
    provider = MagicMock()
    provider.supports_pkce = True
    provider.cli = CliAuthConfig(
        preferred_flow=FlowType.browser,
        callback_port=8080,
        callback_path="/callback",
        supports_manual_code=True,
        supports_device_flow=True,
    )
    provider.get_authorization_url = AsyncMock()
    provider.handle_callback = AsyncMock()
    provider.save_credentials = AsyncMock()
    provider.start_device_flow = AsyncMock()
    provider.complete_device_flow = AsyncMock()
    provider.exchange_manual_code = AsyncMock()
    return provider


class TestBrowserFlow:
    @pytest.mark.asyncio
    async def test_success(self, mock_provider: MagicMock) -> None:
        mock_provider.get_authorization_url.return_value = "https://example.com/auth"
        mock_provider.handle_callback.return_value = {"access_token": "token"}
        mock_provider.save_credentials.return_value = True

        with (
            patch("ccproxy.auth.oauth.flows.CLICallbackServer") as mock_server_class,
            patch("ccproxy.auth.oauth.flows.webbrowser") as mock_webbrowser,
            patch("ccproxy.auth.oauth.flows.render_qr_code") as mock_qr,
        ):
            mock_server = AsyncMock()
            mock_server.wait_for_callback.return_value = {
                "code": "auth_code",
                "state": "state",
            }
            mock_server_class.return_value = mock_server

            flow = BrowserFlow()
            result = await flow.run(mock_provider, no_browser=False)

        assert result is True
        mock_server.start.assert_called_once()
        mock_server.stop.assert_called_once()
        mock_webbrowser.open.assert_called_once()
        mock_qr.assert_called_once()
        mock_provider.get_authorization_url.assert_called_once()
        mock_provider.handle_callback.assert_called_once()
        mock_provider.save_credentials.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_browser_skips_open(self, mock_provider: MagicMock) -> None:
        mock_provider.get_authorization_url.return_value = "https://example.com/auth"
        mock_provider.handle_callback.return_value = {"access_token": "token"}
        mock_provider.save_credentials.return_value = True

        with (
            patch("ccproxy.auth.oauth.flows.CLICallbackServer") as mock_server_class,
            patch("ccproxy.auth.oauth.flows.webbrowser") as mock_webbrowser,
            patch("ccproxy.auth.oauth.flows.render_qr_code") as mock_qr,
        ):
            mock_server = AsyncMock()
            mock_server_class.return_value = mock_server
            mock_server.wait_for_callback.return_value = {
                "code": "auth_code",
                "state": "state",
            }

            flow = BrowserFlow()
            result = await flow.run(mock_provider, no_browser=True)

        assert result is True
        mock_webbrowser.open.assert_not_called()
        mock_qr.assert_called_once()

    @pytest.mark.asyncio
    async def test_port_bind_error(self, mock_provider: MagicMock) -> None:
        mock_provider.cli = CliAuthConfig(
            preferred_flow=FlowType.browser,
            callback_port=8080,
            callback_path="/callback",
            fixed_redirect_uri="http://localhost:9999/callback",
            supports_manual_code=True,
            supports_device_flow=True,
        )

        with patch("ccproxy.auth.oauth.flows.CLICallbackServer") as mock_server_class:
            mock_server = AsyncMock()
            mock_server.start.side_effect = PortBindError("busy")
            mock_server_class.return_value = mock_server

            flow = BrowserFlow()
            with pytest.raises(AuthProviderError, match="Required port 8080"):
                await flow.run(mock_provider, no_browser=False)

    @pytest.mark.asyncio
    async def test_timeout_falls_back_to_manual(self, mock_provider: MagicMock) -> None:
        mock_provider.cli = CliAuthConfig(
            preferred_flow=FlowType.browser,
            callback_port=8080,
            callback_path="/callback",
            supports_manual_code=True,
            supports_device_flow=False,
        )
        mock_provider.get_authorization_url.side_effect = [
            "https://example.com/auth",
            "https://example.com/auth?redirect_uri=urn:ietf:wg:oauth:2.0:oob",
        ]
        mock_provider.handle_callback.return_value = {"access_token": "token"}
        mock_provider.save_credentials.return_value = True

        with (
            patch("ccproxy.auth.oauth.flows.CLICallbackServer") as mock_server_class,
            patch("ccproxy.auth.oauth.flows.webbrowser") as mock_webbrowser,
            patch("ccproxy.auth.oauth.flows.render_qr_code") as mock_qr,
            patch("typer.prompt", return_value="manual_code") as mock_prompt,
        ):
            mock_server = AsyncMock()
            mock_server.wait_for_callback.side_effect = TimeoutError("timeout")
            mock_server_class.return_value = mock_server

            flow = BrowserFlow()
            result = await flow.run(mock_provider, no_browser=False)

        assert result is True
        mock_webbrowser.open.assert_called_once()
        mock_qr.assert_called_once()
        mock_prompt.assert_called_once()
        assert mock_provider.get_authorization_url.call_count == 2
        mock_provider.handle_callback.assert_called_once()


class TestDeviceCodeFlow:
    @pytest.mark.asyncio
    async def test_success(self, mock_provider: MagicMock) -> None:
        mock_provider.start_device_flow.return_value = (
            "device_code",
            "user_code",
            "https://example.com/verify",
            600,
        )
        mock_provider.complete_device_flow.return_value = {"access_token": "token"}
        mock_provider.save_credentials.return_value = True

        with patch("ccproxy.auth.oauth.flows.render_qr_code") as mock_qr:
            flow = DeviceCodeFlow()
            result = await flow.run(mock_provider)

        assert result is True
        mock_provider.start_device_flow.assert_called_once()
        mock_provider.complete_device_flow.assert_called_once()
        mock_provider.save_credentials.assert_called_once()
        mock_qr.assert_called_once_with("https://example.com/verify")


class TestManualCodeFlow:
    @pytest.mark.asyncio
    async def test_success(self, mock_provider: MagicMock) -> None:
        mock_provider.get_authorization_url.return_value = "https://example.com/auth"
        mock_provider.handle_callback.return_value = {"access_token": "token"}
        mock_provider.save_credentials.return_value = True

        with (
            patch(
                "ccproxy.auth.oauth.flows.typer.prompt", return_value="auth_code"
            ) as mock_prompt,
            patch("ccproxy.auth.oauth.flows.render_qr_code") as mock_qr,
        ):
            flow = ManualCodeFlow()
            result = await flow.run(mock_provider)

        assert result is True
        mock_prompt.assert_called_once()
        mock_qr.assert_called_once()
        mock_provider.handle_callback.assert_called_once()
        args = mock_provider.handle_callback.call_args[0]
        assert args[0] == "auth_code"
        assert args[3] == "urn:ietf:wg:oauth:2.0:oob"

    @pytest.mark.asyncio
    async def test_claude_code_state_format(self, mock_provider: MagicMock) -> None:
        mock_provider.get_authorization_url.return_value = "https://example.com/auth"
        mock_provider.handle_callback.return_value = {"access_token": "token"}
        mock_provider.save_credentials.return_value = True

        with (
            patch(
                "ccproxy.auth.oauth.flows.typer.prompt",
                return_value="code_part#state_part",
            ),
            patch("ccproxy.auth.oauth.flows.render_qr_code"),
        ):
            flow = ManualCodeFlow()
            result = await flow.run(mock_provider)

        assert result is True
        args = mock_provider.handle_callback.call_args[0]
        assert args[0] == "code_part"
        assert args[1] == "state_part"
        assert args[3] == "urn:ietf:wg:oauth:2.0:oob"


class TestCLICallbackServer:
    @pytest.mark.asyncio
    async def test_start_and_stop(self) -> None:
        server = CLICallbackServer(8080, "/callback")

        with patch("uvicorn.Server") as mock_server_class:
            mock_server = MagicMock()
            mock_server.serve = AsyncMock()
            mock_server.should_exit = False
            mock_server_class.return_value = mock_server

            await server.start()
            assert server.server is not None
            mock_server.serve.assert_called_once()

            await server.stop()
            assert mock_server.should_exit is True

    @pytest.mark.asyncio
    async def test_port_in_use_raises(self) -> None:
        server = CLICallbackServer(8080, "/callback")

        with patch("uvicorn.Server") as mock_server_class:
            mock_server = MagicMock()
            # Simulate uvicorn's behavior of calling sys.exit(1) on port binding errors
            mock_server.serve = AsyncMock(side_effect=SystemExit(1))
            mock_server_class.return_value = mock_server

            with pytest.raises(PortBindError):
                await server.start()


class TestQRCode:
    def test_render_qr_code_prints(self) -> None:
        with (
            patch("sys.stdout.isatty", return_value=True),
            patch("ccproxy.auth.oauth.flows.console.print") as mock_print,
        ):
            render_qr_code("https://example.com")

        mock_print.assert_called()
