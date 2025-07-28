"""Unit tests for version checker utilities."""

import tempfile
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from packaging import version as pkg_version

from ccproxy.utils.version_checker import (
    VersionCheckState,
    compare_versions,
    fetch_latest_github_version,
    get_current_version,
    get_version_check_state_path,
    load_check_state,
    save_check_state,
)


class TestVersionCheckState:
    """Test VersionCheckState model."""

    def test_version_check_state_creation(self) -> None:
        """Test creating VersionCheckState instance."""
        now = datetime.now(UTC)
        state = VersionCheckState(
            last_check_at=now,
            latest_version_found="1.2.3",
        )

        assert state.last_check_at == now
        assert state.latest_version_found == "1.2.3"

    def test_version_check_state_without_version(self) -> None:
        """Test creating VersionCheckState without latest version."""
        now = datetime.now(UTC)
        state = VersionCheckState(last_check_at=now)

        assert state.last_check_at == now
        assert state.latest_version_found is None


class TestVersionComparison:
    """Test version comparison functionality."""

    def test_compare_versions_newer_available(self) -> None:
        """Test comparison when newer version is available."""
        assert compare_versions("1.0.0", "1.1.0") is True
        assert compare_versions("1.0.0", "2.0.0") is True
        assert compare_versions("1.0.0", "1.0.1") is True

    def test_compare_versions_same_version(self) -> None:
        """Test comparison when versions are the same."""
        assert compare_versions("1.0.0", "1.0.0") is False

    def test_compare_versions_older_latest(self) -> None:
        """Test comparison when current version is newer."""
        assert compare_versions("1.1.0", "1.0.0") is False
        assert compare_versions("2.0.0", "1.9.9") is False

    def test_compare_versions_dev_versions(self) -> None:
        """Test comparison with development versions."""
        assert compare_versions("1.0.0.dev1", "1.0.0") is True
        assert compare_versions("1.0.0", "1.0.1.dev1") is True

    def test_compare_versions_invalid_format(self) -> None:
        """Test comparison with invalid version formats."""
        assert compare_versions("invalid", "1.0.0") is False
        assert compare_versions("1.0.0", "invalid") is False
        assert compare_versions("invalid", "also-invalid") is False


class TestCurrentVersion:
    """Test current version retrieval."""

    def test_get_current_version(self) -> None:
        """Test getting current version."""
        version = get_current_version()
        assert isinstance(version, str)
        assert len(version) > 0
        # Should be parseable as a version
        pkg_version.parse(version)


class TestGitHubVersionFetching:
    """Test GitHub version fetching functionality."""

    @pytest.mark.asyncio
    async def test_fetch_latest_github_version_success(self) -> None:
        """Test successful GitHub version fetch."""
        from unittest.mock import MagicMock

        mock_response = MagicMock()
        mock_response.json.return_value = {"tag_name": "v1.2.3"}
        mock_response.raise_for_status.return_value = None

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response

        with patch("httpx.AsyncClient") as mock_async_client:
            mock_async_client.return_value.__aenter__.return_value = mock_client

            result = await fetch_latest_github_version()

            assert result == "1.2.3"
            mock_client.get.assert_called_once()
            call_args = mock_client.get.call_args
            assert "repos/CaddyGlow/ccproxy-api/releases/latest" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_fetch_latest_github_version_no_v_prefix(self) -> None:
        """Test GitHub version fetch when tag has no 'v' prefix."""
        from unittest.mock import MagicMock

        mock_response = MagicMock()
        mock_response.json.return_value = {"tag_name": "1.2.3"}
        mock_response.raise_for_status.return_value = None

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response

        with patch("httpx.AsyncClient") as mock_async_client:
            mock_async_client.return_value.__aenter__.return_value = mock_client

            result = await fetch_latest_github_version()

            assert result == "1.2.3"

    @pytest.mark.asyncio
    async def test_fetch_latest_github_version_missing_tag(self) -> None:
        """Test GitHub version fetch when tag_name is missing."""
        mock_response = AsyncMock()
        mock_response.json.return_value = {}
        mock_response.raise_for_status.return_value = None

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response

        with patch("httpx.AsyncClient") as mock_async_client:
            mock_async_client.return_value.__aenter__.return_value = mock_client

            result = await fetch_latest_github_version()

            assert result is None

    @pytest.mark.asyncio
    async def test_fetch_latest_github_version_timeout(self) -> None:
        """Test GitHub version fetch timeout."""
        mock_client = AsyncMock()
        mock_client.get.side_effect = httpx.TimeoutException("Timeout")

        with patch("httpx.AsyncClient") as mock_async_client:
            mock_async_client.return_value.__aenter__.return_value = mock_client

            result = await fetch_latest_github_version()

            assert result is None

    @pytest.mark.asyncio
    async def test_fetch_latest_github_version_http_error(self) -> None:
        """Test GitHub version fetch HTTP error."""
        mock_response = AsyncMock()
        mock_response.status_code = 404

        mock_client = AsyncMock()
        mock_client.get.side_effect = httpx.HTTPStatusError(
            "Not found", request=AsyncMock(), response=mock_response
        )

        with patch("httpx.AsyncClient") as mock_async_client:
            mock_async_client.return_value.__aenter__.return_value = mock_client

            result = await fetch_latest_github_version()

            assert result is None

    @pytest.mark.asyncio
    async def test_fetch_latest_github_version_generic_error(self) -> None:
        """Test GitHub version fetch generic error."""
        mock_client = AsyncMock()
        mock_client.get.side_effect = Exception("Network error")

        with patch("httpx.AsyncClient") as mock_async_client:
            mock_async_client.return_value.__aenter__.return_value = mock_client

            result = await fetch_latest_github_version()

            assert result is None


class TestStateManagement:
    """Test version check state file management."""

    @pytest.mark.asyncio
    async def test_save_and_load_check_state(self) -> None:
        """Test saving and loading version check state."""
        with tempfile.TemporaryDirectory() as temp_dir:
            state_path = Path(temp_dir) / "version_check.json"
            now = datetime.now(UTC)

            # Create and save state
            original_state = VersionCheckState(
                last_check_at=now,
                latest_version_found="1.2.3",
            )

            await save_check_state(state_path, original_state)

            # Verify file was created
            assert state_path.exists()

            # Load state back
            loaded_state = await load_check_state(state_path)

            assert loaded_state is not None
            assert loaded_state.last_check_at == now
            assert loaded_state.latest_version_found == "1.2.3"

    @pytest.mark.asyncio
    async def test_load_check_state_nonexistent_file(self) -> None:
        """Test loading state from nonexistent file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            state_path = Path(temp_dir) / "nonexistent.json"

            result = await load_check_state(state_path)

            assert result is None

    @pytest.mark.asyncio
    async def test_load_check_state_invalid_json(self) -> None:
        """Test loading state from file with invalid JSON."""
        with tempfile.TemporaryDirectory() as temp_dir:
            state_path = Path(temp_dir) / "invalid.json"

            # Write invalid JSON
            with state_path.open("w") as f:
                f.write("invalid json content")

            result = await load_check_state(state_path)

            assert result is None

    @pytest.mark.asyncio
    async def test_save_check_state_creates_directory(self) -> None:
        """Test that save_check_state creates parent directories."""
        with tempfile.TemporaryDirectory() as temp_dir:
            nested_path = Path(temp_dir) / "nested" / "dirs" / "version_check.json"
            now = datetime.now(UTC)

            state = VersionCheckState(
                last_check_at=now,
                latest_version_found="1.0.0",
            )

            await save_check_state(nested_path, state)

            # Verify file and directories were created
            assert nested_path.exists()
            assert nested_path.parent.exists()

    def test_get_version_check_state_path(self) -> None:
        """Test getting version check state path."""
        path = get_version_check_state_path()

        assert isinstance(path, Path)
        assert path.name == "version_check.json"
        assert "ccproxy" in str(path)


# Integration test for realistic scenario
class TestVersionCheckIntegration:
    """Integration tests for version checking workflow."""

    @pytest.mark.asyncio
    async def test_complete_version_check_workflow(self) -> None:
        """Test complete version check workflow."""
        with tempfile.TemporaryDirectory() as temp_dir:
            state_path = Path(temp_dir) / "version_check.json"

            # Mock GitHub API response
            from unittest.mock import MagicMock

            mock_response = MagicMock()
            mock_response.json.return_value = {"tag_name": "v1.5.0"}
            mock_response.raise_for_status.return_value = None

            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response

            with patch("httpx.AsyncClient") as mock_async_client:
                mock_async_client.return_value.__aenter__.return_value = mock_client

                # Fetch latest version
                latest_version = await fetch_latest_github_version()
                assert latest_version == "1.5.0"

                # Get current version
                current_version = get_current_version()

                # Compare versions (assuming current is older)
                has_update = compare_versions(current_version, latest_version)

                # Save state
                now = datetime.now(UTC)
                state = VersionCheckState(
                    last_check_at=now,
                    latest_version_found=latest_version,
                )
                await save_check_state(state_path, state)

                # Load state back to verify persistence
                loaded_state = await load_check_state(state_path)
                assert loaded_state is not None
                assert loaded_state.latest_version_found == "1.5.0"
