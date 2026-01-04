from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ccproxy.config.settings import Settings
from ccproxy.plugins.claude_api.detection_service import ClaudeAPIDetectionService


@pytest.mark.asyncio
async def test_claude_detection_falls_back_when_cli_missing(tmp_path: Path) -> None:
    settings = MagicMock(spec=Settings)
    cli_service = MagicMock()
    cli_service.get_cli_info.return_value = {"is_available": False, "command": None}
    cli_service.detect_cli = AsyncMock(
        return_value=SimpleNamespace(is_available=False, version=None)
    )

    service = ClaudeAPIDetectionService(settings=settings, cli_service=cli_service)
    service.cache_dir = tmp_path

    expected_fallback = service._get_fallback_data()

    with (
        patch.object(
            service,
            "_get_fallback_data",
            MagicMock(return_value=expected_fallback),
        ),
        patch.object(
            service,
            "_detect_claude_headers",
            AsyncMock(
                side_effect=FileNotFoundError(
                    "Claude CLI not found for header detection"
                )
            ),
        ) as mock_detect,
        patch.object(service, "_save_to_cache", MagicMock()) as mock_save,
    ):
        result = await service.initialize_detection()

    assert cli_service.detect_cli.await_count == 1
    mock_detect.assert_awaited_once_with("unknown")
    mock_save.assert_not_called()
    assert result is expected_fallback
    assert service.get_cached_data() is expected_fallback
