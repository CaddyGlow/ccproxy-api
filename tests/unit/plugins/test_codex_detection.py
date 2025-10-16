from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ccproxy.config.settings import Settings
from ccproxy.plugins.codex.detection_service import CodexDetectionService


@pytest.mark.asyncio
async def test_codex_detection_falls_back_when_cli_missing(tmp_path: Path) -> None:
    settings = MagicMock(spec=Settings)
    cli_service = MagicMock()
    cli_service.get_cli_info.return_value = {"is_available": False, "command": None}
    cli_service.detect_cli = AsyncMock(
        return_value=SimpleNamespace(is_available=False, version=None)
    )

    service = CodexDetectionService(settings=settings, cli_service=cli_service)
    service.cache_dir = tmp_path

    with (
        patch.object(
            service,
            "_get_codex_version",
            AsyncMock(side_effect=FileNotFoundError("missing cli")),
        ),
        patch.object(
            service,
            "_detect_codex_headers",
            AsyncMock(side_effect=RuntimeError("should not run")),
        ),
    ):
        result = await service.initialize_detection()

    assert result == service._get_fallback_data()
    assert service.get_cached_data() == result
