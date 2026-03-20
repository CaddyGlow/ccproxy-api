from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ccproxy.config.settings import Settings
from ccproxy.models.detection import DetectedPrompts
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

    expected_fallback = service._get_fallback_data()

    with (
        patch.object(
            service,
            "_get_fallback_data",
            MagicMock(return_value=expected_fallback),
        ),
        patch.object(
            service,
            "_detect_codex_headers",
            AsyncMock(
                side_effect=FileNotFoundError(
                    "Codex CLI not found for header detection"
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


def test_codex_detection_ignores_content_encoding_header() -> None:
    assert "content-encoding" in CodexDetectionService.ignores_header


def test_codex_detection_merges_partial_prompt_cache_with_fallback() -> None:
    settings = MagicMock(spec=Settings)
    cli_service = MagicMock()
    service = CodexDetectionService(settings=settings, cli_service=cli_service)

    cached_prompts = DetectedPrompts.from_body(
        {"tools": [{"type": "function", "name": "exec_command"}]}
    )
    fallback_prompts = DetectedPrompts.from_body(
        {
            "instructions": "Fallback instructions",
            "include": ["reasoning.encrypted_content"],
            "tool_choice": "auto",
        }
    )

    with (
        patch.object(
            service,
            "get_cached_data",
            return_value=SimpleNamespace(prompts=cached_prompts),
        ),
        patch.object(
            service,
            "_safe_fallback_data",
            return_value=SimpleNamespace(prompts=fallback_prompts),
        ),
    ):
        prompts = service.get_detected_prompts()

    assert prompts.instructions == "Fallback instructions"
    assert prompts.raw["tools"] == [{"type": "function", "name": "exec_command"}]
    assert prompts.raw["include"] == ["reasoning.encrypted_content"]
    assert prompts.raw["tool_choice"] == "auto"
