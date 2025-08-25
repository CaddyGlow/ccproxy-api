"""Version checking utilities for ccproxy."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import aiofiles
import httpx
import structlog
from packaging import version as pkg_version
from pydantic import BaseModel

from ccproxy._version import __version__
from ccproxy.config.discovery import get_ccproxy_config_dir


logger = structlog.get_logger(__name__)


class VersionCheckState(BaseModel):
    """State tracking for version checks."""

    last_check_at: datetime
    latest_version_found: str | None = None


async def fetch_latest_github_version() -> str | None:
    """
    Fetch the latest version from GitHub releases API.

    Returns:
        Latest version string or None if unable to fetch
    """
    url = "https://api.github.com/repos/CaddyGlow/ccproxy-api/releases/latest"
    headers = {
        "User-Agent": f"ccproxy-api/{__version__}",
        "Accept": "application/vnd.github.v3+json",
    }

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(url, headers=headers)
            response.raise_for_status()

            data: dict[str, Any] = response.json()
            tag_name: str = str(data.get("tag_name", "")).lstrip("v")

            if tag_name:
                logger.debug("github_version_fetched", latest_version=tag_name)
                return tag_name

            logger.warning("github_version_missing_tag")
            return None

    except httpx.TimeoutException:
        logger.warning("github_version_timeout")
        return None
    except httpx.HTTPStatusError as e:
        logger.warning("github_version_http_error", status_code=e.response.status_code)
        return None
    except httpx.RequestError as e:
        logger.warning(
            "github_version_fetch_http_error",
            error=str(e),
            error_type=type(e).__name__,
        )
        return None
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        logger.warning(
            "github_version_parse_error",
            error=str(e),
            error_type=type(e).__name__,
        )
        return None
    except Exception as e:
        logger.warning(
            "github_version_fetch_unexpected_error",
            error=str(e),
            error_type=type(e).__name__,
        )
        return None


def get_current_version() -> str:
    """
    Get the current version of ccproxy.

    Returns:
        Current version string
    """
    return __version__


def compare_versions(current: str, latest: str) -> bool:
    """
    Compare version strings to determine if an update is available.

    Args:
        current: Current version string
        latest: Latest version string

    Returns:
        True if latest version is newer than current
    """
    try:
        current_parsed = pkg_version.parse(current)
        latest_parsed = pkg_version.parse(latest)

        # For dev versions, compare base version instead
        if current_parsed.is_devrelease:
            current_base = pkg_version.parse(current_parsed.base_version)
            return latest_parsed > current_base

        return latest_parsed > current_parsed
    except (ValueError, TypeError, AttributeError) as e:
        logger.error(
            "version_comparison_parse_error",
            current=current,
            latest=latest,
            error=str(e),
            error_type=type(e).__name__,
        )
        return False
    except Exception as e:
        logger.error(
            "version_comparison_unexpected_error",
            current=current,
            latest=latest,
            error=str(e),
            error_type=type(e).__name__,
        )
        return False


async def load_check_state(path: Path) -> VersionCheckState | None:
    """
    Load version check state from file.

    Args:
        path: Path to state file

    Returns:
        VersionCheckState if file exists and is valid, None otherwise
    """
    if not path.exists():
        return None

    try:
        async with aiofiles.open(path) as f:
            content = await f.read()
            data = json.loads(content)
            return VersionCheckState(**data)
    except (OSError, FileNotFoundError, PermissionError) as e:
        logger.warning(
            "version_check_state_load_file_error",
            path=str(path),
            error=str(e),
            error_type=type(e).__name__,
        )
        return None
    except (json.JSONDecodeError, ValueError, TypeError) as e:
        logger.warning(
            "version_check_state_load_parse_error",
            path=str(path),
            error=str(e),
            error_type=type(e).__name__,
        )
        return None
    except Exception as e:
        logger.warning(
            "version_check_state_load_unexpected_error",
            path=str(path),
            error=str(e),
            error_type=type(e).__name__,
        )
        return None


async def save_check_state(path: Path, state: VersionCheckState) -> None:
    """
    Save version check state to file.

    Args:
        path: Path to state file
        state: VersionCheckState to save
    """
    try:
        # Ensure directory exists
        path.parent.mkdir(parents=True, exist_ok=True)

        # Convert state to dict with ISO format datetime
        state_dict = state.model_dump()
        state_dict["last_check_at"] = state.last_check_at.isoformat()

        async with aiofiles.open(path, "w") as f:
            await f.write(json.dumps(state_dict, indent=2))

        logger.debug("version_check_state_saved", path=str(path))
    except (OSError, FileNotFoundError, PermissionError) as e:
        logger.warning(
            "version_check_state_save_file_error",
            path=str(path),
            error=str(e),
            error_type=type(e).__name__,
        )
    except (TypeError, ValueError) as e:
        logger.warning(
            "version_check_state_save_serialize_error",
            path=str(path),
            error=str(e),
            error_type=type(e).__name__,
        )
    except Exception as e:
        logger.warning(
            "version_check_state_save_unexpected_error",
            path=str(path),
            error=str(e),
            error_type=type(e).__name__,
        )


def get_version_check_state_path() -> Path:
    """
    Get the path to the version check state file.

    Returns:
        Path to version_check.json in ccproxy config directory
    """
    return get_ccproxy_config_dir() / "version_check.json"


__all__ = [
    "VersionCheckState",
    "fetch_latest_github_version",
    "get_current_version",
    "compare_versions",
    "load_check_state",
    "save_check_state",
    "get_version_check_state_path",
]
