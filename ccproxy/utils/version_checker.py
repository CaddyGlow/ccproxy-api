"""Version checking utilities for ccproxy."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import aiofiles
import httpx
import structlog
from packaging import version as pkg_version
from pydantic import BaseModel

from ccproxy.config.utils import get_ccproxy_config_dir
from ccproxy.core._version import __version__


logger = structlog.get_logger(__name__)


BRANCH_OVERRIDE_ENV_VAR = "CCPROXY_VERSION_BRANCH"
GITHUB_API_BASE = "https://api.github.com/repos/CaddyGlow/ccproxy-api"


class VersionCheckState(BaseModel):
    """State tracking for version checks."""

    last_check_at: datetime
    latest_version_found: str | None = None
    latest_branch_name: str | None = None
    latest_branch_commit: str | None = None
    running_version: str | None = None
    running_commit: str | None = None


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


def extract_commit_from_version(version: str) -> str | None:
    """Extract a git commit SHA from a setuptools-scm formatted version."""

    match = re.search(r"\+g(?P<sha>[0-9a-f]{7,40})", version)
    if match:
        return match.group("sha")
    return None


def commit_refs_match(current: str | None, latest: str | None) -> bool:
    """Return True when two commit references identify the same commit."""

    if not current or not latest:
        return current == latest

    current_lower = current.lower()
    latest_lower = latest.lower()

    # Normalize shorter/longer pair for prefix comparison
    if len(current_lower) <= len(latest_lower):
        return latest_lower.startswith(current_lower)

    return current_lower.startswith(latest_lower)


def get_branch_override() -> str | None:
    """Return branch override from environment if provided."""

    env_branch = os.getenv(BRANCH_OVERRIDE_ENV_VAR, "").strip()
    return env_branch or None


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


async def fetch_latest_branch_commit(branch: str) -> str | None:
    """Fetch the latest commit SHA for a given branch from GitHub."""

    url = f"{GITHUB_API_BASE}/branches/{branch}"
    headers = {
        "User-Agent": f"ccproxy-api/{__version__}",
        "Accept": "application/vnd.github.v3+json",
    }

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(url, headers=headers)
            response.raise_for_status()

            data: dict[str, Any] = response.json()
            commit_info = data.get("commit", {})
            latest_sha = commit_info.get("sha")

            if isinstance(latest_sha, str) and latest_sha:
                logger.debug(
                    "github_branch_commit_fetched",
                    branch=branch,
                    latest_sha=latest_sha,
                )
                return latest_sha

            logger.warning(
                "github_branch_commit_missing_sha",
                branch=branch,
            )
            return None

    except httpx.TimeoutException:
        logger.warning("github_branch_commit_timeout", branch=branch)
        return None
    except httpx.HTTPStatusError as e:
        logger.warning(
            "github_branch_commit_http_error",
            branch=branch,
            status_code=e.response.status_code,
        )
        return None
    except httpx.RequestError as e:
        logger.warning(
            "github_branch_commit_http_request_error",
            branch=branch,
            error=str(e),
            error_type=type(e).__name__,
        )
        return None
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        logger.warning(
            "github_branch_commit_parse_error",
            branch=branch,
            error=str(e),
            error_type=type(e).__name__,
        )
        return None
    except Exception as e:
        logger.warning(
            "github_branch_commit_unexpected_error",
            branch=branch,
            error=str(e),
            error_type=type(e).__name__,
        )
        return None


async def fetch_branch_names_for_commit(commit: str) -> list[str]:
    """Fetch branch names for which the given commit is the HEAD."""

    url = f"{GITHUB_API_BASE}/commits/{commit}/branches-where-head"
    headers = {
        "User-Agent": f"ccproxy-api/{__version__}",
        "Accept": "application/vnd.github.v3+json",
    }

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(url, headers=headers)
            response.raise_for_status()

            data = response.json()
            if not isinstance(data, list):
                logger.warning(
                    "github_commit_branches_unexpected_payload",
                    commit=commit,
                    payload_type=type(data).__name__,
                )
                return []

            branch_names: list[str] = []
            for entry in data:
                if not isinstance(entry, dict):
                    continue
                name = entry.get("name")
                if isinstance(name, str) and name:
                    branch_names.append(name)

            logger.debug(
                "github_commit_branches_fetched",
                commit=commit,
                branch_count=len(branch_names),
            )
            return branch_names

    except httpx.TimeoutException:
        logger.warning("github_commit_branches_timeout", commit=commit)
        return []
    except httpx.HTTPStatusError as e:
        logger.warning(
            "github_commit_branches_http_error",
            commit=commit,
            status_code=e.response.status_code,
        )
        return []
    except httpx.RequestError as e:
        logger.warning(
            "github_commit_branches_http_request_error",
            commit=commit,
            error=str(e),
            error_type=type(e).__name__,
        )
        return []
    except (json.JSONDecodeError, ValueError, TypeError) as e:
        logger.warning(
            "github_commit_branches_parse_error",
            commit=commit,
            error=str(e),
            error_type=type(e).__name__,
        )
        return []
    except Exception as e:
        logger.warning(
            "github_commit_branches_unexpected_error",
            commit=commit,
            error=str(e),
            error_type=type(e).__name__,
        )
        return []


async def resolve_branch_for_commit(commit: str) -> str | None:
    """Resolve the branch name associated with the provided commit hash."""

    override = get_branch_override()
    if override:
        return override

    if not commit:
        return None

    branch_candidates = await fetch_branch_names_for_commit(commit)
    if not branch_candidates:
        return None

    # Prefer mainline branches if available
    preferred_order = ("main", "master", "develop", "dev")
    for preferred in preferred_order:
        if preferred in branch_candidates:
            return preferred

    # Otherwise return the first branch reported by GitHub
    return branch_candidates[0]


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
    "fetch_latest_branch_commit",
    "fetch_branch_names_for_commit",
    "resolve_branch_for_commit",
    "get_current_version",
    "extract_commit_from_version",
    "commit_refs_match",
    "get_branch_override",
    "compare_versions",
    "load_check_state",
    "save_check_state",
    "get_version_check_state_path",
]
