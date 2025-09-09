"""GitHub CLI detection service for Copilot plugin."""

import asyncio
import json
import shutil
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, cast

from ccproxy.config.settings import Settings
from ccproxy.core.logging import get_plugin_logger

from .models import CopilotCacheData, CopilotCliInfo


if TYPE_CHECKING:
    from ccproxy.services.cli_detection import CLIDetectionService


logger = get_plugin_logger()


class CopilotDetectionService:
    """GitHub CLI detection and capability discovery service."""

    def __init__(self, settings: Settings, cli_service: "CLIDetectionService"):
        """Initialize detection service.

        Args:
            settings: Application settings
            cli_service: Core CLI detection service
        """
        self.settings = settings
        self._cli_service = cli_service
        self._cache: CopilotCacheData | None = None
        self._cache_ttl = timedelta(minutes=5)  # Cache for 5 minutes

    async def initialize_detection(self) -> CopilotCacheData:
        """Initialize GitHub CLI detection and cache results.

        Returns:
            Cached detection data
        """
        if self._cache and not self._is_cache_expired():
            logger.debug(
                "using_cached_detection_data",
                cache_age=(datetime.now() - self._cache.last_check).total_seconds(),
            )
            return self._cache

        logger.debug("initializing_github_cli_detection")

        # Check if GitHub CLI is available
        cli_path = self.get_cli_path()
        cli_available = cli_path is not None

        cli_version = None
        auth_status = None
        username = None

        if cli_available and cli_path:
            try:
                # Get CLI version
                version_result = await asyncio.create_subprocess_exec(
                    *cli_path,
                    "--version",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, stderr = await version_result.communicate()

                if version_result.returncode == 0:
                    version_output = stdout.decode().strip()
                    # Parse version from "gh version 2.x.x" format
                    for line in version_output.split("\n"):
                        if line.startswith("gh version"):
                            cli_version = (
                                line.split()[2] if len(line.split()) >= 3 else None
                            )
                            break

                # Check authentication status
                auth_result = await asyncio.create_subprocess_exec(
                    *cli_path,
                    "auth",
                    "status",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, stderr = await auth_result.communicate()

                if auth_result.returncode == 0:
                    auth_status = "authenticated"
                    auth_output = (
                        stderr.decode() + stdout.decode()
                    )  # gh auth status uses stderr

                    # Extract username from output
                    for line in auth_output.split("\n"):
                        if "Logged in to github.com as" in line:
                            parts = line.split()
                            if len(parts) >= 6:
                                username = parts[5].strip()
                                break
                else:
                    auth_status = "not_authenticated"

            except Exception as e:
                logger.warning(
                    "github_cli_check_failed",
                    error=str(e),
                    exc_info=e,
                )
                auth_status = "check_failed"

        # Update cache
        self._cache = CopilotCacheData(
            cli_available=cli_available,
            cli_version=cli_version,
            auth_status=auth_status,
            username=username,
            last_check=datetime.now(),
        )

        logger.info(
            "github_cli_detection_completed",
            cli_available=cli_available,
            cli_version=cli_version,
            auth_status=auth_status,
            username=username,
        )

        return self._cache

    def get_cli_path(self) -> list[str] | None:
        """Get GitHub CLI command path.

        Returns:
            CLI command as list of strings, or None if not available
        """
        # Try to find GitHub CLI
        cli_binary = shutil.which("gh")
        if cli_binary:
            return [cli_binary]

        logger.debug("github_cli_not_found")
        return None

    def get_cli_health_info(self) -> CopilotCliInfo:
        """Get GitHub CLI health information.

        Returns:
            CLI health information
        """
        if not self._cache:
            return CopilotCliInfo(
                available=False,
                version=None,
                authenticated=False,
                username=None,
                error="Detection not initialized - call initialize_detection() first",
            )

        return CopilotCliInfo(
            available=self._cache.cli_available,
            version=self._cache.cli_version,
            authenticated=self._cache.auth_status == "authenticated",
            username=self._cache.username,
            error=None if self._cache.cli_available else "GitHub CLI not found in PATH",
        )

    async def check_copilot_access(self) -> bool:
        """Check if the authenticated user has Copilot access.

        Returns:
            True if user has Copilot access, False otherwise
        """
        cli_path = self.get_cli_path()
        if not cli_path:
            logger.debug("no_cli_for_copilot_check")
            return False

        try:
            # Try to access Copilot API through GitHub CLI
            result = await asyncio.create_subprocess_exec(
                *cli_path,
                "api",
                "/user/copilot_business_accounts",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await result.communicate()

            if result.returncode == 0:
                # Parse response to check for business accounts
                try:
                    data = json.loads(stdout.decode())
                    business_accounts = data.get("copilot_business_accounts", [])
                    if business_accounts:
                        logger.debug("copilot_business_access_confirmed")
                        return True
                except json.JSONDecodeError:
                    pass

            # Try individual Copilot access
            result = await asyncio.create_subprocess_exec(
                *cli_path,
                "api",
                "/user/copilot",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await result.communicate()

            if result.returncode == 200:  # HTTP 200 OK
                logger.debug("copilot_individual_access_confirmed")
                return True

            logger.debug(
                "copilot_access_check_failed",
                returncode=result.returncode,
                stderr=stderr.decode().strip()[:200],  # First 200 chars
            )
            return False

        except Exception as e:
            logger.warning(
                "copilot_access_check_error",
                error=str(e),
                exc_info=e,
            )
            return False

    async def get_github_user_info(self) -> dict[str, Any] | None:
        """Get GitHub user information via CLI.

        Returns:
            User information dict or None if unavailable
        """
        cli_path = self.get_cli_path()
        if not cli_path:
            return None

        try:
            result = await asyncio.create_subprocess_exec(
                *cli_path,
                "api",
                "/user",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await result.communicate()

            if result.returncode == 0:
                try:
                    user_data = json.loads(stdout.decode())
                    logger.debug(
                        "github_user_info_retrieved",
                        login=user_data.get("login"),
                        name=user_data.get("name"),
                    )
                    return dict(user_data)
                except json.JSONDecodeError as e:
                    logger.warning(
                        "github_user_info_parse_failed",
                        error=str(e),
                    )

        except Exception as e:
            logger.warning(
                "github_user_info_request_failed",
                error=str(e),
                exc_info=e,
            )

        return None

    def _is_cache_expired(self) -> bool:
        """Check if detection cache has expired.

        Returns:
            True if cache is expired
        """
        if not self._cache:
            return True

        return datetime.now() - self._cache.last_check > self._cache_ttl

    async def refresh_cache(self) -> CopilotCacheData:
        """Force refresh of detection cache.

        Returns:
            Fresh detection data
        """
        logger.debug("forcing_detection_cache_refresh")
        self._cache = None
        return await self.initialize_detection()

    def get_recommended_headers(self) -> dict[str, str]:
        """Get recommended headers for Copilot API requests.

        Returns:
            Dictionary of headers
        """
        headers = {
            "Content-Type": "application/json",
            "Copilot-Integration-Id": "vscode-chat",
            "Editor-Version": "vscode/1.85.0",
            "Editor-Plugin-Version": "copilot-chat/0.26.7",
            "User-Agent": "GitHubCopilotChat/0.26.7",
            "X-GitHub-Api-Version": "2025-04-01",
        }

        # Add CLI version if available
        if self._cache and self._cache.cli_version:
            headers["X-GitHub-CLI-Version"] = self._cache.cli_version

        return headers

    async def validate_environment(self) -> dict[str, Any]:
        """Validate the environment for Copilot usage.

        Returns:
            Validation results with status and details
        """
        await self.initialize_detection()

        validation = {
            "status": "healthy",
            "details": {
                "github_cli": {
                    "available": self._cache.cli_available if self._cache else False,
                    "version": self._cache.cli_version if self._cache else None,
                    "authenticated": (
                        self._cache.auth_status == "authenticated"
                        if self._cache
                        else False
                    ),
                    "username": self._cache.username if self._cache else None,
                },
                "copilot_access": await self.check_copilot_access(),
                "last_check": self._cache.last_check.isoformat()
                if self._cache
                else None,
            },
        }

        # Determine overall health
        issues: list[str] = []
        details = cast(dict[str, Any], validation["details"])
        github_cli = cast(dict[str, Any], details["github_cli"])

        if not github_cli["available"]:
            issues.append("GitHub CLI not available")
        if not github_cli["authenticated"]:
            issues.append("GitHub CLI not authenticated")
        if not details["copilot_access"]:
            issues.append("No Copilot access detected")

        if issues:
            validation["status"] = "unhealthy"
            validation["issues"] = issues

        return validation
