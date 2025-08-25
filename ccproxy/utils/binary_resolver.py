"""Binary resolution with package manager fallback support."""

import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple, TypedDict

from ccproxy.core.logging import TraceBoundLogger, get_logger
from ccproxy.utils.caching import ttl_cache


if TYPE_CHECKING:
    from ccproxy.config.settings import Settings

logger: TraceBoundLogger = get_logger()


class BinaryCommand(NamedTuple):
    """Represents a resolved binary command."""

    command: list[str]
    is_direct: bool
    is_in_path: bool
    package_manager: str | None = None


class PackageManagerConfig(TypedDict, total=False):
    """Configuration for a package manager."""

    check_cmd: list[str]
    priority: int
    exec_cmd: str  # Optional field


class CLIInfo(TypedDict):
    """Common structure for CLI information."""

    name: str  # CLI name (e.g., "claude", "codex")
    version: str | None  # Version string
    source: str  # "path" | "package_manager" | "unknown"
    path: str | None  # Direct path if available
    command: list[str]  # Full command to execute
    package_manager: str | None  # Package manager used (if applicable)
    is_available: bool  # Whether the CLI is accessible


class BinaryResolver:
    """Resolves binaries with fallback to package managers."""

    PACKAGE_MANAGERS: dict[str, PackageManagerConfig] = {
        "bunx": {"check_cmd": ["bun", "--version"], "priority": 1},
        "pnpm": {"check_cmd": ["pnpm", "--version"], "exec_cmd": "dlx", "priority": 2},
        "npx": {"check_cmd": ["npx", "--version"], "priority": 3},
    }

    KNOWN_PACKAGES = {
        "claude": "@anthropic-ai/claude-code",
        "codex": "@openai/codex",
        "gemini": "@google/gemini-cli",
    }

    def __init__(
        self,
        fallback_enabled: bool = True,
        package_manager_only: bool = False,
        preferred_package_manager: str | None = None,
        package_manager_priority: list[str] | None = None,
    ):
        """Initialize the binary resolver.

        Args:
            fallback_enabled: Whether to use package manager fallback
            package_manager_only: Skip direct binary lookup and use package managers exclusively
            preferred_package_manager: Preferred package manager (bunx, pnpm, npx)
            package_manager_priority: Custom priority order for package managers
        """
        self.fallback_enabled = fallback_enabled
        self.package_manager_only = package_manager_only
        self.preferred_package_manager = preferred_package_manager
        self.package_manager_priority = package_manager_priority or [
            "bunx",
            "pnpm",
            "npx",
        ]
        self._available_managers: dict[str, bool] | None = None

    @ttl_cache(maxsize=32, ttl=300.0)
    def find_binary(
        self,
        binary_name: str,
        package_name: str | None = None,
        package_manager_only: bool | None = None,
        fallback_enabled: bool | None = None,
    ) -> BinaryCommand | None:
        """Find a binary with optional package manager fallback.

        Args:
            binary_name: Name of the binary to find. Can be:
                - Simple binary name (e.g., "claude")
                - Full package name (e.g., "@anthropic-ai/claude-code")
            package_name: NPM package name if different from binary name

        Returns:
            BinaryCommand with resolved command or None if not found
        """
        if package_manager_only is None:
            package_manager_only = self.package_manager_only
        if fallback_enabled is None:
            fallback_enabled = self.fallback_enabled

        # Determine if binary_name is a full package name (contains @ or /)
        is_full_package = "@" in binary_name or "/" in binary_name

        if is_full_package and package_name is None:
            # If binary_name is a full package name, use it as the package
            # and extract the binary name from it
            package_name = binary_name
            # Extract binary name from package (last part after /)
            binary_name = binary_name.split("/")[-1]

        # If package_manager_only mode, skip direct binary lookup
        if package_manager_only:
            package_name = package_name or self.KNOWN_PACKAGES.get(
                binary_name, binary_name
            )
            result = self._find_via_package_manager(binary_name, package_name)
            if result:
                logger.trace(
                    "binary_resolved",
                    binary=binary_name,
                    manager=result.package_manager,
                    command=result.command,
                    source="package_manager",
                )
            else:
                logger.trace(
                    "binary_resolution_failed",
                    binary=binary_name,
                    source="package_manager",
                )
            return result

        # First, try direct binary lookup in PATH
        direct_path = shutil.which(binary_name)
        if direct_path:
            return BinaryCommand(command=[direct_path], is_direct=True, is_in_path=True)

        # Check common installation locations
        common_paths = self._get_common_paths(binary_name)
        for path in common_paths:
            if path.exists() and path.is_file():
                logger.debug(
                    "binary_found_in_common_path", binary=binary_name, path=str(path)
                )
                return BinaryCommand(
                    command=[str(path)], is_direct=True, is_in_path=False
                )

        # If fallback is disabled, stop here
        if not fallback_enabled:
            logger.debug("binary_fallback_disabled", binary=binary_name)
            return None

        # Try package manager fallback
        package_name = package_name or self.KNOWN_PACKAGES.get(binary_name, binary_name)
        return self._find_via_package_manager(binary_name, package_name)

    def _find_via_package_manager(
        self, binary_name: str, package_name: str
    ) -> BinaryCommand | None:
        """Find binary via package manager execution.

        Args:
            binary_name: Name of the binary
            package_name: NPM package name

        Returns:
            BinaryCommand with package manager command or None
        """
        # Get available package managers
        available = self._get_available_managers()

        # If preferred manager is set and available, try it first
        if (
            self.preferred_package_manager
            and self.preferred_package_manager in available
        ):
            cmd = self._build_package_manager_command(
                self.preferred_package_manager, package_name
            )
            if cmd:
                logger.debug(
                    "binary_using_preferred_manager",
                    binary=binary_name,
                    manager=self.preferred_package_manager,
                    command=cmd,
                )
                return BinaryCommand(
                    command=cmd,
                    is_direct=False,
                    is_in_path=False,
                    package_manager=self.preferred_package_manager,
                )

        # Try package managers in priority order
        for manager_name in self.package_manager_priority:
            if manager_name not in available or not available[manager_name]:
                continue

            cmd = self._build_package_manager_command(manager_name, package_name)
            if cmd:
                return BinaryCommand(
                    command=cmd,
                    is_direct=False,
                    is_in_path=False,
                    package_manager=manager_name,
                )

        logger.debug(
            "binary_not_found_with_fallback",
            binary=binary_name,
            package=package_name,
            available_managers=list(available.keys()),
        )
        return None

    def _build_package_manager_command(
        self, manager_name: str, package_name: str
    ) -> list[str] | None:
        """Build command for executing via package manager.

        Args:
            manager_name: Name of the package manager
            package_name: Package to execute

        Returns:
            Command list or None if manager not configured
        """
        commands = {
            "bunx": ["bunx", package_name],
            "pnpm": ["pnpm", "dlx", package_name],
            "npx": ["npx", "--yes", package_name],
        }
        return commands.get(manager_name)

    def _get_common_paths(self, binary_name: str) -> list[Path]:
        """Get common installation paths for a binary.

        Args:
            binary_name: Name of the binary

        Returns:
            List of paths to check
        """
        paths = [
            # User-specific locations
            Path.home() / ".cache" / ".bun" / "bin" / binary_name,
            Path.home() / ".local" / "bin" / binary_name,
            Path.home() / ".local" / "share" / "nvim" / "mason" / "bin" / binary_name,
            Path.home() / ".npm-global" / "bin" / binary_name,
            Path.home() / "bin" / binary_name,
            # System locations
            Path("/usr/local/bin") / binary_name,
            Path("/usr/bin") / binary_name,
            Path("/opt/homebrew/bin") / binary_name,  # macOS ARM
            # Node/npm locations
            Path.home()
            / ".nvm"
            / "versions"
            / "node"
            / "default"
            / "bin"
            / binary_name,
            Path.home() / ".volta" / "bin" / binary_name,
        ]
        return paths

    def _get_available_managers(self) -> dict[str, bool]:
        """Get available package managers on the system.

        Returns:
            Dictionary of manager names to availability status
        """
        if self._available_managers is not None:
            return self._available_managers

        self._available_managers = {}
        manager_info = {}

        for manager_name, config in self.PACKAGE_MANAGERS.items():
            check_cmd = config["check_cmd"]
            try:
                # Use subprocess.run with capture to check availability
                result = subprocess.run(
                    check_cmd,
                    capture_output=True,
                    text=True,
                    timeout=2,
                    check=False,
                )
                available = result.returncode == 0
                self._available_managers[manager_name] = available
                if available:
                    version = result.stdout.strip() if result.stdout else "unknown"
                    manager_info[manager_name] = version
            except (subprocess.TimeoutExpired, FileNotFoundError):
                self._available_managers[manager_name] = False

        # Log all available managers in one consolidated message
        if manager_info:
            logger.debug(
                "package_managers_detected",
                managers=manager_info,
                count=len(manager_info),
            )

        return self._available_managers

    def get_available_package_managers(self) -> list[str]:
        """Get list of available package managers on the system.

        Returns:
            List of package manager names that are available (e.g., ['bunx', 'pnpm'])
        """
        available = self._get_available_managers()
        return [name for name, is_available in available.items() if is_available]

    def get_package_manager_info(self) -> dict[str, dict[str, str | bool | int]]:
        """Get detailed information about package managers.

        Returns:
            Dictionary with package manager info including availability and priority
        """
        available = self._get_available_managers()
        info: dict[str, dict[str, str | bool | int]] = {}

        for name, config in self.PACKAGE_MANAGERS.items():
            exec_cmd = config.get("exec_cmd", name)
            info[name] = {
                "available": bool(available.get(name, False)),
                "priority": int(config["priority"]),
                "check_command": str(" ".join(config["check_cmd"])),
                "exec_command": str(exec_cmd if exec_cmd is not None else name),
            }

        return info

    def get_cli_info(
        self,
        binary_name: str,
        package_name: str | None = None,
        version: str | None = None,
    ) -> CLIInfo:
        """Get comprehensive CLI information in common format.

        Args:
            binary_name: Name of the binary to find
            package_name: NPM package name if different from binary name
            version: Optional version string (if known)

        Returns:
            CLIInfo dictionary with structured information
        """
        result = self.find_binary(binary_name, package_name)

        if not result:
            return CLIInfo(
                name=binary_name,
                version=version,
                source="unknown",
                path=None,
                command=[],
                package_manager=None,
                is_available=False,
            )

        # Determine source and path
        if result.is_direct:
            source = "path"
            path = result.command[0] if result.command else None
        else:
            source = "package_manager"
            path = None

        return CLIInfo(
            name=binary_name,
            version=version,
            source=source,
            path=path,
            command=result.command,
            package_manager=result.package_manager,
            is_available=True,
        )

    def clear_cache(self) -> None:
        """Clear all caches."""
        # Reset the available managers cache
        self._available_managers = None

    @classmethod
    def from_settings(cls, settings: "Settings") -> "BinaryResolver":
        """Create a BinaryResolver from application settings.

        Args:
            settings: Application settings

        Returns:
            Configured BinaryResolver instance
        """
        return cls(
            fallback_enabled=settings.binary.fallback_enabled,
            package_manager_only=settings.binary.package_manager_only,
            preferred_package_manager=settings.binary.preferred_package_manager,
            package_manager_priority=settings.binary.package_manager_priority,
        )


# Global instance for convenience
_default_resolver = BinaryResolver()


def find_binary_with_fallback(
    binary_name: str,
    package_name: str | None = None,
    fallback_enabled: bool = True,
) -> list[str] | None:
    """Convenience function to find a binary with package manager fallback.

    Args:
        binary_name: Name of the binary to find. Can be:
            - Simple binary name (e.g., "claude")
            - Full package name (e.g., "@anthropic-ai/claude-code")
        package_name: NPM package name if different from binary name
        fallback_enabled: Whether to use package manager fallback

    Returns:
        Command list to execute the binary, or None if not found
    """
    resolver = BinaryResolver(fallback_enabled=fallback_enabled)
    result = resolver.find_binary(binary_name, package_name)
    return result.command if result else None


def is_package_manager_command(command: list[str]) -> bool:
    """Check if a command uses a package manager.

    Args:
        command: Command list to check

    Returns:
        True if command uses a package manager
    """
    if not command:
        return False
    first_cmd = Path(command[0]).name
    return first_cmd in ["npx", "bunx", "pnpm"]


def get_available_package_managers() -> list[str]:
    """Convenience function to get available package managers using default resolver.

    Returns:
        List of package manager names that are available
    """
    return _default_resolver.get_available_package_managers()


def get_package_manager_info() -> dict[str, dict[str, str | bool | int]]:
    """Convenience function to get package manager info using default resolver.

    Returns:
        Dictionary with package manager info including availability and priority
    """
    return _default_resolver.get_package_manager_info()
