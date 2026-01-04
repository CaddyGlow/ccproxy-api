"""Helpers for collecting status information used across interfaces."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from ccproxy.config.settings import Settings
from ccproxy.core.logging import get_logger
from ccproxy.core.plugins.discovery import discover_and_load_plugins


logger = get_logger(__name__)


@dataclass(frozen=True)
class DirectoryStatus:
    """Represents availability of a directory."""

    path: Path
    exists: bool


@dataclass(frozen=True)
class SystemSnapshot:
    """Summary of system-level settings."""

    host: str
    port: int
    log_level: str
    auth_token_configured: bool
    plugins_enabled: bool
    plugin_directories: tuple[DirectoryStatus, ...]


@dataclass(frozen=True)
class ConfigSource:
    """Represents a potential configuration file location."""

    path: Path
    exists: bool


@dataclass(frozen=True)
class ConfigSnapshot:
    """Collected configuration source information."""

    sources: tuple[ConfigSource, ...]


@dataclass(frozen=True)
class PluginInfo:
    """Represents the status of an individual plugin."""

    name: str
    state: Literal["enabled", "error"]
    version: str | None
    description: str | None
    error: str | None = None


@dataclass(frozen=True)
class PluginSnapshot:
    """Collection of plugin discovery results."""

    plugin_system_enabled: bool
    enabled_plugins: tuple[PluginInfo, ...]
    disabled_plugins: tuple[str, ...]
    configuration_notes: tuple[str, ...]

    @property
    def enabled_count(self) -> int:
        return len(self.enabled_plugins)

    @property
    def disabled_count(self) -> int:
        return len(self.disabled_plugins)

    @property
    def total_count(self) -> int:
        return self.enabled_count + self.disabled_count


def collect_system_snapshot(settings: Settings) -> SystemSnapshot:
    """Build a system snapshot from settings."""
    directories: list[DirectoryStatus] = []
    for directory in settings.plugin_discovery.directories:
        dir_path = Path(directory)
        directories.append(DirectoryStatus(path=dir_path, exists=dir_path.exists()))

    return SystemSnapshot(
        host=str(settings.server.host),
        port=int(settings.server.port),
        log_level=settings.logging.level.upper(),
        auth_token_configured=bool(settings.security.auth_token),
        plugins_enabled=bool(settings.enable_plugins),
        plugin_directories=tuple(directories),
    )


def collect_config_snapshot(*, cwd: Path | None = None) -> ConfigSnapshot:
    """Inspect common configuration locations relative to the current working directory."""
    effective_cwd = cwd or Path.cwd()
    candidates: Iterable[Path] = (
        effective_cwd / ".ccproxy.toml",
        effective_cwd / "ccproxy.toml",
        Path.home() / ".ccproxy" / "config.toml",
    )

    sources = tuple(
        ConfigSource(path=path, exists=path.exists()) for path in candidates
    )
    return ConfigSnapshot(sources=sources)


def collect_plugin_snapshot(settings: Settings) -> PluginSnapshot:
    """Discover plugins and report basic status information."""
    if not settings.enable_plugins:
        notes = _collect_plugin_configuration_notes(settings)
        return PluginSnapshot(
            plugin_system_enabled=False,
            enabled_plugins=(),
            disabled_plugins=(),
            configuration_notes=notes,
        )

    plugin_infos: list[PluginInfo] = []
    try:
        plugin_factories = discover_and_load_plugins(settings)
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.error("plugin_discovery_failed", error=str(exc), exc_info=exc)
        return PluginSnapshot(
            plugin_system_enabled=True,
            enabled_plugins=(),
            disabled_plugins=(),
            configuration_notes=_collect_plugin_configuration_notes(settings),
        )

    for name, factory in sorted(plugin_factories.items()):
        try:
            manifest = factory.get_manifest()
            plugin_infos.append(
                PluginInfo(
                    name=name,
                    state="enabled",
                    version=manifest.version,
                    description=manifest.description,
                )
            )
        except Exception as exc:
            error_text = str(exc)
            logger.error(
                "plugin_manifest_failed", plugin=name, error=error_text, exc_info=exc
            )
            plugin_infos.append(
                PluginInfo(
                    name=name,
                    state="error",
                    version=None,
                    description=None,
                    error=error_text,
                )
            )

    disabled_plugins = tuple(
        sorted(_find_disabled_plugins(settings, set(plugin_factories.keys())))
    )

    return PluginSnapshot(
        plugin_system_enabled=True,
        enabled_plugins=tuple(plugin_infos),
        disabled_plugins=disabled_plugins,
        configuration_notes=_collect_plugin_configuration_notes(settings),
    )


def _collect_plugin_configuration_notes(settings: Settings) -> tuple[str, ...]:
    notes: list[str] = []
    if settings.plugins_disable_local_discovery:
        notes.append("Local discovery disabled")
    if settings.disabled_plugins:
        notes.append(f"Explicitly disabled: {len(settings.disabled_plugins)}")
    if settings.enabled_plugins:
        notes.append(f"Allow-list active: {len(settings.enabled_plugins)} allowed")
    return tuple(notes)


def _find_disabled_plugins(settings: Settings, enabled_plugins: set[str]) -> set[str]:
    """Find plugins that exist but are disabled in the configuration."""
    disabled: set[str] = set()

    if settings.plugins_disable_local_discovery:
        return disabled

    for plugin_dir_path in settings.plugin_discovery.directories:
        plugin_dir = Path(plugin_dir_path)
        if not plugin_dir.exists():
            continue

        for item in plugin_dir.iterdir():
            if not item.is_dir() or item.name.startswith("_"):
                continue
            plugin_file = item / "plugin.py"
            if not plugin_file.exists():
                continue
            if item.name not in enabled_plugins:
                disabled.add(item.name)

    return disabled
