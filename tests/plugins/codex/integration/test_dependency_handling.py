"""Integration tests covering plugin dependency handling during discovery."""

from __future__ import annotations

from importlib import metadata
from pathlib import Path
from typing import Any

import pytest
from pytest import MonkeyPatch

from ccproxy.config.core import LoggingSettings
from ccproxy.config.settings import Settings
from ccproxy.core.plugins.discovery import discover_and_load_plugins
from ccproxy.core.plugins.factories import PluginRegistry
from ccproxy.core.plugins.interfaces import factory_type_name
from ccproxy.core.status_report import collect_plugin_snapshot


pytestmark = [pytest.mark.integration, pytest.mark.codex]


ROOT_DIR = Path(__file__).resolve().parents[4]
PLUGIN_DIR = ROOT_DIR / "ccproxy" / "plugins"


@pytest.fixture(scope="module", autouse=True)
def _stub_cli_detection() -> Any:
    """Short-circuit Codex CLI detection to avoid expensive probing."""

    from ccproxy.models.detection import DetectedHeaders, DetectedPrompts
    from ccproxy.plugins.codex.models import CodexCacheData
    from ccproxy.services.cli_detection import CLIDetectionResult
    from ccproxy.utils.binary_resolver import CLIInfo

    monkeypatch = MonkeyPatch()

    class _NoopHookThreadManager:
        """Minimal background hook thread manager that never spawns threads."""

        def start(self) -> None:  # pragma: no cover
            return

        def stop(self, timeout: float = 5.0) -> None:  # pragma: no cover
            return

        def emit_async(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover
            return

    def fake_cli_info(self: Any, binary_name: str) -> CLIInfo:  # pragma: no cover
        return CLIInfo(
            name=binary_name,
            version="test",
            source="fallback",
            path=None,
            command=[],
            package_manager=None,
            is_available=False,
        )

    async def fake_detect_cli(
        self: Any,
        binary_name: str,
        *args: Any,
        **kwargs: Any,
    ) -> CLIDetectionResult:  # pragma: no cover
        return CLIDetectionResult(
            name=binary_name,
            version="test",
            command=[],
            is_available=False,
            source="fallback",
            package_manager=None,
            cached=True,
            fallback_data={"version": "test"},
        )

    async def fake_initialize_detection(
        self: Any,
    ) -> CodexCacheData:  # pragma: no cover
        data = CodexCacheData(
            codex_version="test",
            headers=DetectedHeaders(),
            prompts=DetectedPrompts(),
        )
        self._cached_data = data
        return data

    monkeypatch.setattr(
        "ccproxy.services.cli_detection.CLIDetectionService.get_cli_info",
        fake_cli_info,
        raising=True,
    )
    monkeypatch.setattr(
        "ccproxy.services.cli_detection.CLIDetectionService.detect_cli",
        fake_detect_cli,
        raising=True,
    )
    monkeypatch.setattr(
        "ccproxy.plugins.codex.detection_service.CodexDetectionService.initialize_detection",
        fake_initialize_detection,
        raising=True,
    )
    monkeypatch.setattr(
        "ccproxy.services.factories.ConcreteServiceFactory.create_background_hook_thread_manager",
        lambda self: _NoopHookThreadManager(),
        raising=True,
    )

    try:
        yield
    finally:
        monkeypatch.undo()


def _available_plugins() -> set[str]:
    """Return filesystem plugin package names."""

    if not PLUGIN_DIR.exists():
        return set()
    return {
        entry.name
        for entry in PLUGIN_DIR.iterdir()
        if entry.is_dir() and (entry / "plugin.py").exists()
    }


def _entry_point_plugins() -> set[str]:
    """Return plugin names advertised via package entry points."""

    groups = metadata.entry_points()
    if hasattr(groups, "select"):
        items = groups.select(group="ccproxy.plugins")
    else:  # pragma: no cover - legacy importlib.metadata behaviour
        items = groups.get("ccproxy.plugins", [])
    return {ep.name for ep in items}


def _make_settings(
    *,
    plugin_configs: dict[str, dict[str, Any]],
    enabled_plugins: list[str] | None = None,
    disabled_plugins: list[str] | None = None,
) -> Settings:
    """Construct Settings limiting active plugins to the supplied configs."""

    available = _available_plugins()
    explicit_disabled = set(disabled_plugins or [])
    requested = set(plugin_configs.keys())

    if enabled_plugins is not None:
        allowed = set(enabled_plugins)
        explicit_disabled.update(name for name in available if name not in allowed)
    else:
        explicit_disabled.update(name for name in available if name not in requested)

    normalized_configs = dict(plugin_configs)
    if "duckdb_storage" not in normalized_configs:
        normalized_configs["duckdb_storage"] = {"enabled": False}
        explicit_disabled.add("duckdb_storage")

    final_disabled: list[str] | None
    if disabled_plugins is not None:
        final_disabled = disabled_plugins
    else:
        final_disabled = sorted(explicit_disabled) if explicit_disabled else None

    return Settings(
        enable_plugins=True,
        plugins_disable_local_discovery=False,
        enabled_plugins=enabled_plugins,
        disabled_plugins=final_disabled,
        plugins=normalized_configs,
        logging=LoggingSettings(
            **{
                "level": "DEBUG",
                "verbose_api": False,
            }
        ),
    )


async def _get_plugin_status(settings: Settings) -> dict[str, Any]:
    """Return a lightweight approximation of /plugins/status for the given settings."""

    plugin_factories = discover_and_load_plugins(settings)
    registry = PluginRegistry()
    for factory in plugin_factories.values():
        registry.register_factory(factory)

    init_order = registry.resolve_dependencies(settings)

    plugins: list[dict[str, Any]] = []
    for name, factory in sorted(plugin_factories.items()):
        manifest = factory.get_manifest()
        plugins.append(
            {
                "name": name,
                "version": manifest.version,
                "type": factory_type_name(factory),
                "provides": list(manifest.provides),
                "requires": list(manifest.requires),
                "optional_requires": list(manifest.optional_requires),
                "initialized": name in init_order,
            }
        )

    return {
        "initialization_order": init_order,
        "services": {},
        "plugins": plugins,
    }


def _find_plugin_entry(payload: dict[str, Any], name: str) -> dict[str, Any] | None:
    for entry in payload.get("plugins", []):
        if entry.get("name") == name:
            return entry
    return None


@pytest.mark.asyncio(loop_scope="module")
async def test_codex_skipped_when_dependency_disabled_by_config() -> None:
    """Codex registers but does not initialize when oauth_codex config disables it."""

    settings = _make_settings(
        plugin_configs={
            "codex": {"enabled": True},
            "oauth_codex": {"enabled": False},
        }
    )

    payload = await _get_plugin_status(settings)
    codex_entry = _find_plugin_entry(payload, "codex")

    assert codex_entry is not None
    assert codex_entry["initialized"] is False
    assert "oauth_codex" not in {entry["name"] for entry in payload.get("plugins", [])}
    assert "codex" not in payload.get("initialization_order", [])


@pytest.mark.asyncio(loop_scope="module")
async def test_codex_skipped_when_dependency_not_whitelisted() -> None:
    """enabled_plugins whitelist excludes oauth_codex so codex dependency is missing."""

    settings = _make_settings(
        plugin_configs={"codex": {"enabled": True}},
        enabled_plugins=["codex"],
    )

    payload = await _get_plugin_status(settings)
    codex_entry = _find_plugin_entry(payload, "codex")

    assert codex_entry is not None
    assert codex_entry["initialized"] is False
    assert {entry["name"] for entry in payload.get("plugins", [])} == {"codex"}
    assert "codex" not in payload.get("initialization_order", [])
    assert payload.get("services", {}) == {}


@pytest.mark.asyncio(loop_scope="module")
async def test_codex_initializes_with_dependency_enabled() -> None:
    """When both codex and oauth_codex participate, codex should initialize."""

    settings = _make_settings(
        plugin_configs={
            "codex": {"enabled": True},
            "oauth_codex": {"enabled": True},
        },
        enabled_plugins=["codex", "oauth_codex"],
    )

    payload = await _get_plugin_status(settings)

    codex_entry = _find_plugin_entry(payload, "codex")
    oauth_entry = _find_plugin_entry(payload, "oauth_codex")

    assert codex_entry is not None and codex_entry["initialized"] is True
    assert oauth_entry is not None and oauth_entry["initialized"] is True
    assert "codex" in payload.get("initialization_order", [])
    assert "oauth_codex" in payload.get("initialization_order", [])


@pytest.mark.asyncio(loop_scope="module")
async def test_codex_removed_when_plugin_config_disables_it() -> None:
    """plugins.codex.enabled=False removes codex even if dependency is active."""

    settings = _make_settings(
        plugin_configs={
            "codex": {"enabled": False},
            "oauth_codex": {"enabled": True},
        }
    )

    payload = await _get_plugin_status(settings)

    assert _find_plugin_entry(payload, "codex") is None
    oauth_entry = _find_plugin_entry(payload, "oauth_codex")
    assert oauth_entry is not None and oauth_entry["initialized"] is True


@pytest.mark.asyncio(loop_scope="module")
async def test_enabled_plugins_whitelist_is_respected() -> None:
    """Only explicitly whitelisted plugins load when enabled_plugins is set."""

    settings = _make_settings(
        plugin_configs={},
        enabled_plugins=["oauth_codex"],
    )

    payload = await _get_plugin_status(settings)

    assert {entry["name"] for entry in payload.get("plugins", [])} == {"oauth_codex"}
    oauth_entry = _find_plugin_entry(payload, "oauth_codex")
    assert oauth_entry is not None


@pytest.mark.asyncio(loop_scope="module")
async def test_disabled_plugins_blacklist_is_respected() -> None:
    """All plugins except the disabled ones remain available when blacklist is set."""

    available = _available_plugins()
    entry_point_plugins = _entry_point_plugins()
    target_disable = {"codex", "oauth_codex"}

    settings = _make_settings(
        plugin_configs={},
        disabled_plugins=list(target_disable),
    )

    # Using the status snapshot avoids booting the full app while still exercising discovery filtering.
    snapshot = collect_plugin_snapshot(settings)
    loaded = {info.name for info in snapshot.enabled_plugins}

    assert target_disable.isdisjoint(loaded)
    expected = (available & entry_point_plugins) - target_disable - {"duckdb_storage"}
    assert loaded.intersection(available) == expected
