"""API route coverage for /plugins/status."""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from ccproxy.api.app import create_app
from ccproxy.api.bootstrap import create_service_container
from ccproxy.config import LoggingSettings, Settings
from ccproxy.core.plugins.declaration import PluginManifest
from ccproxy.core.plugins.factories import PluginRegistry
from ccproxy.core.plugins.middleware import MiddlewareManager


@contextmanager
def _status_client() -> TestClient:
    """Yield a client with plugins enabled and metrics surfaced."""

    registry = PluginRegistry()

    metrics_manifest = PluginManifest(
        name="metrics",
        version="1.0.0",
        description="Metrics system plugin",
    )
    metrics_manifest.is_provider = False
    metrics_manifest.provides = ["metrics"]
    metrics_manifest.requires = []
    metrics_manifest.optional_requires = []

    provider_manifest = PluginManifest(
        name="codex",
        version="1.0.0",
        description="Codex provider plugin",
    )
    provider_manifest.is_provider = True
    provider_manifest.provides = ["codex"]
    provider_manifest.requires = []
    provider_manifest.optional_requires = ["pricing"]

    class _StubFactory:
        def __init__(self, manifest: PluginManifest, plugin_type: str) -> None:
            self._manifest = manifest
            self._plugin_type = plugin_type

        def get_manifest(self) -> PluginManifest:
            return self._manifest

    registry.factories = {
        "metrics": _StubFactory(metrics_manifest, "system"),
        "codex": _StubFactory(provider_manifest, "provider"),
    }
    registry.runtimes = {
        "metrics": SimpleNamespace(initialized=True),
        "codex": SimpleNamespace(initialized=True),
    }
    registry.initialization_order = ["metrics", "codex"]
    registry._service_providers = {"metrics": "metrics"}

    def _factory_type_name(factory: Any) -> str:
        return getattr(factory, "_plugin_type", "plugin")

    def _fake_load_plugin_system(
        _: Settings,
    ) -> tuple[PluginRegistry, MiddlewareManager]:
        return registry, MiddlewareManager()

    async def _fake_initialize(app: Any, *_: Any, **__: Any) -> None:
        app.state.plugin_registry = registry

    async def _fake_shutdown(app: Any) -> None:
        app.state.plugin_registry = registry

    settings = Settings(
        enable_plugins=True,
        plugins_disable_local_discovery=False,
        plugins={
            "metrics": {"enabled": True, "metrics_endpoint_enabled": True},
            "request_tracer": {
                "enabled": True,
                "json_logs_enabled": False,
                "raw_http_enabled": False,
            },
        },
        logging=LoggingSettings(
            level="ERROR",
            verbose_api=False,
        ),
    )

    with patch("ccproxy.api.app.load_plugin_system", _fake_load_plugin_system):
        container = create_service_container(settings)
        app = create_app(container)

    with (
        patch("ccproxy.core.plugins.factory_type_name", _factory_type_name),
        patch("ccproxy.api.app.initialize_plugins_startup", _fake_initialize),
        patch("ccproxy.api.app.shutdown_plugins", _fake_shutdown),
        TestClient(app) as client,
    ):
        yield client


def _fetch_status_payload() -> dict[str, Any]:
    """Return the JSON payload for /plugins/status."""

    with _status_client() as client:
        response = client.get("/plugins/status")

    assert response.status_code == 200
    return response.json()


@pytest.fixture(scope="module")
def status_payload() -> dict[str, Any]:
    """Return /plugins/status payload once per module."""

    return _fetch_status_payload()


def test_plugins_status_types(status_payload: dict[str, Any]) -> None:
    """Plugins status report surfaces system and provider plugin types."""

    names_to_types = {
        entry["name"]: entry["type"] for entry in status_payload["plugins"]
    }

    assert "metrics" in names_to_types
    assert names_to_types["metrics"] == "system"

    provider_candidates = {"claude_api", "codex"}
    providers = provider_candidates & names_to_types.keys()
    assert providers, "expected at least one provider plugin"
    for provider in providers:
        assert names_to_types[provider] in {"provider", "auth_provider"}
        matching_entry = next(
            entry for entry in status_payload["plugins"] if entry["name"] == provider
        )
        assert matching_entry["initialized"] is True


def test_plugins_status_initialization_order_matches_plugins(
    status_payload: dict[str, Any],
) -> None:
    """Initialization order tracks the plugins reported by the endpoint."""

    init_order = status_payload["initialization_order"]
    plugin_names = {entry["name"] for entry in status_payload["plugins"]}

    assert init_order, "expected initialization order to be populated"
    assert set(init_order).issubset(plugin_names)
    assert "metrics" in init_order


def test_plugins_status_provider_metadata(status_payload: dict[str, Any]) -> None:
    """Provider plugins expose optional requirements and initialization flags."""

    providers = [
        entry
        for entry in status_payload["plugins"]
        if entry["type"] in {"provider", "auth_provider"}
    ]

    assert providers, "expected provider plugins to be present"

    optional_require_lists = [set(entry["optional_requires"]) for entry in providers]
    assert any("pricing" in opts for opts in optional_require_lists)
    assert all(entry["initialized"] is True for entry in providers)
