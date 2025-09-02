# Plugin Authoring Guide

This guide shows how to build CCProxy plugins that integrate cleanly with the core and other plugins. It covers plugin types, structure, configuration, discovery, and best practices.

## Plugin Types

- Auth Provider Plugin (factory: `AuthProviderPluginFactory`)
  - Provides OAuth login/refresh and token management. Does not proxy provider requests directly.
  - Example responsibilities: expose `/oauth/*` routes, manage credentials, emit auth hooks.
- Provider Plugin (factory: `ProviderPluginFactory`)
  - Proxies API calls to a provider (e.g., Anthropic, OpenAI Codex). Owns adapter, detection, and credentials.
  - Example responsibilities: register API routes, provide services (e.g., pricing), add hooks and tasks.
- System Plugin (factory: `SystemPluginFactory`)
  - Adds system-wide features like metrics, logging, tracing, permissions, pricing. No external provider proxy.

Use `GET /api/plugins/status` to see each plugin’s `type` as `auth_provider`, `provider`, or `system`.

## Minimal Structure

- Manifest (static declaration): `PluginManifest`
  - `name`, `version`, `description`
  - `is_provider`: for provider and auth provider plugins
  - `provides`: service names this plugin provides (e.g., `pricing`)
  - `requires`: required services (hard fail if missing)
  - `optional_requires`: optional services
  - `middleware`: ordered by priority (see `MiddlewareLayer`)
  - `routes`: one or more `APIRouter`s and prefixes
  - `tasks`: scheduled jobs registered with the scheduler
  - `hooks`: event subscribers
  - `config_class`: Pydantic model for plugin config (optional)

- Runtime: subclass of `SystemPluginRuntime` or `ProviderPluginRuntime`
  - Initialize from `PluginContext` (injected by core): `settings`, `http_client`, `logger`, `scheduler`, `plugin_registry`, `request_tracer`, `streaming_handler`, `config`, etc.
  - Register hooks/services/routes as needed.
  - Implement `health_check`, `validate`, and `shutdown` when applicable.

- Factory: subclass of the corresponding factory
  - Build `PluginManifest`
  - Create runtime
  - For providers, create `adapter`, `detection_service`, `credentials_manager` if applicable

## Discovery

Plugins are discovered from two sources and merged:
- Local `plugins/` directory: any subfolder with a `plugin.py` exporting `factory` (a `PluginFactory`) is loaded (filesystem discovery).
- Installed entry points: Python packages that declare an entry under `ccproxy.plugins` providing a `PluginFactory` or a callable returning one.

Local filesystem plugins take precedence over entry points on name conflicts. To disable filesystem discovery and load plugins only from entry points, set `plugins_disable_local_discovery = true` in your `.ccproxy.toml` or export `PLUGINS_DISABLE_LOCAL_DISCOVERY=true`.

### Declaring Entry Points (pyproject.toml)

```
[project.entry-points."ccproxy.plugins"]
my_plugin = "my_package.my_plugin:factory"
# or a callable that returns a PluginFactory
other_plugin = "my_package.other:create_factory"
```

## Configuration

- Provide a `config_class` (Pydantic BaseModel) on the manifest.
- Core populates `PluginContext["config"]` with validated settings from:
  - Defaults < TOML config < Env (`PLUGINS__{NAME}__FIELD`) < CLI overrides
- Example env nest: `PLUGINS__METRICS__ENABLED=true`.

## Routes & Middleware

- Add routes via `RouteSpec(router=..., prefix=..., tags=[...])`. Core mounts them with plugin-specific tags.
- Add middleware via `MiddlewareSpec(middleware_class, priority=MiddlewareLayer.OBSERVABILITY, kwargs={...})`.
- Keep handlers fast and non-blocking; use async I/O and avoid CPU-heavy work in request path.

## Hooks

- Subscribe to events with `HookSpec(hook_class=..., kwargs={...})`.
- Common events are in `HookEvent`, e.g., `REQUEST_STARTED`, `REQUEST_COMPLETED`, `PROVIDER_REQUEST_SENT`, `PROVIDER_STREAM_*`.
- Use hook priorities consistently. Avoid raising from hooks; log and continue.

## Services

- Provide services by calling `registry.register_service(name, instance, provider_plugin=name)` from runtime.
- Consume services by calling `registry.get_service(name, ExpectedType)`; returns `None` if absent.
- Avoid globals; rely on the plugin registry and container-managed clients.

## Health & Status

- Implement `health_check()` in runtime to return IETF-style health.
- Check `/api/plugins/status` to inspect:
  - `initialization_order` (dependency order)
  - `services` map (service -> provider)
  - per-plugin summary (name, version, type, provides/requires, initialized)

## Logging & Security

- Use structured logs via `get_plugin_logger()` or context-provided logger.
- Do not log secrets or sensitive request bodies. Mask tokens in logs.
- Respect repository logging conventions and levels.

## Testing

- Use `create_app(Settings(...))` + `initialize_plugins_startup` to bootstrap.
- Prefer `httpx.ASGITransport` for tests (no server needed).
- For timing-sensitive code, keep tests deterministic and avoid global registries.

## Example Skeleton (Provider)

```python
# plugin.py (inside plugins/my_provider)
from ccproxy.plugins import PluginManifest, ProviderPluginRuntime, ProviderPluginFactory
from pydantic import BaseModel
from fastapi import APIRouter

class MyProviderConfig(BaseModel):
    enabled: bool = True

router = APIRouter()

class MyProviderRuntime(ProviderPluginRuntime):
    async def _on_initialize(self) -> None:
        if self.context.get("config").enabled:
            # Register routes, hooks, services as needed
            pass

class MyProviderFactory(ProviderPluginFactory):
    def __init__(self) -> None:
        manifest = PluginManifest(
            name="my_provider",
            version="1.0.0",
            is_provider=True,
            config_class=MyProviderConfig,
        )
        super().__init__(manifest)

    def create_context(self, core_services):
        context = super().create_context(core_services)
        # If you have routes:
        # context["router"] = router
        return context

# Export factory for discovery
factory = MyProviderFactory()
```

## Publishing

- Package your plugin and declare the `ccproxy.plugins` entry point in `pyproject.toml`.
- Version it semantically and document configuration fields and routes.

## Best Practices

- Keep adapters and detection logic small and focused.
- Use the container-managed HTTP client; never create your own long-lived clients.
- Avoid global singletons; favor dependency injection via the container and plugin registry.
- Ensure hooks and tasks fail gracefully; log errors without breaking the app.
- Write minimal, clear tests; keep integration tests fast.

---

See also:
- `docs/PLUGIN_SYSTEM_DOCUMENTATION.md` for more on the plugin runtime model
- Metrics/logging plugins (e.g., `plugins/metrics`, `plugins/analytics`) for observability patterns
- `GET /api/plugins/status` for runtime inspection
