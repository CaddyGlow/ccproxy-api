# Plugin Authoring Guide

This guide shows how to build CCProxy plugins that integrate cleanly with the core and other plugins. It covers plugin types, structure, configuration, discovery, and best practices.

## Plugin Types

### Auth Provider Plugin (factory: `AuthProviderPluginFactory`)
- Provides standalone OAuth authentication without proxying requests
- **Key Components**: OAuth provider, token manager, secure storage, CLI integration
- **Example**: `oauth_claude` - provides Claude OAuth without API proxying
- **Pattern**: Extends `AuthProviderPluginRuntime`, registers OAuth provider in registry
- **CLI Safe**: `cli_safe = True` (safe for CLI usage)

### Provider Plugin (factory: `BaseProviderPluginFactory`)
- Proxies API requests to external providers with full request/response lifecycle
- **Key Components**: HTTP adapter (delegation pattern), detection service, credentials manager, transformers, format adapters, hooks
- **Example**: `codex` - proxies OpenAI Codex API with format conversion and streaming metrics
- **Pattern**: Class-based configuration, declarative format adapters, streaming support
- **CLI Safe**: `cli_safe = False` (heavy provider - not for CLI)

### System Plugin (factory: `SystemPluginFactory`)
- Adds system-wide functionality using hooks and services
- **Key Components**: Hooks for request lifecycle, services (analytics, pricing), routes for management APIs
- **Example**: `access_log` - hook-based request logging, `analytics` - provides query/ingest services
- **Pattern**: Hook-based architecture, service registration, background tasks

Use `GET /plugins/status` to see each plugin's `type` as `auth_provider`, `provider`, or `system`, along with initialization status, dependencies, and provided services.

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
- Check `/plugins/status` to inspect:
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

## Complete Plugin Examples

### Provider Plugin with Format Conversion

```python
# plugin.py (inside ccproxy/plugins/my_provider)
from ccproxy.core.plugins import (
    BaseProviderPluginFactory,
    ProviderPluginRuntime,
    PluginManifest,
    FormatAdapterSpec
)
from pydantic import BaseModel, Field
from fastapi import APIRouter

# Configuration
class MyProviderConfig(BaseModel):
    enabled: bool = Field(default=True)
    base_url: str = Field(default="https://api.example.com")
    supports_streaming: bool = Field(default=True)

# Router
router = APIRouter()

@router.post("/responses")
async def create_response(request: dict):
    # Provider-specific endpoint
    pass

# Runtime
class MyProviderRuntime(ProviderPluginRuntime):
    async def _on_initialize(self) -> None:
        """Initialize with format adapters and hooks."""
        config = self.context.get(MyProviderConfig)

        # Call parent (creates adapter, detection service)
        await super()._on_initialize()

        # Register streaming metrics hook
        if config.supports_streaming:
            await self._register_streaming_hook()

        logger.info("my_provider_initialized", enabled=config.enabled)

    async def _register_streaming_hook(self) -> None:
        """Register provider-specific streaming metrics hook."""
        hook_registry = self.context.get(HookRegistry)
        if hook_registry:
            hook = MyStreamingHook()
            hook_registry.register(hook)

# Factory with class-based configuration
class MyProviderFactory(BaseProviderPluginFactory):
    # Declarative configuration
    plugin_name = "my_provider"
    plugin_description = "My provider with streaming and format conversion"
    runtime_class = MyProviderRuntime
    adapter_class = MyProviderAdapter
    detection_service_class = MyDetectionService
    credentials_manager_class = MyCredentialsManager
    config_class = MyProviderConfig
    router = router
    route_prefix = "/api/my-provider"
    dependencies = ["oauth_my_provider"]
    optional_requires = ["pricing"]

    # Format adapter specifications
    format_adapters = [
        FormatAdapterSpec(
            from_format="openai",
            to_format="my_format",
            adapter_factory=lambda: MyFormatAdapter(),
            priority=40,
            description="OpenAI to My Provider conversion"
        )
    ]

# Export factory for discovery
factory = MyProviderFactory()
```

### System Plugin with Hooks and Services

```python
# plugin.py (inside ccproxy/plugins/my_system)
from ccproxy.core.plugins import (
    SystemPluginFactory,
    SystemPluginRuntime,
    PluginManifest,
    RouteSpec
)
from ccproxy.core.plugins.hooks import Hook, HookEvent, HookRegistry
from pydantic import BaseModel, Field
from fastapi import APIRouter

# Configuration
class MySystemConfig(BaseModel):
    enabled: bool = Field(default=True)
    buffer_size: int = Field(default=100)

# Routes
router = APIRouter()

@router.get("/status")
async def get_status():
    return {"status": "active"}

# Hook implementation
class MySystemHook(Hook):
    name = "my_system"
    events = [HookEvent.REQUEST_STARTED, HookEvent.REQUEST_COMPLETED]
    priority = 750

    def __init__(self, config: MySystemConfig):
        self.config = config
        self.buffer = []

    async def __call__(self, context: HookContext) -> None:
        if context.event == HookEvent.REQUEST_STARTED:
            # Process request start
            self._buffer_request_data(context.data)
        elif context.event == HookEvent.REQUEST_COMPLETED:
            # Process completion with metrics
            self._buffer_completion_data(context)

# Service implementation
class MySystemService:
    def __init__(self, config: MySystemConfig):
        self.config = config

    def process_data(self, data: dict) -> dict:
        # Service logic
        return data

# Runtime
class MySystemRuntime(SystemPluginRuntime):
    def __init__(self, manifest: PluginManifest):
        super().__init__(manifest)
        self.hook = None
        self.service = None
        self.config = None

    async def _on_initialize(self) -> None:
        """Initialize hooks and services."""
        if not self.context:
            raise RuntimeError("Context not set")

        # Get configuration
        config = self.context.get("config")
        if not isinstance(config, MySystemConfig):
            config = MySystemConfig()
        self.config = config

        if not config.enabled:
            return

        # Create and register hook
        self.hook = MySystemHook(config)
        hook_registry = self.context.get(HookRegistry)
        if hook_registry:
            hook_registry.register(self.hook)

        # Create and register service
        self.service = MySystemService(config)
        plugin_registry = self.context.get("plugin_registry")
        if plugin_registry:
            plugin_registry.register_service(
                "my_service", self.service, self.manifest.name
            )

        logger.info("my_system_initialized")

    async def _on_shutdown(self) -> None:
        """Cleanup resources."""
        if self.hook:
            # Hook cleanup
            await self.hook.close()

# Factory
class MySystemFactory(SystemPluginFactory):
    def __init__(self) -> None:
        manifest = PluginManifest(
            name="my_system",
            version="1.0.0",
            description="My system plugin with hooks and services",
            is_provider=False,
            config_class=MySystemConfig,
            provides=["my_service"],
            dependencies=["analytics"],
            routes=[
                RouteSpec(
                    router=router,
                    prefix="/my-system",
                    tags=["my-system"]
                )
            ]
        )
        super().__init__(manifest)

    def create_runtime(self) -> MySystemRuntime:
        return MySystemRuntime(self.manifest)

# Export factory
factory = MySystemFactory()
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
- `GET /plugins/status` for runtime inspection
