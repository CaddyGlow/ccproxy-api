# CCProxy Plugin System v2 Documentation

## Table of Contents
1. [Plugin System Overview](#plugin-system-overview)
2. [Architecture](#architecture)
3. [Plugin Types](#plugin-types)
4. [Core Components](#core-components)
5. [Plugin Lifecycle](#plugin-lifecycle)
6. [API Documentation](#api-documentation)
7. [Integration Guide](#integration-guide)
8. [Creating Plugins](#creating-plugins)
9. [Configuration](#configuration)
10. [Authoring Guide](#authoring-guide)

## Plugin System Overview

CCProxy uses a modern plugin system (v2) that provides a flexible, declarative architecture for extending the proxy server's functionality. The system supports three types of plugins:

- **Provider Plugins**: Proxy requests to external AI providers (Claude API, Claude SDK, Codex)
- **Auth Provider Plugins**: Provide OAuth authentication without proxying requests (OAuth Claude)
- **System Plugins**: Add functionality like logging, monitoring, analytics, and permissions

For a practical, end-to-end walkthrough on creating your own plugin (types, structure, config, routes, hooks, and publishing), see the Plugin Authoring Guide: `docs/PLUGIN_AUTHORING.md`.

### Key Features

- **Declarative Configuration**: Plugins declare their capabilities at import time
- **Lifecycle Management**: Proper initialization and shutdown phases
- **Dependency Resolution**: Automatic handling of inter-plugin dependencies
- **Component Support**: Middleware, routes, tasks, hooks, and auth commands
- **Type Safety**: Full type hints and protocol definitions

## Architecture

The plugin system follows a three-layer architecture:

```
┌─────────────────────────────────────────────────────┐
│                   Declaration Layer                  │
│  (PluginManifest, RouteSpec, MiddlewareSpec, etc.)  │
├─────────────────────────────────────────────────────┤
│                    Factory Layer                     │
│  (PluginFactory, PluginRegistry, Discovery)          │
├─────────────────────────────────────────────────────┤
│                    Runtime Layer                     │
│  (PluginRuntime, Context, Services)                  │
└─────────────────────────────────────────────────────┘
```

### Declaration Layer
Defines static plugin capabilities that can be determined at module import time.

### Factory Layer
Manages plugin creation and registration, bridging declaration and runtime.

### Runtime Layer
Handles plugin instances and their lifecycle after application startup.

## Plugin Types

### Provider Plugins

Provider plugins proxy requests to external API providers. They extend `BaseProviderPluginFactory` and `ProviderPluginRuntime` and include:

- **Adapter**: Handles request/response processing using HTTP delegation pattern
- **Detection Service**: Detects provider capabilities and CLI availability
- **Credentials Manager**: Manages authentication tokens and refresh logic
- **Transformers**: Transform requests/responses for protocol conversion
- **Format Adapters**: Convert between different API formats (OpenAI ↔ Anthropic)
- **Hooks**: Provider-specific event handling (e.g., streaming metrics)

Example providers: `claude_api`, `claude_sdk`, `codex`

### Auth Provider Plugins

Auth provider plugins provide standalone OAuth authentication without proxying requests. They extend `AuthProviderPluginFactory` and `AuthProviderPluginRuntime` and include:

- **OAuth Provider**: Implements OAuth flow (authorization, callback, token refresh)
- **Token Manager**: Manages credential storage and validation
- **Storage**: Secure credential persistence
- **CLI Integration**: Automatic CLI auth command registration

Example: `oauth_claude`

### System Plugins

System plugins add functionality without proxying to external providers. They extend `SystemPluginFactory` and `SystemPluginRuntime` and include:

- **Hooks**: Event-based request/response processing
- **Routes**: Additional API endpoints for analytics, logs, etc.
- **Services**: Shared services like analytics ingestion or pricing calculation
- **Background Tasks**: Scheduled operations

Example system plugins: `access_log`, `analytics`, `permissions`

## Core Components

### PluginManifest

The central declaration of a plugin's capabilities:

```python
@dataclass
class PluginManifest:
    # Basic metadata
    name: str                           # Unique plugin identifier
    version: str                        # Plugin version
    description: str = ""               # Plugin description
    dependencies: list[str] = field(default_factory=list)

    # Plugin type
    is_provider: bool = False           # True for provider plugins

    # Static specifications
    middleware: list[MiddlewareSpec] = field(default_factory=list)
    routes: list[RouteSpec] = field(default_factory=list)
    tasks: list[TaskSpec] = field(default_factory=list)
    hooks: list[HookSpec] = field(default_factory=list)
    auth_commands: list[AuthCommandSpec] = field(default_factory=list)

    # Configuration
    config_class: type[BaseModel] | None = None

    # OAuth support (provider plugins)
    oauth_provider_factory: Callable[[], OAuthProviderProtocol] | None = None
```

### PluginFactory

Abstract factory for creating plugin runtime instances:

```python
class PluginFactory(ABC):
    @abstractmethod
    def get_manifest(self) -> PluginManifest:
        """Get the plugin manifest."""

    @abstractmethod
    def create_runtime(self) -> BasePluginRuntime:
        """Create a runtime instance."""

    @abstractmethod
    def create_context(self, core_services: Any) -> PluginContext:
        """Create the context for plugin initialization."""
```

### PluginRuntime

Base runtime for all plugins:

```python
class BasePluginRuntime(PluginRuntimeProtocol):
    async def initialize(self, context: PluginContext) -> None:
        """Initialize with runtime context."""

    async def shutdown(self) -> None:
        """Cleanup on shutdown."""

    async def validate(self) -> bool:
        """Validate plugin is ready."""

    async def health_check(self) -> dict[str, Any]:
        """Perform health check."""
```

### PluginRegistry

Central registry managing all plugins:

```python
class PluginRegistry:
    def register_factory(self, factory: PluginFactory) -> None:
        """Register a plugin factory."""

    def resolve_dependencies(self) -> list[str]:
        """Resolve plugin dependencies."""

    async def initialize_all(self, core_services: Any) -> None:
        """Initialize all plugins in dependency order."""

    async def shutdown_all(self) -> None:
        """Shutdown all plugins in reverse order."""
```

### Component Specifications

#### MiddlewareSpec
```python
@dataclass
class MiddlewareSpec:
    middleware_class: type[BaseHTTPMiddleware]
    priority: int = MiddlewareLayer.APPLICATION
    kwargs: dict[str, Any] = field(default_factory=dict)
```

#### RouteSpec
```python
@dataclass
class RouteSpec:
    router: APIRouter
    prefix: str
    tags: list[str] = field(default_factory=list)
    dependencies: list[Any] = field(default_factory=list)
```

#### TaskSpec
```python
@dataclass
class TaskSpec:
    task_name: str
    task_type: str
    task_class: type[BaseScheduledTask]
    interval_seconds: float
    enabled: bool = True
    kwargs: dict[str, Any] = field(default_factory=dict)
```

## Plugin Lifecycle

### 1. Discovery Phase (App Creation)
- Plugins loaded via `load_plugin_system(settings)` (bundled + entry points)
- Plugin factories loaded and validated
- Dependencies resolved

### 2. Registration Phase (App Creation)
- Factories registered with PluginRegistry
- Manifests populated with configuration
- Middleware and routes collected

### 3. Application Phase (App Creation)
- Middleware applied to FastAPI app
- Routes registered with app
- Registry stored in app state

### 4. Initialization Phase (App Startup)
- Plugins initialized in dependency order
- Runtime instances created
- Services and adapters configured

### 5. Runtime Phase (App Running)
- Plugins handle requests
- Background tasks execute
- Health checks available

### 6. Shutdown Phase (App Shutdown)
- Plugins shutdown in reverse order
- Resources cleaned up
- Connections closed

## API Documentation

### Plugin Management Endpoints

All plugin management endpoints are prefixed with `/plugins`.

#### List Plugins
```http
GET /plugins
```

**Response:**
```json
{
  "plugins": [
    {
      "name": "claude_api",
      "type": "plugin",
      "status": "active",
      "version": "1.0.0"
    },
    {
      "name": "raw_http_logger",
      "type": "plugin",
      "status": "active",
      "version": "1.0.0"
    }
  ],
  "total": 2
}
```

#### Plugin Health Check
```http
GET /plugins/{plugin_name}/health
```

**Parameters:**
- `plugin_name` (path): Name of the plugin

**Response:**
```json
{
  "plugin": "claude_api",
  "status": "healthy",
  "adapter_loaded": true,
  "details": {
    "type": "provider",
    "initialized": true,
    "has_adapter": true,
    "has_detection": true,
    "has_credentials": true,
    "cli_version": "0.7.5",
    "cli_path": "/usr/local/bin/claude"
  }
}
```

#### Status
```http
GET /plugins/status
```

Returns manifests and initialization state for all loaded plugins.

## Integration Guide

### Application Integration

The plugin system integrates with the FastAPI application during the `create_app` function in `ccproxy/api/app.py`:

```python
def create_app(settings: Settings | None = None) -> FastAPI:
    # Phase 1: Discovery and Registration
    plugin_registry = PluginRegistry()
    middleware_manager = MiddlewareManager()

    if settings.enable_plugins:
        # Load plugin system via centralized loader
        plugin_registry, middleware_manager = load_plugin_system(settings)

        # Create context for manifest population
        manifest_services = ManifestPopulationServices(settings)

        # Populate manifests (context already created in loader in real code)
        for name, factory in plugin_registry.factories.items():
            factory.create_context(manifest_services)

        # Collect middleware from plugins
        for name, factory in plugin_registry.factories.items():
            manifest = factory.get_manifest()
            if manifest.middleware:
                middleware_manager.add_plugin_middleware(name, manifest.middleware)

        # Register plugin routes
        for name, factory in plugin_registry.factories.items():
            manifest = factory.get_manifest()
            for route_spec in manifest.routes:
                app.include_router(
                    route_spec.router,
                    prefix=route_spec.prefix,
                    tags=list(route_spec.tags)
                )

    # Store registry for runtime initialization
    app.state.plugin_registry = plugin_registry

    # Apply middleware
    setup_default_middleware(middleware_manager)
    middleware_manager.apply_to_app(app)

    return app
```

### Lifespan Integration

During application lifespan, plugins are initialized and shutdown:

```python
async def initialize_plugins_v2_startup(app: FastAPI, settings: Settings) -> None:
    """Initialize v2 plugins during startup."""
    if not settings.enable_plugins:
        return

    plugin_registry: PluginRegistry = app.state.plugin_registry

    # Get the service container created during app construction
    service_container = app.state.service_container

    # Create core services adapter
    core_services = CoreServicesAdapter(service_container)

    # Initialize all plugins
    await plugin_registry.initialize_all(core_services)

    # Note: The hook system (HookRegistry/HookManager) is created during app
    # startup and registered into the DI container. Plugins should obtain the
    # HookManager from the provided context or from the container rather than
    # creating their own instances.

async def shutdown_plugins_v2(app: FastAPI) -> None:
    """Shutdown v2 plugins."""
    if hasattr(app.state, "plugin_registry"):
        plugin_registry: PluginRegistry = app.state.plugin_registry
        await plugin_registry.shutdown_all()
```

## Creating Plugins

### Provider Plugin Example

```python
from ccproxy.core.plugins import (
    BaseProviderPluginFactory,
    PluginManifest,
    ProviderPluginRuntime,
    RouteSpec,
    FormatAdapterSpec
)

class MyProviderRuntime(ProviderPluginRuntime):
    async def _on_initialize(self) -> None:
        """Initialize the provider."""
        # Get configuration and services from context
        config = self.context.get(MyProviderConfig)

        # Call parent initialization
        await super()._on_initialize()

        # Provider-specific initialization
        logger.info("my_provider_initialized", enabled=config.enabled)

class MyProviderFactory(BaseProviderPluginFactory):
    # Class-based configuration
    plugin_name = "my_provider"
    plugin_description = "My provider plugin with format conversion"
    runtime_class = MyProviderRuntime
    adapter_class = MyProviderAdapter
    detection_service_class = MyDetectionService
    credentials_manager_class = MyCredentialsManager
    config_class = MyProviderConfig
    router = my_router
    route_prefix = "/api/my-provider"
    dependencies = ["oauth_my_provider"]
    optional_requires = ["pricing"]

    # Declarative format adapter specification
    format_adapters = [
        FormatAdapterSpec(
            from_format="openai",
            to_format="my_format",
            adapter_factory=lambda: MyFormatAdapter(),
            priority=50,
            description="OpenAI to My Provider format conversion"
        )
    ]

    def create_detection_service(self, context: PluginContext) -> MyDetectionService:
        settings = context.get(Settings)
        cli_service = context.get(CLIDetectionService)
        return MyDetectionService(settings, cli_service)

# Export factory instance
factory = MyProviderFactory()
```

### System Plugin Example (Hook-based)

```python
from ccproxy.core.plugins import (
    SystemPluginFactory,
    SystemPluginRuntime,
    PluginManifest,
    RouteSpec
)
from ccproxy.core.plugins.hooks import HookRegistry

class MySystemRuntime(SystemPluginRuntime):
    def __init__(self, manifest: PluginManifest):
        super().__init__(manifest)
        self.hook = None
        self.config = None

    async def _on_initialize(self) -> None:
        """Initialize the system plugin."""
        if not self.context:
            raise RuntimeError("Context not set")

        # Get configuration
        config = self.context.get("config")
        if not isinstance(config, MySystemConfig):
            config = MySystemConfig()  # Use defaults
        self.config = config

        if not config.enabled:
            return

        # Create and register hook
        self.hook = MySystemHook(config)

        # Get hook registry from context
        hook_registry = self.context.get(HookRegistry)
        if hook_registry:
            hook_registry.register(self.hook)
            logger.info("my_system_hook_registered")

        # Register services if needed
        registry = self.context.get("plugin_registry")
        if registry:
            service = MySystemService(config)
            registry.register_service("my_service", service, self.manifest.name)

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
            routes=[RouteSpec(router=my_router, prefix="/my-system", tags=["my-system"])]
        )
        super().__init__(manifest)

    def create_runtime(self) -> MySystemRuntime:
        return MySystemRuntime(self.manifest)

# Export factory instance
factory = MySystemFactory()
```

### Auth Provider Plugin Example

```python
from ccproxy.core.plugins import (
    AuthProviderPluginFactory,
    AuthProviderPluginRuntime,
    PluginManifest
)

class MyOAuthRuntime(AuthProviderPluginRuntime):
    def __init__(self, manifest: PluginManifest):
        super().__init__(manifest)
        self.config = None

    async def _on_initialize(self) -> None:
        """Initialize the OAuth provider."""
        if self.context:
            config = self.context.get("config")
            if not isinstance(config, MyOAuthConfig):
                config = MyOAuthConfig()
            self.config = config

        # Call parent initialization (handles provider registration)
        await super()._on_initialize()

class MyOAuthFactory(AuthProviderPluginFactory):
    cli_safe = True  # Safe for CLI - provides auth only

    def __init__(self) -> None:
        manifest = PluginManifest(
            name="oauth_my_provider",
            version="1.0.0",
            description="My OAuth authentication provider",
            is_provider=True,  # Auth provider
            config_class=MyOAuthConfig,
            dependencies=[],
            routes=[],  # No HTTP routes needed
            tasks=[]    # No scheduled tasks needed
        )
        super().__init__(manifest)

    def create_runtime(self) -> MyOAuthRuntime:
        return MyOAuthRuntime(self.manifest)

    def create_auth_provider(self, context=None) -> MyOAuthProvider:
        """Create OAuth provider instance."""
        config = context.get("config") if context else MyOAuthConfig()
        http_client = context.get("http_client") if context else None
        return MyOAuthProvider(config, http_client=http_client)

# Export factory instance
factory = MyOAuthFactory()
```

## Configuration

### Plugin Configuration

Plugins can define configuration using Pydantic models:

```python
from pydantic import BaseModel, Field

class MyPluginConfig(BaseModel):
    """Configuration for my plugin."""

    enabled: bool = Field(default=True, description="Enable plugin")
    base_url: str = Field(
        default="https://api.example.com",
        description="Base URL for API"
    )
    timeout: int = Field(default=30, description="Request timeout")
```

### Settings Integration

Plugin configurations are loaded from the main settings:

```toml
# .ccproxy.toml or ccproxy.toml

[plugins.my_plugin]
enabled = true
base_url = "https://api.custom.com"
timeout = 60
```

Or via environment variables:
```bash
export PLUGINS__MY_PLUGIN__ENABLED=true
export PLUGINS__MY_PLUGIN__BASE_URL="https://api.custom.com"
export PLUGINS__MY_PLUGIN__TIMEOUT=60
```

### Enabling/Disabling Plugins

Control which plugins are loaded:

```python
# settings.py
class Settings(BaseModel):
    enable_plugins: bool = True
    enabled_plugins: list[str] | None = None  # None = all
    disabled_plugins: list[str] | None = None
```

Environment variables:
```bash
export ENABLE_PLUGINS=true
export ENABLED_PLUGINS="claude_api,raw_http_logger"
export DISABLED_PLUGINS="codex"
```

## Plugin Directory Structure

```
plugins/
├── __init__.py
├── claude_api/
│   ├── __init__.py
│   ├── plugin.py          # Main plugin file (exports 'factory')
│   ├── adapter.py         # Provider adapter
│   ├── config.py          # Configuration model
│   ├── detection_service.py
│   ├── routes.py          # API routes
│   ├── tasks.py           # Scheduled tasks
│   └── transformers/      # Request/response transformers
│       ├── request.py
│       └── response.py
└── raw_http_logger/
    ├── __init__.py
    ├── plugin.py          # Main plugin file (exports 'factory')
    ├── config.py          # Configuration model
    ├── logger.py          # Core logging functionality
    ├── middleware.py      # HTTP middleware
    └── transport.py       # HTTP transport wrapper
```

## Middleware Layers

Middleware is organized into layers with specific priorities:

```python
class MiddlewareLayer(IntEnum):
    SECURITY = 100         # Authentication, rate limiting
    OBSERVABILITY = 200    # Logging, metrics
    TRANSFORMATION = 300   # Compression, encoding
    ROUTING = 400         # Path rewriting, proxy
    APPLICATION = 500     # Business logic
```

Middleware is applied in reverse order (highest priority runs first).

## Advanced Plugin Features

### Format Adapter System

CCProxy includes a declarative format adapter system for protocol conversion between different API formats (OpenAI ↔ Anthropic ↔ Custom formats).

#### Declarative Format Adapter Specification

Plugins declare format adapters in their factory classes:

```python
from ccproxy.core.plugins.declaration import FormatAdapterSpec, FormatPair

class MyProviderFactory(BaseProviderPluginFactory):
    # Declarative format adapter specification
    format_adapters = [
        FormatAdapterSpec(
            from_format="openai",
            to_format="anthropic",
            adapter_factory=lambda: MyFormatAdapter(),
            priority=40,  # Lower number = higher priority
            description="OpenAI to Anthropic format conversion"
        )
    ]

    # Define format adapter dependencies
    requires_format_adapters: list[FormatPair] = [
        ("anthropic.messages", "openai.responses"),  # Provided by core
    ]
```

#### Format Registry Integration

The system automatically handles conflicts between plugins registering the same format pairs using priority-based resolution with automatic logging.

#### Migration-Safe Runtime Pattern

The system supports dual-path operation during migration:

```python
async def _setup_format_registry(self) -> None:
    """Format registry setup with feature flag control."""
    settings = get_settings()

    # Skip manual setup if manifest system is enabled
    if settings.features.manifest_format_adapters:
        logger.debug("using_manifest_format_adapters")
        return

    # Legacy manual registration as fallback
    registry = self.context.get_service_container().get_format_registry()
    registry.register("openai", "anthropic", MyFormatAdapter(), "my_plugin")
```

### Adapter Compatibility System

CCProxy includes a compatibility shim system that enables seamless integration between legacy dict-based adapters and modern strongly-typed adapters. This system ensures backward compatibility while allowing gradual migration to the new typed interface.

#### AdapterShim Overview

The `AdapterShim` class provides a compatibility layer that wraps strongly-typed adapters from `ccproxy.llms.formatters` to work with existing code that expects `dict[str, Any]` interfaces.

**Key Features:**
- **Automatic Type Conversion**: Seamlessly converts between dict and BaseModel formats
- **Error Preservation**: Maintains meaningful error messages and stack traces
- **Streaming Support**: Handles async generators with proper type conversion
- **Direct Access**: Provides access to underlying typed adapter when needed

#### Architecture

```
┌─────────────────────────────────────────────────────┐
│                Legacy Code                          │
│           (dict[str, Any] interface)                │
├─────────────────────────────────────────────────────┤
│                AdapterShim                          │
│    (Automatic dict ↔ BaseModel conversion)         │
├─────────────────────────────────────────────────────┤
│             Typed Adapters                          │
│        (BaseModel interface with types)             │
└─────────────────────────────────────────────────────┘
```

The shim sits between legacy code expecting dict-based interfaces and modern typed adapters, performing automatic bidirectional conversion:

- **Incoming**: `dict[str, Any]` → `BaseModel` (via generic model creation)
- **Outgoing**: `BaseModel` → `dict[str, Any]` (via `model_dump()`)

#### Usage Examples

##### Manual Shim Creation

```python
from ccproxy.llms.formatters.shim import AdapterShim
from ccproxy.llms.formatters.anthropic_to_openai.messages_to_responses import (
    AnthropicMessagesToOpenAIResponsesAdapter
)

# Create typed adapter
typed_adapter = AnthropicMessagesToOpenAIResponsesAdapter()

# Wrap with shim for legacy compatibility
legacy_adapter = AdapterShim(typed_adapter)

# Now use with legacy dict-based code
request_dict = {"model": "claude-3-sonnet", "messages": [...]}
response_dict = await legacy_adapter.adapt_request(request_dict)
```

##### Registry Integration

The shim system integrates automatically with the plugin registry:

```python
class MyProviderPlugin(BaseProviderPluginFactory):
    def create_format_adapters(self, context: PluginContext) -> list[APIAdapter]:
        """Create format adapters with automatic shim wrapping."""
        typed_adapter = MyTypedAdapter()

        # Registry automatically wraps with shim if needed
        return [typed_adapter]  # Will be shimmed automatically

    def create_legacy_adapter(self, context: PluginContext) -> APIAdapter:
        """Explicit shim creation for legacy systems."""
        typed_adapter = MyTypedAdapter()
        return AdapterShim(typed_adapter)
```

##### Streaming Support

The shim properly handles streaming responses:

```python
# Legacy streaming code works unchanged
async def process_stream(adapter: APIAdapter, stream_data):
    # stream_data is AsyncIterator[dict[str, Any]]
    adapted_stream = adapter.adapt_stream(stream_data)

    # adapted_stream is AsyncGenerator[dict[str, Any], None]
    async for chunk_dict in adapted_stream:
        # chunk_dict is automatically converted from BaseModel
        process_chunk(chunk_dict)
```

#### Error Handling

The shim provides comprehensive error handling with meaningful messages:

```python
try:
    result = await shimmed_adapter.adapt_request(invalid_request)
except ValueError as e:
    # Error messages include adapter name and conversion context
    # e.g., "Invalid request format for anthropic_to_openai: validation error..."
    logger.error("Adapter failed", error=str(e))
```

**Error Categories:**
- **ValidationError**: Invalid input format during dict→BaseModel conversion
- **ValueError**: Adaptation failure in underlying typed adapter
- **TypeError**: Type conversion issues during BaseModel→dict conversion

#### Direct Adapter Access

Access the underlying typed adapter when needed:

```python
shim = AdapterShim(typed_adapter)

# Direct typed operations (bypasses shim conversion)
typed_request = MyRequestModel(model="claude-3-sonnet")
typed_response = await shim.wrapped_adapter.adapt_request(typed_request)

# Legacy operations (uses shim conversion)
dict_response = await shim.adapt_request({"model": "claude-3-sonnet"})
```

#### Migration Patterns

##### Gradual Migration

```python
class MyAdapter:
    def __init__(self, use_typed: bool = False):
        if use_typed:
            # Direct typed adapter
            self._adapter = MyTypedAdapter()
        else:
            # Shimmed adapter for legacy compatibility
            self._adapter = AdapterShim(MyTypedAdapter())

    async def adapt_request(self, request):
        return await self._adapter.adapt_request(request)
```

##### Feature Flag Migration

```python
async def create_adapter(settings: Settings) -> APIAdapter:
    """Create adapter with feature flag control."""
    typed_adapter = MyTypedAdapter()

    if settings.features.use_typed_adapters:
        return typed_adapter  # Direct typed usage
    else:
        return AdapterShim(typed_adapter)  # Legacy compatibility
```

#### Best Practices

1. **Use for Migration**: Employ shims during gradual migration from dict to typed interfaces
2. **Avoid Long-term**: Shims add overhead; migrate to typed adapters when possible
3. **Error Handling**: Always handle `ValueError` exceptions from shim operations
4. **Direct Access**: Use `wrapped_adapter` property for performance-critical typed operations
5. **Testing**: Test both shimmed and direct adapter usage patterns

#### Performance Considerations

- **Conversion Overhead**: Dict↔BaseModel conversion adds processing time
- **Memory Usage**: Temporary model objects created during conversion
- **Streaming**: Minimal overhead for streaming due to lazy evaluation
- **Caching**: Consider caching converted models for repeated operations

#### Troubleshooting

##### Shim Not Converting Properly

1. Check input dict structure matches expected BaseModel fields
2. Verify BaseModel allows extra fields (Config.extra = "allow")
3. Review conversion error messages for validation details

##### Performance Issues

1. Profile conversion overhead in performance-critical paths
2. Consider using direct typed adapter for high-frequency operations
3. Implement caching for repeated conversions

##### Type Safety Issues

1. Use TypedDict hints for better type checking with shimmed adapters
2. Consider migrating critical code paths to direct typed usage
3. Add runtime validation for complex type conversions

### Hook System

CCProxy uses a comprehensive event-driven hook system for request/response lifecycle management.

#### Hook Implementation

```python
from ccproxy.core.plugins.hooks import Hook, HookContext, HookEvent

class MyHook(Hook):
    name = "my_hook"
    events = [
        HookEvent.REQUEST_STARTED,
        HookEvent.REQUEST_COMPLETED,
        HookEvent.PROVIDER_STREAM_END
    ]
    priority = 750  # Higher number = later execution

    async def __call__(self, context: HookContext) -> None:
        """Handle hook events."""
        if context.event == HookEvent.REQUEST_STARTED:
            # Extract request data
            request_id = context.data.get("request_id")
            method = context.data.get("method")

        elif context.event == HookEvent.PROVIDER_STREAM_END:
            # Handle streaming completion with metrics
            usage_metrics = context.data.get("usage_metrics", {})
            tokens_input = usage_metrics.get("input_tokens", 0)
```

#### Hook Registration

Hooks are registered during plugin initialization:

```python
class MySystemRuntime(SystemPluginRuntime):
    async def _on_initialize(self) -> None:
        # Create hook instance
        self.hook = MyHook(self.config)

        # Get hook registry from context
        hook_registry = self.context.get(HookRegistry)
        if hook_registry:
            hook_registry.register(self.hook)
```

#### Available Hook Events

- `REQUEST_STARTED`: Request initiated by client
- `REQUEST_COMPLETED`: Request completed successfully
- `REQUEST_FAILED`: Request failed with error
- `PROVIDER_REQUEST_SENT`: Request sent to provider
- `PROVIDER_RESPONSE_RECEIVED`: Response received from provider
- `PROVIDER_ERROR`: Provider request failed
- `PROVIDER_STREAM_START`: Streaming response started
- `PROVIDER_STREAM_CHUNK`: Streaming chunk received
- `PROVIDER_STREAM_END`: Streaming response completed

### Service Registry

Plugins can provide and consume services through the plugin registry:

#### Providing Services

```python
class MySystemRuntime(SystemPluginRuntime):
    async def _on_initialize(self) -> None:
        # Create service instance
        service = MyAnalyticsService(self.config)

        # Register service
        registry = self.context.get("plugin_registry")
        if registry:
            registry.register_service("my_analytics", service, self.manifest.name)
```

#### Consuming Services

```python
class MyProviderRuntime(ProviderPluginRuntime):
    async def _on_initialize(self) -> None:
        # Get optional service
        registry = self.context.get("plugin_registry")
        if registry:
            pricing_service = registry.get_service("pricing", PricingService)
            if pricing_service:
                self.pricing_service = pricing_service
```

### Plugin Context

The plugin context provides access to core services and components:

#### Available Context Services

- `settings`: Global application settings
- `http_client`: Managed HTTP client with hooks
- `plugin_registry`: Plugin registry for service discovery
- `hook_registry`: Hook registry for event subscription
- `service_container`: Core service container
- `config`: Plugin-specific validated configuration
- `request_tracer`: Request tracing service
- `streaming_handler`: Streaming response handler
- `format_registry`: Format adapter registry

## Best Practices

1. **Use Type Hints**: Ensure all plugin code is fully typed
2. **Handle Errors Gracefully**: Plugins should not crash the application
3. **Implement Health Checks**: Provide meaningful health status
4. **Log Appropriately**: Use structured logging with context
5. **Clean Up Resources**: Implement proper shutdown logic
6. **Document Configuration**: Provide clear configuration documentation
7. **Test Thoroughly**: Include unit and integration tests
8. **Version Appropriately**: Use semantic versioning

## Troubleshooting

### Plugin Not Loading

1. Check plugin directory structure
2. Verify `plugin.py` exports `factory` variable
3. Check for import errors in logs
4. Ensure dependencies are satisfied

### Plugin Initialization Fails

1. Check configuration is valid
2. Verify required services are available
3. Check for permission errors
4. Review initialization logs

### Middleware Not Applied

1. Verify middleware spec in manifest
2. Check priority settings
3. Ensure middleware class is valid
4. Review middleware application logs

### Routes Not Available

1. Check route spec in manifest
2. Verify router prefix is unique
3. Ensure routes are registered during app creation
4. Check for route conflicts

## OAuth Integration

The plugin system includes comprehensive OAuth support, allowing plugins to provide their own OAuth authentication flows. OAuth providers are registered dynamically at runtime through the plugin manifest.

### OAuth Architecture

```
┌─────────────────────────────────────────────────────┐
│                 OAuth Registry                       │
│  (Central registry for all OAuth providers)          │
├─────────────────────────────────────────────────────┤
│              Plugin OAuth Providers                  │
│  (Plugin-specific OAuth implementations)             │
├─────────────────────────────────────────────────────┤
│                OAuth Components                      │
│  (Client, Storage, Config, Session Manager)          │
└─────────────────────────────────────────────────────┘
```

### OAuth Provider Registration

Plugins register OAuth providers through their manifest:

```python
@dataclass
class PluginManifest:
    # ... other fields ...

    # OAuth provider factory
    oauth_provider_factory: Callable[[], OAuthProviderProtocol] | None = None
```

### OAuth Provider Protocol

All OAuth providers must implement the `OAuthProviderProtocol`:

```python
class OAuthProviderProtocol(Protocol):
    @property
    def provider_name(self) -> str:
        """Internal provider name (e.g., 'claude-api', 'codex')."""

    @property
    def provider_display_name(self) -> str:
        """Display name for UI (e.g., 'Claude API', 'OpenAI Codex')."""

    @property
    def supports_pkce(self) -> bool:
        """Whether this provider supports PKCE flow."""

    @property
    def supports_refresh(self) -> bool:
        """Whether this provider supports token refresh."""

    async def get_authorization_url(
        self, state: str, code_verifier: str | None = None
    ) -> str:
        """Get the authorization URL for OAuth flow."""

    async def handle_callback(
        self, code: str, state: str, code_verifier: str | None = None
    ) -> Any:
        """Handle OAuth callback and exchange code for tokens."""

    async def refresh_access_token(self, refresh_token: str) -> Any:
        """Refresh access token using refresh token."""

    async def revoke_token(self, token: str) -> None:
        """Revoke an access or refresh token."""

    def get_storage(self) -> Any:
        """Get storage implementation for this provider."""

    def get_credential_summary(self, credentials: Any) -> dict[str, Any]:
        """Get a summary of credentials for display."""
```

### Plugin OAuth Implementation

#### 1. Create OAuth Provider

```python
# ccproxy/plugins/claude_api/oauth/provider.py
from ccproxy.auth.oauth.registry import OAuthProviderInfo, OAuthProviderProtocol

class ClaudeOAuthProvider:
    def __init__(self, config=None, storage=None):
        self.config = config or ClaudeOAuthConfig()
        self.storage = storage or ClaudeTokenStorage()
        self.client = ClaudeOAuthClient(self.config, self.storage)

    @property
    def provider_name(self) -> str:
        return "claude-api"

    @property
    def provider_display_name(self) -> str:
        return "Claude API"

    # ... implement other protocol methods ...
```

#### 2. Register in Plugin Manifest

```python
# ccproxy/plugins/claude_api/plugin.py
class ClaudeAPIPlugin(PluginFactory):
    def get_manifest(self) -> PluginManifest:
        return PluginManifest(
            name="claude_api",
            version="1.0.0",
            description="Claude API provider plugin",
            is_provider=True,
            oauth_provider_factory=self._create_oauth_provider,
        )

    def _create_oauth_provider(self) -> OAuthProviderProtocol:
        """Create OAuth provider instance."""
        from .oauth.provider import ClaudeOAuthProvider
        return ClaudeOAuthProvider()
```

#### 3. OAuth Components

Each plugin OAuth implementation typically includes:

- **Provider**: Main OAuth provider implementing the protocol
- **Client**: OAuth client handling token exchange and refresh
- **Storage**: Token storage implementation
- **Config**: OAuth configuration (client ID, URLs, scopes)

### OAuth Registry

The central registry manages all OAuth providers:

```python
# ccproxy/auth/oauth/registry.py
class OAuthRegistry:
    def register_provider(self, provider: OAuthProviderProtocol) -> None:
        """Register an OAuth provider."""

    def get_provider(self, provider_name: str) -> OAuthProviderProtocol | None:
        """Get a registered provider by name."""

    def list_providers(self) -> dict[str, OAuthProviderInfo]:
        """List all registered providers."""

    def unregister_provider(self, provider_name: str) -> None:
        """Unregister a provider (not supported at runtime in v2)."""
```

### CLI Integration

OAuth providers are automatically available through the CLI:

```bash
# List available OAuth providers
ccproxy auth providers

# Login with a provider
ccproxy auth login claude-api

# Check authentication status
ccproxy auth status claude-api

# Refresh tokens
ccproxy auth refresh claude-api

# Logout
ccproxy auth logout claude-api
```

### OAuth Flow

1. **Discovery**: Plugins register OAuth providers during initialization
2. **Authorization**: User initiates OAuth flow through CLI
3. **Callback**: OAuth callback handled by provider
4. **Token Storage**: Credentials stored securely
5. **Token Refresh**: Automatic or manual token refresh
6. **Revocation**: Token revocation on logout

### Security Considerations

- **PKCE Support**: Use PKCE for public clients
- **State Validation**: Prevent CSRF attacks
- **Secure Storage**: Encrypt sensitive tokens
- **Token Expiry**: Handle token expiration gracefully
- **Scope Management**: Request minimal required scopes

### Example: Complete OAuth Provider

```python
# ccproxy/plugins/codex/oauth/provider.py
class CodexOAuthProvider:
    def __init__(self, config=None, storage=None):
        self.config = config or CodexOAuthConfig()
        self.storage = storage or CodexTokenStorage()
        self.client = CodexOAuthClient(self.config, self.storage)

    @property
    def provider_name(self) -> str:
        return "codex"

    @property
    def provider_display_name(self) -> str:
        return "OpenAI Codex"

    @property
    def supports_pkce(self) -> bool:
        return self.config.use_pkce

    @property
    def supports_refresh(self) -> bool:
        return True

    async def get_authorization_url(
        self, state: str, code_verifier: str | None = None
    ) -> str:
        params = {
            "client_id": self.config.client_id,
            "redirect_uri": self.config.redirect_uri,
            "response_type": "code",
            "scope": " ".join(self.config.scopes),
            "state": state,
            "audience": self.config.audience,
        }

        if self.config.use_pkce and code_verifier:
            # Add PKCE challenge
            code_challenge = self._generate_challenge(code_verifier)
            params["code_challenge"] = code_challenge
            params["code_challenge_method"] = "S256"

        return f"{self.config.authorize_url}?{urlencode(params)}"

    async def handle_callback(
        self, code: str, state: str, code_verifier: str | None = None
    ) -> Any:
        # Exchange code for tokens
        credentials = await self.client.handle_callback(
            code, state, code_verifier or ""
        )

        # Store credentials
        if self.storage:
            await self.storage.save_credentials(credentials)

        return credentials

    async def refresh_access_token(self, refresh_token: str) -> Any:
        credentials = await self.client.refresh_token(refresh_token)

        if self.storage:
            await self.storage.save_credentials(credentials)

        return credentials

    async def revoke_token(self, token: str) -> None:
        # OpenAI doesn't have a revoke endpoint
        # Delete stored credentials instead
        if self.storage:
            await self.storage.delete_credentials()

    def get_provider_info(self) -> OAuthProviderInfo:
        return OAuthProviderInfo(
            name=self.provider_name,
            display_name=self.provider_display_name,
            description="OAuth authentication for OpenAI Codex",
            supports_pkce=self.supports_pkce,
            scopes=self.config.scopes,
            is_available=True,
            plugin_name="codex",
        )

    def get_storage(self) -> Any:
        return self.storage

    def get_credential_summary(self, credentials: OpenAICredentials) -> dict[str, Any]:
        return {
            "provider": self.provider_display_name,
            "authenticated": bool(credentials),
            "account_id": credentials.account_id if credentials else None,
            "expired": credentials.is_expired() if credentials else False,
        }
```

## Authoring Guide

For step-by-step instructions on building plugins, including configuration precedence, entry point publishing, service registration, and test patterns, refer to `docs/PLUGIN_AUTHORING.md`.
