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

## Plugin System Overview

CCProxy uses a modern plugin system (v2) that provides a flexible, declarative architecture for extending the proxy server's functionality. The system supports two types of plugins:

- **Provider Plugins**: Proxy requests to external AI providers (Claude API, Claude SDK, Codex)
- **System Plugins**: Add functionality like logging, monitoring, and permissions

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

Provider plugins proxy requests to external API providers. They implement the `ProviderPlugin` protocol and include:

- **Adapter**: Handles request/response processing
- **Detection Service**: Detects provider capabilities
- **Credentials Manager**: Manages authentication
- **Transformers**: Transform requests/responses

Example providers: `claude_api`, `claude_sdk`, `codex`

### System Plugins

System plugins add functionality without proxying to external providers. They implement the `SystemPlugin` protocol and include:

- **Middleware**: Request/response processing
- **Routes**: Additional API endpoints
- **Tasks**: Background scheduled tasks
- **Hooks**: Event-based extensions

Example system plugins: `raw_http_logger`, `permissions`

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
- Plugins discovered from `plugins/` directory
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

All plugin management endpoints are prefixed with `/api/plugins`.

#### List Plugins
```http
GET /api/plugins
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
GET /api/plugins/{plugin_name}/health
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

#### Reload Plugin
```http
POST /api/plugins/{plugin_name}/reload
```

**Note:** In v2 plugin system, plugins are loaded at startup and cannot be reloaded at runtime. This endpoint returns HTTP 501 Not Implemented.

#### Discover Plugins
```http
POST /api/plugins/discover
```

**Note:** Returns the current list of plugins. Dynamic discovery at runtime is not supported in v2.

#### Unregister Plugin
```http
DELETE /api/plugins/{plugin_name}
```

**Parameters:**
- `plugin_name` (path): Name of the plugin to unregister

**Response:**
```json
{
  "status": "success",
  "message": "Plugin 'raw_http_logger' unregistered successfully"
}
```

## Integration Guide

### Application Integration

The plugin system integrates with the FastAPI application during the `create_app` function in `ccproxy/api/app.py`:

```python
def create_app(settings: Settings | None = None) -> FastAPI:
    # Phase 1: Discovery and Registration
    plugin_registry = PluginRegistry()
    middleware_manager = MiddlewareManager()

    if settings.enable_plugins:
        # Discover and load plugin factories
        plugin_factories = discover_and_load_plugins(settings)

        # Register all plugin factories
        for factory in plugin_factories.values():
            plugin_registry.register_factory(factory)

        # Create context for manifest population
        manifest_services = ManifestPopulationServices(settings)

        # Populate manifests
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

    # Create service container
    service_container = ServiceContainer(settings)

    # Create core services adapter
    core_services = CoreServicesAdapter(service_container)

    # Initialize all plugins
    await plugin_registry.initialize_all(core_services)

async def shutdown_plugins_v2(app: FastAPI) -> None:
    """Shutdown v2 plugins."""
    if hasattr(app.state, "plugin_registry"):
        plugin_registry: PluginRegistry = app.state.plugin_registry
        await plugin_registry.shutdown_all()
```

## Creating Plugins

### Provider Plugin Example

```python
from ccproxy.plugins import (
    PluginManifest,
    ProviderPluginFactory,
    ProviderPluginRuntime,
    RouteSpec
)

class MyProviderRuntime(ProviderPluginRuntime):
    async def _on_initialize(self) -> None:
        """Initialize the provider."""
        await super()._on_initialize()
        # Provider-specific initialization

class MyProviderFactory(ProviderPluginFactory):
    def __init__(self):
        manifest = PluginManifest(
            name="my_provider",
            version="1.0.0",
            description="My provider plugin",
            is_provider=True,
            config_class=MyProviderConfig,
            routes=[
                RouteSpec(
                    router=my_router,
                    prefix="/my-provider",
                    tags=["plugin-my-provider"]
                )
            ]
        )
        super().__init__(manifest)

    def create_runtime(self) -> MyProviderRuntime:
        return MyProviderRuntime(self.manifest)

    def create_adapter(self, context: PluginContext) -> Any:
        return MyProviderAdapter(
            proxy_service=context.get("proxy_service"),
            http_client=context.get("http_client")
        )

    def create_detection_service(self, context: PluginContext) -> Any:
        return MyDetectionService()

    def create_credentials_manager(self, context: PluginContext) -> Any:
        return MyCredentialsManager()

# Export factory instance
factory = MyProviderFactory()
```

### System Plugin Example

```python
from ccproxy.plugins import (
    PluginManifest,
    SystemPluginFactory,
    SystemPluginRuntime,
    MiddlewareSpec,
    MiddlewareLayer
)

class MySystemRuntime(SystemPluginRuntime):
    async def _on_initialize(self) -> None:
        """Initialize the system plugin."""
        # System-specific initialization

class MySystemFactory(SystemPluginFactory):
    def __init__(self):
        manifest = PluginManifest(
            name="my_system",
            version="1.0.0",
            description="My system plugin",
            is_provider=False,
            config_class=MySystemConfig,
            middleware=[
                MiddlewareSpec(
                    middleware_class=MyMiddleware,
                    priority=MiddlewareLayer.OBSERVABILITY,
                    kwargs={"param": "value"}
                )
            ]
        )
        super().__init__(manifest)

    def create_runtime(self) -> MySystemRuntime:
        return MySystemRuntime(self.manifest)

# Export factory instance
factory = MySystemFactory()
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
# plugins/claude_api/oauth/provider.py
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
# plugins/claude_api/plugin.py
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
        """Unregister a provider."""
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
# plugins/codex/oauth/provider.py
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
