# Explicit Dependency Injection Refactoring Plan

## Overview
Replace the service locator pattern (ProxyService/PluginContext) with explicit dependency injection. This will eliminate hidden dependencies, improve testability, and make the codebase more maintainable.

**No backward compatibility required** - Clean break refactoring.

## Current Problems

### Service Locator Anti-Pattern
```python
# Current: Dependencies are hidden
class ClaudeAPIAdapter:
    def __init__(self, proxy_service, ...):
        self.proxy_service = proxy_service
        # Later in code:
        http_client = proxy_service.http_client  # Hidden dependency
        metrics = proxy_service.metrics  # Hidden dependency
```

### Issues:
1. **Hidden Dependencies**: Can't see what an adapter needs from its constructor
2. **Tight Coupling**: Adapters coupled to ProxyService structure
3. **Hard to Test**: Must mock entire ProxyService
4. **God Object Risk**: ProxyService/PluginContext becoming dumping grounds

## Target Architecture: Explicit Dependencies

```python
# Target: All dependencies explicit
class ClaudeAPIAdapter:
    def __init__(self,
                 http_client: AsyncClient,
                 auth_manager: AuthManager,
                 detection_service: DetectionService,
                 request_tracer: RequestTracer | None = None,
                 metrics: PrometheusMetrics | None = None,
                 streaming_handler: StreamingHandler | None = None):
        # Every dependency is explicit and typed
```

## Implementation Plan

### Phase 1: Service Interface Definitions

**New file**: `ccproxy/services/interfaces.py`

```python
from typing import Protocol, AsyncIterator
import httpx
from starlette.responses import Response

class IRequestTracer(Protocol):
    """Request tracing interface."""
    async def trace_request(self, request_id: str, ...) -> None: ...
    def should_trace(self) -> bool: ...

class IMetricsCollector(Protocol):
    """Metrics collection interface."""
    def track_request(self, method: str, path: str) -> None: ...
    def track_response(self, status: int, duration: float) -> None: ...

class IStreamingHandler(Protocol):
    """Streaming response handler interface."""
    async def handle_stream(self, response: AsyncIterator[bytes]) -> AsyncIterator[bytes]: ...

# Null implementations for optional dependencies
class NullRequestTracer:
    async def trace_request(self, *args, **kwargs) -> None: pass
    def should_trace(self) -> bool: return False

class NullMetricsCollector:
    def track_request(self, *args, **kwargs) -> None: pass
    def track_response(self, *args, **kwargs) -> None: pass
```

### Phase 2: Update BaseHTTPAdapter

**File**: `ccproxy/services/adapters/http_adapter.py`

```python
class BaseHTTPAdapter(BaseAdapter):
    def __init__(self,
                 # Required dependencies
                 http_client: AsyncClient,
                 auth_manager: AuthManager,
                 
                 # Optional dependencies with null defaults
                 request_tracer: IRequestTracer | None = None,
                 metrics: IMetricsCollector | None = None,
                 streaming_handler: IStreamingHandler | None = None,
                 request_transformer: RequestTransformer | None = None,
                 response_transformer: ResponseTransformer | None = None,
                 logger: structlog.BoundLogger | None = None):
        
        # Store required dependencies
        self.http_client = http_client
        self.auth_manager = auth_manager
        
        # Use null object pattern for optional dependencies
        self.request_tracer = request_tracer or NullRequestTracer()
        self.metrics = metrics or NullMetricsCollector()
        self.streaming_handler = streaming_handler
        self.logger = logger or structlog.get_logger()
        
        # Initialize HTTP handler with explicit dependencies
        self._http_handler = PluginHTTPHandler(
            http_client=http_client,
            request_tracer=self.request_tracer
        )
```

### Phase 3: Update Concrete Adapters

#### ClaudeAPIAdapter
```python
class ClaudeAPIAdapter(BaseHTTPAdapter):
    def __init__(self,
                 # Required dependencies
                 http_client: AsyncClient,
                 auth_manager: ClaudeApiTokenManager,
                 detection_service: ClaudeAPIDetectionService,
                 
                 # Optional dependencies
                 request_tracer: IRequestTracer | None = None,
                 metrics: IMetricsCollector | None = None,
                 streaming_handler: IStreamingHandler | None = None,
                 pricing_service: Any | None = None,
                 openai_adapter: OpenAIAdapter | None = None,
                 logger: structlog.BoundLogger | None = None,
                 
                 # Configuration
                 config: ClaudeAPISettings | None = None):
        
        # Initialize transformers
        request_transformer = ClaudeAPIRequestTransformer(detection_service)
        response_transformer = ClaudeAPIResponseTransformer()
        
        # Call parent with explicit dependencies
        super().__init__(
            http_client=http_client,
            auth_manager=auth_manager,
            request_tracer=request_tracer,
            metrics=metrics,
            streaming_handler=streaming_handler,
            request_transformer=request_transformer,
            response_transformer=response_transformer,
            logger=logger
        )
        
        # Store Claude-specific dependencies
        self.detection_service = detection_service
        self.pricing_service = pricing_service
        self.openai_adapter = openai_adapter or OpenAIAdapter()
        self.config = config or ClaudeAPISettings()
```

#### CodexAdapter (similar pattern)
```python
class CodexAdapter(BaseHTTPAdapter):
    def __init__(self,
                 http_client: AsyncClient,
                 auth_manager: CodexTokenManager,
                 detection_service: CodexDetectionService,
                 # ... optional dependencies
                 ):
        # Similar explicit initialization
```

#### ClaudeSDKAdapter (doesn't extend BaseHTTPAdapter)
```python
class ClaudeSDKAdapter(BaseAdapter):
    def __init__(self,
                 config: ClaudeSDKSettings,
                 session_manager: SessionManager | None = None,
                 metrics: IMetricsCollector | None = None,
                 logger: structlog.BoundLogger | None = None):
        # SDK adapter has different dependencies (no HTTP client)
        self.config = config
        self.session_manager = session_manager
        self.metrics = metrics or NullMetricsCollector()
        self.logger = logger or structlog.get_logger()
        self.handler = ClaudeSDKHandler(config, session_manager)
```

### Phase 4: Update Factory Pattern

**File**: `ccproxy/plugins/base_factory.py`

```python
class BaseProviderPluginFactory:
    def create_adapter(self, context: PluginContext) -> BaseAdapter:
        """Create adapter with explicit dependencies."""
        
        # Extract services from context (one-time extraction)
        http_client = context.get("http_client")
        request_tracer = context.get("request_tracer")
        metrics = context.get("metrics")
        streaming_handler = context.get("streaming_handler")
        logger = context.get("logger")
        
        # Create auth and detection services
        auth_manager = self.create_credentials_manager(context)
        detection_service = self.create_detection_service(context)
        
        # Get pricing service if available
        pricing_service = self._get_pricing_service(context)
        
        # Create adapter with explicit dependencies
        if issubclass(self.adapter_class, BaseHTTPAdapter):
            return self.adapter_class(
                http_client=http_client,
                auth_manager=auth_manager,
                detection_service=detection_service,
                request_tracer=request_tracer,
                metrics=metrics,
                streaming_handler=streaming_handler,
                pricing_service=pricing_service,
                logger=logger,
                config=context.get("config")
            )
        else:
            # Non-HTTP adapters (like ClaudeSDK) have different dependencies
            return self.adapter_class(
                config=context.get("config"),
                metrics=metrics,
                logger=logger
            )
```

### Phase 5: Update ServiceContainer

**File**: `ccproxy/services/container.py`

```python
class ServiceContainer:
    """Service container that provides individual services, not ProxyService."""
    
    def get_adapter_dependencies(self) -> dict[str, Any]:
        """Get all services an adapter might need."""
        return {
            "http_client": self.get_http_client(),
            "request_tracer": self.get_request_tracer(),
            "metrics": self.get_metrics(),
            "streaming_handler": self.get_streaming_handler(),
            "logger": self.get_logger(),
        }
    
    # Remove create_proxy_service method entirely
    # Individual service getters remain
```

### Phase 6: Update Routes

**Example**: `plugins/claude_api/routes.py`

```python
# Remove ProxyServiceDep entirely
# Routes already call adapters directly (from Phase 3 of previous refactoring)

@router.post("/v1/messages")
async def create_anthropic_message(
    request: Request,
    auth: ConditionalAuthDep,
    adapter: ClaudeAPIAdapterDep,  # Already injected with all dependencies
) -> Response | StreamingResponse | DeferredStreaming:
    """Route handler with adapter that has explicit dependencies."""
    return await adapter.handle_request(
        request=request,
        endpoint="/v1/messages",
        method=request.method
    )
```

### Phase 7: Delete ProxyService

Once all adapters use explicit dependencies:
1. Delete `ccproxy/services/proxy_service.py`
2. Remove ProxyService from imports
3. Update ServiceContainer to not create ProxyService
4. Remove ProxyService from PluginContext type definition

## Testing Benefits

### Before (Service Locator)
```python
def test_adapter():
    # Must create entire ProxyService mock
    mock_proxy = Mock()
    mock_proxy.http_client = Mock()
    mock_proxy.metrics = Mock()
    mock_proxy.request_tracer = Mock()
    # ... many more mocks
    
    adapter = ClaudeAPIAdapter(
        proxy_service=mock_proxy,
        auth_manager=Mock(),
        detection_service=Mock()
    )
```

### After (Explicit Dependencies)
```python
def test_adapter():
    # Only mock what you need
    adapter = ClaudeAPIAdapter(
        http_client=Mock(),
        auth_manager=Mock(),
        detection_service=Mock()
        # Optional dependencies default to null implementations
    )
    
    # Or test with specific service
    adapter = ClaudeAPIAdapter(
        http_client=Mock(),
        auth_manager=Mock(),
        detection_service=Mock(),
        metrics=Mock()  # Only when testing metrics
    )
```

## Implementation Order

### Week 1: Foundation
1. Create service interfaces (`interfaces.py`)
2. Update BaseHTTPAdapter with explicit dependencies
3. Create null implementations for optional services

### Week 2: Adapter Migration
1. Update ClaudeAPIAdapter
2. Update CodexAdapter
3. Update ClaudeSDKAdapter
4. Update any other adapters

### Week 3: Factory and Container Updates
1. Update BaseProviderPluginFactory
2. Update individual plugin factories if needed
3. Update ServiceContainer (remove ProxyService creation)

### Week 4: Cleanup
1. Verify all routes working
2. Delete ProxyService
3. Update type definitions
4. Clean up imports

## Success Metrics

- ✅ Zero service locator patterns
- ✅ All dependencies explicit in constructors
- ✅ ProxyService completely removed
- ✅ Improved unit test coverage
- ✅ Reduced coupling between components
- ✅ Clear dependency graph

## Risk Mitigation

Since no backward compatibility is required:
- Make changes in feature branch
- Run comprehensive tests after each phase
- Use type checker (mypy) to catch issues
- Review dependency graph for circular dependencies

## Key Principles

1. **Explicit over Implicit**: All dependencies visible in constructor
2. **Null Object Pattern**: Optional services have null implementations
3. **Interface Segregation**: Depend on interfaces, not implementations
4. **Single Responsibility**: Each service has one clear purpose
5. **Dependency Inversion**: Depend on abstractions, not concretions

## Expected Outcome

A cleaner, more testable architecture where:
- Every dependency is explicit and typed
- No hidden service access through proxy objects
- Easy to test with minimal mocking
- Clear understanding of what each component needs
- No god objects or service locators