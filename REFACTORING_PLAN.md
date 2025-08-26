# CCProxy Architecture Refactoring Plan

## Executive Summary
This document outlines a comprehensive refactoring of the CCProxy plugin architecture to reduce code duplication, simplify request flow, and improve maintainability. The refactoring is divided into manageable phases with clear success metrics and rollback strategies.

## Current Architecture Issues

### 1. Plugin Factory Duplication
- **Problem**: 60-70% code duplication across plugin factories (ClaudeAPIFactory, CodexFactory, ClaudeSDKFactory)
- **Impact**: Adding new providers requires copying boilerplate code
- **Solution**: Create BaseProviderPluginFactory to centralize common patterns

### 2. Circular Delegation Pattern
- **Problem**: Routes call ProxyService.handle_request() which calls back to adapter.handle_request()
- **Impact**: Unnecessary indirection, confusing code flow, performance overhead
- **Solution**: Routes should call adapters directly; move ProxyService responsibilities to middleware

### 3. Incomplete Request Tracing
- **Problem**: Raw HTTP interception only captures provider-side requests/responses
- **Impact**: Missing client-side data for debugging
- **Solution**: Complete four-point interception in RequestTracingMiddleware

### 4. Mixed Adapter Responsibilities
- **Problem**: HTTP and SDK-based adapters forced into same pattern despite different needs
- **Impact**: Unnecessary complexity in SDK adapters
- **Solution**: Create separate base classes for HTTP vs SDK adapters

## Implementation Phases

### Phase 1: Low Risk, High Value (Week 1)

#### Stage 4: Complete Client-Side Raw Data Interception
**Priority**: HIGH | **Risk**: LOW | **Value**: HIGH

**Files to modify**:
- `plugins/request_tracer/middleware.py` - Add client request/response capture
- `plugins/request_tracer/tracer.py` - Ensure methods exist for client-side logging

**Implementation**:
1. Capture raw client request from ASGI scope in RequestTracingMiddleware
2. Build raw HTTP request format from headers, method, path
3. Log to `{request_id}_client_request.http`
4. Wrap send() to capture response headers and body
5. Log to `{request_id}_client_response.http`

**Success Criteria**:
- Four `.http` files generated per request
- Raw data matches actual wire format
- No performance impact when disabled

#### Stage 1: Create BaseProviderPluginFactory
**Priority**: HIGH | **Risk**: LOW | **Value**: HIGH

**New file**:
- `ccproxy/plugins/base_factory.py` - BaseProviderPluginFactory implementation

**Files to refactor**:
- `plugins/claude_api/plugin.py` - Inherit from BaseProviderPluginFactory
- `plugins/codex/plugin.py` - Inherit from BaseProviderPluginFactory
- `plugins/claude_sdk/plugin.py` - Inherit from BaseProviderPluginFactory

**Implementation**:
```python
class BaseProviderPluginFactory(ProviderPluginFactory):
    # Required class attributes
    plugin_name: str
    plugin_version: str = "1.0.0"
    plugin_description: str
    runtime_class: type[ProviderPluginRuntime]
    adapter_class: type[BaseAdapter]
    detection_service_class: type | None = None
    credentials_manager_class: type | None = None
    config_class: type[BaseSettings]
    router: APIRouter | None = None
    route_prefix: str = "/api"
    dependencies: list[str] = []
    optional_requires: list[str] = []
    tasks: list[TaskSpec] = []

    def __init__(self):
        manifest = self._build_manifest()
        super().__init__(manifest)

    def _build_manifest(self) -> PluginManifest:
        return PluginManifest(
            name=self.plugin_name,
            version=self.plugin_version,
            description=self.plugin_description,
            is_provider=True,
            config_class=self.config_class,
            dependencies=self.dependencies,
            optional_requires=self.optional_requires,
            routes=self._build_routes(),
            tasks=self.tasks,
        )

    def _build_routes(self) -> list[RouteSpec]:
        if not self.router:
            return []
        return [RouteSpec(
            router=self.router,
            prefix=self.route_prefix,
            tags=[f"plugin-{self.plugin_name}"]
        )]

    def create_runtime(self) -> ProviderPluginRuntime:
        return self.runtime_class(self.manifest)

    def create_adapter(self, context: PluginContext) -> BaseAdapter:
        kwargs = self._prepare_adapter_kwargs(context)
        return self.adapter_class(**kwargs)

    def _prepare_adapter_kwargs(self, context: PluginContext) -> dict:
        """Prepare kwargs for adapter initialization."""
        return {
            "proxy_service": context.get("proxy_service"),
            "auth_manager": context.get("credentials_manager"),
            "detection_service": context.get("detection_service"),
            "http_client": context.get("http_client"),
            "logger": context.get("logger"),
            "context": context
        }

    def create_detection_service(self, context: PluginContext):
        if not self.detection_service_class:
            return None
        settings = context.get("settings") or Settings()
        cli_service = context.get("cli_detection_service")
        return self.detection_service_class(settings, cli_service)

    def create_credentials_manager(self, context: PluginContext):
        if not self.credentials_manager_class:
            return None
        return self.credentials_manager_class()
```

**Success Criteria**:
- 60-70% reduction in plugin factory code
- All existing tests pass
- New factories are declarative (class attributes only)

### Phase 2: Medium Risk, Medium Value (Week 2)

#### Stage 3: Create BaseHTTPAdapter
**Priority**: MEDIUM | **Risk**: MEDIUM | **Value**: MEDIUM

**New file**:
- `ccproxy/services/adapters/http_adapter.py` - BaseHTTPAdapter implementation

**Files to refactor**:
- `plugins/claude_api/adapter.py` - Inherit from BaseHTTPAdapter
- `plugins/codex/adapter.py` - Inherit from BaseHTTPAdapter

**Implementation Details**:
- Extract common HTTP orchestration logic
- Handle request/response transformation
- Manage streaming responses
- Integrate with PluginHTTPHandler

#### Stage 2.1: Add HooksMiddleware
**Priority**: MEDIUM | **Risk**: MEDIUM | **Value**: HIGH

**New file**:
- `ccproxy/api/middleware/hooks.py` - HooksMiddleware implementation

**Files to modify**:
- `ccproxy/api/app.py` - Add HooksMiddleware to stack

**Implementation Details**:
- Emit REQUEST_STARTED before processing
- Emit REQUEST_COMPLETED/FAILED after processing
- Maintain RequestContext compatibility

### Phase 3: Higher Risk, Cleanup (Week 3)

#### Stage 2.2: Remove Circular Delegation
**Priority**: LOW | **Risk**: HIGH | **Value**: MEDIUM

**Files to modify**:
- All `plugins/*/routes.py` files
- `ccproxy/services/proxy_service.py`

**Implementation Details**:
- Update routes to call adapters directly
- Remove ProxyService.handle_request method
- Update dependency injection

#### Stage 5: Simplify ProxyService
**Priority**: LOW | **Risk**: MEDIUM | **Value**: LOW

**Files to modify**:
- `ccproxy/services/proxy_service.py`

**Implementation Details**:
- Remove unused methods
- Focus on service initialization only
- Maintain backward compatibility

## Testing Strategy

### Unit Tests
```bash
# Test base factory
pytest tests/unit/plugins/test_base_factory.py -v

# Test individual factories
pytest tests/unit/plugins/test_claude_api_factory.py -v
pytest tests/unit/plugins/test_codex_factory.py -v

# Test middleware
pytest tests/unit/api/middleware/test_hooks.py -v
pytest tests/unit/plugins/request_tracer/test_middleware.py -v

# Test adapters
pytest tests/unit/services/adapters/test_http_adapter.py -v
```

### Integration Tests
```bash
# End-to-end request flow
pytest tests/integration/test_request_flow.py -v

# Raw data capture verification
pytest tests/integration/test_raw_data_interception.py -v

# Hook emission verification
pytest tests/integration/test_hook_emission.py -v
```

### Performance Tests
```bash
# Measure middleware overhead
python scripts/benchmark_middleware.py

# Test raw data logging impact
python scripts/benchmark_raw_logging.py

# Memory usage with streaming
python scripts/benchmark_streaming.py
```

## Rollback Plan

### Feature Flags
```python
# In settings
ENABLE_NEW_FACTORY_PATTERN = env.bool("ENABLE_NEW_FACTORY_PATTERN", False)
ENABLE_HOOKS_MIDDLEWARE = env.bool("ENABLE_HOOKS_MIDDLEWARE", False)
ENABLE_CLIENT_RAW_CAPTURE = env.bool("ENABLE_CLIENT_RAW_CAPTURE", False)
```

### Gradual Migration
1. Implement new patterns alongside old ones
2. Use feature flags to toggle between implementations
3. Monitor metrics and logs for issues
4. Roll back individual components if needed

## Success Metrics

### Code Quality
- [ ] 60-70% reduction in factory boilerplate
- [ ] Simplified request flow (3-4 fewer function calls)
- [ломate test coverage > 90%
- [ ] All existing tests pass

### Performance
- [ ] < 1ms added latency per request
- [ ] No increase in memory usage
- [ ] Raw logging overhead < 5% when enabled

### Functionality
- [ ] Complete four-point raw data interception
- [ ] Hook emission at correct lifecycle points
- [ ] Backward compatibility maintained

## Risk Matrix

| Stage | Risk | Impact | Mitigation |
|-------|------|--------|------------|
| Client-side interception | LOW | HIGH | Feature flag, extensive testing |
| BaseProviderPluginFactory | LOW | HIGH | Incremental migration, keep old factories |
| BaseHTTPAdapter | MEDIUM | MEDIUM | Test with one provider first |
| HooksMiddleware | MEDIUM | HIGH | Run alongside ProxyService initially |
| Remove circular delegation | HIGH | LOW | Careful coordination, phased rollout |
| Simplify ProxyService | MEDIUM | LOW | Maintain compatibility layer |

## Timeline

### Week 1: Phase 1
- Mon-Tue: Complete client-side interception
- Wed-Thu: Create BaseProviderPluginFactory
- Fri: Testing and documentation

### Week 2: Phase 2
- Mon-Tue: Create BaseHTTPAdapter
- Wed-Thu: Add HooksMiddleware
- Fri: Integration testing

### Week 3: Phase 3
- Mon-Wed: Remove circular delegation
- Thu: Simplify ProxyService
- Fri: Final testing and cleanup

## Appendix: File Structure

```
ccproxy-api/
├── ccproxy/
│   ├── plugins/
│   │   └── base_factory.py (NEW)
│   ├── services/
│   │   └── adapters/
│   │       ├── base.py (existing)
│   │       └── http_adapter.py (NEW)
│   └── api/
│       └── middleware/
│           ├── hooks.py (NEW)
│           └── metrics.py (NEW)
├── plugins/
│   ├── claude_api/
│   │   ├── plugin.py (REFACTOR)
│   │   ├── adapter.py (REFACTOR)
│   │   └── routes.py (MODIFY)
│   ├── codex/
│   │   ├── plugin.py (REFACTOR)
│   │   ├── adapter.py (REFACTOR)
│   │   └── routes.py (MODIFY)
│   ├── claude_sdk/
│   │   ├── plugin.py (REFACTOR)
│   │   └── routes.py (MODIFY)
│   └── request_tracer/
│       └── middleware.py (ENHANCE)
└── tests/
    └── (new test files)
```

## Notes

- Each phase is designed to be independently valuable
- Rollback is possible at any stage
- Performance monitoring is critical during rollout
- Documentation updates required after each phase
