# Hooks System Integration Plan

## Overview
This document outlines the plan to complete the hooks system integration and refactor the request_tracer plugin to use hooks instead of the observer pattern.

## Current State Analysis

### What Exists
1. **Hook Infrastructure** (`ccproxy/hooks/`)
   - `HookManager` - Event dispatcher
   - `HookRegistry` - Hook registration
   - `HookEvent` enum - Event definitions
   - `HookContext` - Event context data
   - `HooksMiddleware` - Request lifecycle hooks (not wired up)

2. **Existing Hook Implementation**
   - `RequestLoggingHook` - Example hook implementation
   - App startup hook emission in `app.py`

3. **Request Tracer Plugin**
   - Uses ObservabilityPipeline with observer pattern
   - `TracerObserver` implements `RequestObserver`
   - Direct HTTP transport wrapping

### What's Missing
1. HooksMiddleware not added to FastAPI app
2. No provider-level hook emissions
3. Request tracer still uses observer pattern
4. Incomplete hook event coverage

## Implementation Plan

### Phase 1: Enable Core Hook System
**Goal**: Wire up the existing hooks infrastructure

#### 1.1 Enable HooksMiddleware
- **File**: `ccproxy/api/app.py`
- **Changes**:
  ```python
  # After middleware_manager setup
  from ccproxy.api.middleware.hooks import create_hooks_middleware

  # Add hooks middleware to the stack
  hooks_middleware = create_hooks_middleware(hook_manager)
  middleware_manager.add_middleware(
      hooks_middleware,
      layer=MiddlewareLayer.OBSERVABILITY,
      priority=150  # After RequestContext, before other observability
  )
  ```

#### 1.2 Fix Hook Manager Initialization
- **File**: `ccproxy/api/app.py`
- **Changes**:
  - Ensure hook_manager is properly passed to ServiceContainer
  - Set hook_manager on app.state for middleware access

### Phase 2: Add Provider-Level Hook Events
**Goal**: Emit hooks for provider requests/responses

#### 2.1 Update BaseHTTPAdapter
- **File**: `ccproxy/services/adapters/http_adapter.py`
- **Changes**:
  ```python
  # In _execute_request method, add hook emissions:

  # Before sending to provider
  if self.context and "hook_manager" in self.context:
      hook_manager = self.context["hook_manager"]
      await hook_manager.emit(
          HookEvent.PROVIDER_REQUEST_SENT,
          HookContext(
              event=HookEvent.PROVIDER_REQUEST_SENT,
              provider=self.__class__.__name__,
              data={
                  "url": target_url,
                  "method": method,
                  "headers": headers,
                  "is_streaming": is_streaming,
              },
              metadata={"request_id": request_context.request_id}
          )
      )

  # After receiving response
  await hook_manager.emit(
      HookEvent.PROVIDER_RESPONSE_RECEIVED,
      ...
  )
  ```

#### 2.2 Add Streaming Hook Events
- **File**: `ccproxy/services/streaming.py`
- **Changes**:
  - Emit `PROVIDER_STREAM_START` when streaming begins
  - Emit `PROVIDER_STREAM_CHUNK` for each chunk
  - Emit `PROVIDER_STREAM_END` when complete

### Phase 3: Create RequestTracerHook
**Goal**: Replace observer pattern with hook-based tracing

#### 3.1 Create New Hook Implementation
- **File**: `plugins/request_tracer/hook.py`
- **Implementation**:
  ```python
  from ccproxy.hooks import Hook, HookContext, HookEvent
  from .formatters import JSONFormatter, RawHTTPFormatter

  class RequestTracerHook(Hook):
      """Hook-based request tracer implementation."""

      name = "request_tracer"
      events = [
          HookEvent.REQUEST_STARTED,
          HookEvent.REQUEST_COMPLETED,
          HookEvent.REQUEST_FAILED,
          HookEvent.PROVIDER_REQUEST_SENT,
          HookEvent.PROVIDER_RESPONSE_RECEIVED,
          HookEvent.PROVIDER_STREAM_START,
          HookEvent.PROVIDER_STREAM_END,
      ]

      def __init__(self, config: RequestTracerConfig):
          self.config = config
          self.json_formatter = JSONFormatter(config) if config.json_logs_enabled else None
          self.raw_formatter = RawHTTPFormatter(config) if config.raw_http_enabled else None

      async def __call__(self, context: HookContext) -> None:
          if not self.config.enabled:
              return

          # Route to appropriate handler
          handlers = {
              HookEvent.REQUEST_STARTED: self._handle_request_start,
              HookEvent.REQUEST_COMPLETED: self._handle_request_complete,
              HookEvent.PROVIDER_REQUEST_SENT: self._handle_provider_request,
              HookEvent.PROVIDER_RESPONSE_RECEIVED: self._handle_provider_response,
          }

          handler = handlers.get(context.event)
          if handler:
              await handler(context)
  ```

#### 3.2 Update Plugin Runtime
- **File**: `plugins/request_tracer/plugin.py`
- **Changes**:
  ```python
  async def _on_initialize(self) -> None:
      # Create and register hook instead of observer
      self.hook = RequestTracerHook(self.config)

      # Get hook registry from context
      hook_registry = self.context.get("hook_registry")
      if hook_registry:
          hook_registry.register(self.hook)
          logger.info("request_tracer_hook_registered")
  ```

### Phase 4: Migration Strategy
**Goal**: Safely migrate from observer to hooks

#### 4.1 Parallel Operation (Week 1)
- Keep both observer and hook implementations
- Add feature flag: `HOOKS_ENABLED=true`
- Compare outputs to ensure consistency

#### 4.2 Gradual Cutover (Week 2)
- Disable observer for specific event types
- Monitor for any issues
- Validate logging output

#### 4.3 Complete Migration (Week 3)
- Remove observer implementation
- Clean up ObservabilityPipeline references
- Update documentation

## Testing Plan

### Unit Tests
1. Test hook registration and deregistration
2. Test event emission and handling
3. Test context data propagation
4. Test error handling in hooks

### Integration Tests
1. Test full request lifecycle with hooks
2. Test provider request/response hooks
3. Test streaming hooks
4. Test multiple hooks on same event

### Performance Tests
1. Measure overhead of hook system
2. Compare with observer pattern performance
3. Test with high request volume

## Benefits

1. **Unified Event System**: Single event system for all plugins
2. **Decoupling**: Remove dependencies on ObservabilityPipeline
3. **Flexibility**: Multiple plugins can listen to same events
4. **Consistency**: Standard pattern across codebase
5. **Extensibility**: Easy to add new events and hooks

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Performance regression | Medium | Benchmark before/after, optimize if needed |
| Missing events | High | Run parallel with observer initially |
| Breaking existing plugins | High | Feature flag for gradual rollout |
| Complex debugging | Medium | Add comprehensive logging |

## Timeline

- **Week 1**: Enable core hooks system (Phase 1-2)
- **Week 2**: Implement RequestTracerHook (Phase 3)
- **Week 3**: Parallel operation and testing (Phase 4.1)
- **Week 4**: Complete migration (Phase 4.2-4.3)

## Success Criteria

1. All request/response events captured via hooks
2. No performance degradation (< 5% overhead)
3. Clean removal of observer pattern
4. Documentation updated
5. All tests passing

## Future Extensions

Once hooks system is proven:
1. Migrate access_log plugin to hooks
2. Migrate pricing plugin to hooks
3. Add custom user hooks support
4. Add hook priority/ordering
5. Add hook filtering/conditions
