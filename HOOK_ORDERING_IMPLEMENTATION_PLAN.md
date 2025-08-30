# Hook Ordering and Data Modification Implementation Plan

## Executive Summary

This plan outlines the implementation of a priority-based hook ordering system with controlled data modification capabilities for the CCProxy hook system. The system will enable hooks to execute in defined phases, modify request/response data in a controlled manner, and maintain clear data flow dependencies.

## Current State Analysis

### Existing Architecture
- **Hook Protocol**: Basic hook interface with `name` and `events` properties
- **HookContext**: Mutable dataclass with `data` and `metadata` dicts
- **HookRegistry**: Simple list-based storage, no ordering
- **HookManager**: Sequential execution, error isolation, async/sync support
- **Data Mutability**: Hooks can freely modify `context.data` and `context.metadata`

### Current Limitations
1. No execution order guarantees
2. No way to express hook dependencies
3. Uncontrolled data mutations possible
4. No clear execution phases
5. Difficult to debug data flow

## Proposed Architecture

### 1. Hook Priority System

#### Priority Ranges (0-1000)
```python
class HookPriority(IntEnum):
    # Pre-processing: Setup and validation
    CRITICAL = 0          # System-critical hooks (request ID generation)
    VALIDATION = 100      # Input validation and sanitization
    
    # Enrichment: Add context and metadata
    AUTH = 200           # Authentication and authorization
    CONTEXT = 300        # Context enrichment (session, user data)
    
    # Processing: Core business logic
    TRANSFORM = 400      # Request/response transformation
    BUSINESS = 500       # Business logic and processing
    
    # Observation: Logging and metrics
    METRICS = 600        # Metrics collection
    LOGGING = 700        # Access and audit logging
    
    # Post-processing: Cleanup and finalization
    CACHE = 800          # Cache operations
    CLEANUP = 900        # Resource cleanup
    FINALIZE = 1000      # Final operations
```

### 2. Enhanced Hook Protocol

```python
from typing import Protocol, Set, Optional
from dataclasses import dataclass, field

class DataModification(Enum):
    """Types of data modifications a hook can perform"""
    READ_ONLY = "read_only"           # No modifications
    ADD_ONLY = "add_only"             # Can add new keys, not modify existing
    MODIFY_EXISTING = "modify_existing" # Can modify existing keys
    FULL_ACCESS = "full_access"       # Can add, modify, delete

@dataclass
class HookCapabilities:
    """Declares what a hook can do with data"""
    data_access: DataModification = DataModification.READ_ONLY
    metadata_access: DataModification = DataModification.READ_ONLY
    modifies_request: bool = False
    modifies_response: bool = False
    required_fields: Set[str] = field(default_factory=set)
    provides_fields: Set[str] = field(default_factory=set)

class EnhancedHook(Protocol):
    """Enhanced hook protocol with priority and capabilities"""
    
    @property
    def name(self) -> str:
        """Unique hook identifier"""
        ...
    
    @property
    def events(self) -> list[HookEvent]:
        """Events this hook listens to"""
        ...
    
    @property
    def priority(self) -> int:
        """Execution priority (0-1000, lower executes first)"""
        ...
    
    @property
    def capabilities(self) -> HookCapabilities:
        """Declared capabilities and data access patterns"""
        ...
    
    @property
    def depends_on(self) -> Set[str]:
        """Names of hooks this hook depends on (optional)"""
        return set()
    
    async def __call__(self, context: HookContext) -> Optional[HookContext]:
        """Execute hook, optionally returning modified context"""
        ...
```

### 3. Protected Hook Context

```python
from typing import Any, Dict, MutableMapping
from collections import ChainMap

class ProtectedDict(MutableMapping):
    """Dictionary with controlled modification based on access level"""
    
    def __init__(self, data: Dict[str, Any], access: DataModification):
        self._data = data
        self._access = access
        self._original_keys = set(data.keys())
    
    def __setitem__(self, key: str, value: Any):
        if self._access == DataModification.READ_ONLY:
            raise PermissionError(f"Hook has read-only access")
        elif self._access == DataModification.ADD_ONLY:
            if key in self._original_keys:
                raise PermissionError(f"Hook can only add new keys, not modify '{key}'")
        elif self._access == DataModification.MODIFY_EXISTING:
            if key not in self._original_keys:
                raise PermissionError(f"Hook can only modify existing keys, not add '{key}'")
        self._data[key] = value
    
    def __delitem__(self, key: str):
        if self._access != DataModification.FULL_ACCESS:
            raise PermissionError("Hook does not have delete permission")
        del self._data[key]
    
    # ... other dict methods ...

@dataclass
class ProtectedHookContext:
    """Hook context with access control"""
    
    event: HookEvent
    timestamp: datetime
    _data: Dict[str, Any]
    _metadata: Dict[str, Any]
    request: Optional[Request] = None
    response: Optional[Response] = None
    provider: Optional[str] = None
    plugin: Optional[str] = None
    error: Optional[Exception] = None
    
    # Track modifications for debugging
    _modifications: List[Dict[str, Any]] = field(default_factory=list)
    
    def get_protected_view(self, hook: EnhancedHook) -> 'HookContext':
        """Get a protected view of context based on hook capabilities"""
        return HookContext(
            event=self.event,
            timestamp=self.timestamp,
            data=ProtectedDict(self._data, hook.capabilities.data_access),
            metadata=ProtectedDict(self._metadata, hook.capabilities.metadata_access),
            request=self.request,
            response=self.response,
            provider=self.provider,
            plugin=self.plugin,
            error=self.error
        )
    
    def record_modification(self, hook_name: str, field: str, old_value: Any, new_value: Any):
        """Record data modifications for audit/debugging"""
        self._modifications.append({
            "hook": hook_name,
            "field": field,
            "old_value": old_value,
            "new_value": new_value,
            "timestamp": datetime.utcnow()
        })
```

### 4. Enhanced Hook Registry

```python
from sortedcontainers import SortedList
from collections import defaultdict
import networkx as nx

class EnhancedHookRegistry:
    """Registry with priority ordering and dependency resolution"""
    
    def __init__(self):
        # Store hooks sorted by priority
        self._hooks: Dict[HookEvent, SortedList] = defaultdict(
            lambda: SortedList(key=lambda h: (h.priority, h.name))
        )
        self._hook_map: Dict[str, EnhancedHook] = {}
        self._dependency_graph = nx.DiGraph()
        self._logger = structlog.get_logger(__name__)
    
    def register(self, hook: EnhancedHook) -> None:
        """Register hook with dependency checking"""
        # Validate hook
        self._validate_hook(hook)
        
        # Check for conflicts
        self._check_conflicts(hook)
        
        # Add to registry
        self._hook_map[hook.name] = hook
        for event in hook.events:
            self._hooks[event].add(hook)
        
        # Update dependency graph
        self._dependency_graph.add_node(hook.name)
        for dep in hook.depends_on:
            if dep in self._hook_map:
                self._dependency_graph.add_edge(dep, hook.name)
        
        # Check for circular dependencies
        if not nx.is_directed_acyclic_graph(self._dependency_graph):
            # Remove and raise error
            self.unregister(hook)
            raise ValueError(f"Circular dependency detected with hook {hook.name}")
        
        self._logger.info(
            "hook_registered",
            name=hook.name,
            priority=hook.priority,
            events=[e.value for e in hook.events],
            depends_on=list(hook.depends_on)
        )
    
    def get_hooks(self, event: HookEvent) -> List[EnhancedHook]:
        """Get hooks for event in priority and dependency order"""
        event_hooks = list(self._hooks[event])
        
        # Apply topological sort for dependencies within same priority
        return self._apply_dependency_order(event_hooks)
    
    def _apply_dependency_order(self, hooks: List[EnhancedHook]) -> List[EnhancedHook]:
        """Reorder hooks based on dependencies while respecting priorities"""
        # Group by priority
        by_priority = defaultdict(list)
        for hook in hooks:
            by_priority[hook.priority].append(hook)
        
        # Within each priority level, sort by dependencies
        result = []
        for priority in sorted(by_priority.keys()):
            priority_hooks = by_priority[priority]
            if len(priority_hooks) > 1:
                # Build subgraph for this priority level
                subgraph = self._dependency_graph.subgraph([h.name for h in priority_hooks])
                if subgraph.edges():
                    # Has dependencies, use topological sort
                    sorted_names = list(nx.topological_sort(subgraph))
                    priority_hooks.sort(key=lambda h: sorted_names.index(h.name) if h.name in sorted_names else -1)
            result.extend(priority_hooks)
        
        return result
    
    def _validate_hook(self, hook: EnhancedHook):
        """Validate hook configuration"""
        if not 0 <= hook.priority <= 1000:
            raise ValueError(f"Hook {hook.name} priority must be between 0 and 1000")
        
        # Check that required fields are available from earlier hooks
        for event in hook.events:
            available_fields = self._get_available_fields(event, hook.priority)
            missing = hook.capabilities.required_fields - available_fields
            if missing:
                self._logger.warning(
                    "hook_missing_required_fields",
                    hook=hook.name,
                    missing_fields=list(missing),
                    available_fields=list(available_fields)
                )
    
    def _get_available_fields(self, event: HookEvent, max_priority: int) -> Set[str]:
        """Get fields that will be available at a given priority"""
        available = set()
        for hook in self._hooks[event]:
            if hook.priority < max_priority:
                available.update(hook.capabilities.provides_fields)
        return available
    
    def _check_conflicts(self, new_hook: EnhancedHook):
        """Check for field provision conflicts"""
        for event in new_hook.events:
            for existing in self._hooks[event]:
                if existing.priority == new_hook.priority:
                    # Check for conflicting field provisions
                    overlap = (existing.capabilities.provides_fields & 
                              new_hook.capabilities.provides_fields)
                    if overlap:
                        self._logger.warning(
                            "hook_field_conflict",
                            hook1=existing.name,
                            hook2=new_hook.name,
                            conflicting_fields=list(overlap),
                            priority=new_hook.priority
                        )
```

### 5. Enhanced Hook Manager

```python
class EnhancedHookManager:
    """Manager with priority execution and data flow tracking"""
    
    def __init__(self, registry: EnhancedHookRegistry, debug_mode: bool = False):
        self._registry = registry
        self._debug_mode = debug_mode
        self._logger = structlog.get_logger(__name__)
    
    async def emit(
        self,
        event: HookEvent,
        data: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> ProtectedHookContext:
        """Emit event with protected context and ordered execution"""
        
        # Create protected context
        context = ProtectedHookContext(
            event=event,
            timestamp=datetime.utcnow(),
            _data=data or {},
            _metadata={},
            **kwargs
        )
        
        # Get hooks in priority order
        hooks = self._registry.get_hooks(event)
        
        if self._debug_mode:
            self._logger.debug(
                "hook_execution_order",
                event=event.value,
                hooks=[{"name": h.name, "priority": h.priority} for h in hooks]
            )
        
        # Execute hooks with protection
        for hook in hooks:
            try:
                # Create protected view for this hook
                protected_view = context.get_protected_view(hook)
                
                # Track data before execution
                if self._debug_mode:
                    before_data = deepcopy(context._data)
                    before_metadata = deepcopy(context._metadata)
                
                # Execute hook
                result = await self._execute_hook(hook, protected_view)
                
                # Track modifications
                if self._debug_mode:
                    self._track_modifications(
                        hook, context, before_data, before_metadata
                    )
                
                # Validate provided fields
                self._validate_provisions(hook, context)
                
            except PermissionError as e:
                self._logger.error(
                    "hook_permission_denied",
                    hook=hook.name,
                    error=str(e),
                    event=event.value
                )
                if not self._continue_on_error:
                    raise
            except Exception as e:
                self._logger.error(
                    "hook_execution_failed",
                    hook=hook.name,
                    error=str(e),
                    event=event.value,
                    exc_info=True
                )
                # Continue with other hooks
        
        return context
    
    def _track_modifications(
        self,
        hook: EnhancedHook,
        context: ProtectedHookContext,
        before_data: Dict,
        before_metadata: Dict
    ):
        """Track what data was modified by a hook"""
        # Check data modifications
        for key in set(before_data.keys()) | set(context._data.keys()):
            old_val = before_data.get(key)
            new_val = context._data.get(key)
            if old_val != new_val:
                context.record_modification(
                    hook.name, f"data.{key}", old_val, new_val
                )
                self._logger.debug(
                    "hook_modified_data",
                    hook=hook.name,
                    field=key,
                    old_value=old_val,
                    new_value=new_val
                )
        
        # Check metadata modifications
        for key in set(before_metadata.keys()) | set(context._metadata.keys()):
            old_val = before_metadata.get(key)
            new_val = context._metadata.get(key)
            if old_val != new_val:
                context.record_modification(
                    hook.name, f"metadata.{key}", old_val, new_val
                )
    
    def _validate_provisions(self, hook: EnhancedHook, context: ProtectedHookContext):
        """Validate that hook provided promised fields"""
        for field in hook.capabilities.provides_fields:
            if field not in context._data and field not in context._metadata:
                self._logger.warning(
                    "hook_missing_provision",
                    hook=hook.name,
                    expected_field=field
                )
```

## Implementation Phases

### Phase 1: Core Infrastructure (Week 1)
1. Implement `EnhancedHook` protocol with priority
2. Create `ProtectedDict` for access control
3. Update `HookContext` to `ProtectedHookContext`
4. Add priority support to `HookRegistry`

### Phase 2: Dependency System (Week 2)
1. Add dependency graph to registry
2. Implement topological sorting
3. Add circular dependency detection
4. Create dependency validation

### Phase 3: Data Protection (Week 3)
1. Implement access control levels
2. Add modification tracking
3. Create audit logging
4. Add permission validation

### Phase 4: Migration (Week 4)
1. Create compatibility layer for existing hooks
2. Migrate existing hooks to new system
3. Update hook implementations with priorities
4. Test end-to-end with real requests

### Phase 5: Monitoring & Debugging (Week 5)
1. Add hook execution visualization
2. Create data flow debugging tools
3. Implement performance monitoring
4. Add hook conflict detection

## Migration Strategy

### Backward Compatibility
```python
class LegacyHookAdapter(EnhancedHook):
    """Adapter for legacy hooks"""
    
    def __init__(self, legacy_hook: Hook, priority: int = 500):
        self._legacy = legacy_hook
        self._priority = priority
    
    @property
    def name(self) -> str:
        return self._legacy.name
    
    @property
    def priority(self) -> int:
        return self._priority
    
    @property
    def capabilities(self) -> HookCapabilities:
        # Legacy hooks get full access for compatibility
        return HookCapabilities(
            data_access=DataModification.FULL_ACCESS,
            metadata_access=DataModification.FULL_ACCESS
        )
    
    async def __call__(self, context: HookContext):
        return await self._legacy(context)
```

### Migration Steps
1. **Wrap existing hooks** with `LegacyHookAdapter`
2. **Assign default priorities** based on hook type
3. **Gradually update** hooks to use new protocol
4. **Add capabilities** declarations incrementally
5. **Tighten permissions** as hooks are updated

## Configuration

### Environment Variables
```bash
# Hook system configuration
HOOKS__ENABLE_PRIORITY=true
HOOKS__ENABLE_PROTECTION=true
HOOKS__DEBUG_MODE=false
HOOKS__CONTINUE_ON_ERROR=true
HOOKS__MAX_EXECUTION_TIME_MS=1000
HOOKS__ENABLE_DEPENDENCY_CHECKING=true
```

### TOML Configuration
```toml
[hooks]
enable_priority = true
enable_protection = true
debug_mode = false
continue_on_error = true

[hooks.priorities]
# Override default priorities for specific hooks
access_log = 700
metrics = 600
auth_enrichment = 200

[hooks.capabilities]
# Override capabilities for specific hooks
[hooks.capabilities.access_log]
data_access = "read_only"
metadata_access = "read_only"
```

## Example Implementation

### Authentication Enrichment Hook
```python
class AuthEnrichmentHook:
    """Adds authentication context to requests"""
    
    name = "auth_enrichment"
    events = [HookEvent.REQUEST_STARTED]
    priority = HookPriority.AUTH  # 200
    depends_on = {"request_id_generator"}  # Depends on request ID
    
    capabilities = HookCapabilities(
        data_access=DataModification.ADD_ONLY,
        metadata_access=DataModification.ADD_ONLY,
        provides_fields={"user_id", "auth_method", "permissions"}
    )
    
    async def __call__(self, context: HookContext):
        # Extract auth from request
        if context.request:
            auth_header = context.request.headers.get("Authorization")
            if auth_header:
                # Parse and validate auth
                user_info = await self._validate_auth(auth_header)
                
                # Add to context (only adding new fields)
                context.data["user_id"] = user_info["id"]
                context.data["auth_method"] = user_info["method"]
                context.metadata["permissions"] = user_info["permissions"]
```

### Metrics Collection Hook
```python
class MetricsHook:
    """Collects metrics from enriched request data"""
    
    name = "metrics"
    events = [HookEvent.REQUEST_COMPLETED]
    priority = HookPriority.METRICS  # 600
    depends_on = {"auth_enrichment"}  # Needs user context
    
    capabilities = HookCapabilities(
        data_access=DataModification.READ_ONLY,
        metadata_access=DataModification.READ_ONLY,
        required_fields={"user_id", "duration_ms"}
    )
    
    async def __call__(self, context: HookContext):
        # Read data (no modifications)
        user_id = context.data.get("user_id")
        duration = context.data.get("duration_ms")
        
        # Emit metrics
        await self._emit_metric(
            "request_duration",
            duration,
            tags={"user": user_id}
        )
```

## Testing Strategy

### Unit Tests
1. Test priority ordering
2. Test dependency resolution
3. Test access control
4. Test modification tracking

### Integration Tests
1. Test full hook chain execution
2. Test data flow between hooks
3. Test error handling
4. Test performance impact

### Load Tests
1. Measure overhead of protection
2. Test with many hooks
3. Test with complex dependencies
4. Benchmark vs current system

## Monitoring & Observability

### Metrics
- Hook execution time by priority
- Data modifications per hook
- Permission violations
- Dependency resolution time
- Hook failure rates

### Logging
- Hook registration with priorities
- Execution order per request
- Data modifications (debug mode)
- Permission denials
- Dependency conflicts

### Debugging Tools
```python
class HookDebugger:
    """Interactive hook debugging tool"""
    
    def trace_execution(self, event: HookEvent, data: dict):
        """Trace hook execution with data flow"""
        ...
    
    def visualize_dependencies(self):
        """Generate dependency graph visualization"""
        ...
    
    def analyze_data_flow(self, request_id: str):
        """Analyze how data flowed through hooks"""
        ...
```

## Risk Mitigation

### Risks
1. **Performance overhead** from protection layers
2. **Breaking changes** for existing hooks
3. **Complex debugging** with many hooks
4. **Dependency deadlocks**

### Mitigations
1. **Optimization**: Cache permission checks, use fast data structures
2. **Compatibility layer**: Gradual migration with adapters
3. **Debug tooling**: Comprehensive tracing and visualization
4. **Validation**: Detect circular dependencies at registration

## Success Criteria

1. **Predictable execution order** based on priorities
2. **Clear data flow** with tracked modifications
3. **No unauthorized data changes**
4. **< 5% performance impact**
5. **100% backward compatibility**
6. **Improved debugging capabilities**

## Timeline

- **Week 1**: Core infrastructure
- **Week 2**: Dependency system
- **Week 3**: Data protection
- **Week 4**: Migration and compatibility
- **Week 5**: Monitoring and tools
- **Week 6**: Testing and optimization
- **Week 7**: Documentation and training
- **Week 8**: Production rollout

## Conclusion

This implementation plan provides a comprehensive approach to adding hook ordering and controlled data modification to the CCProxy hook system. The phased approach ensures backward compatibility while gradually introducing new capabilities that will make the system more predictable, debuggable, and maintainable.