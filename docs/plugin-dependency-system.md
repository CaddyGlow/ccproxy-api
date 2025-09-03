# Plugin Dependency System

## Overview

The CCProxy plugin system uses a service registry pattern to manage dependencies between plugins. This allows plugins to provide services that other plugins can consume without tight coupling to the core container.

## Service Registry Pattern

### Registering Services

Plugins that provide services register them during initialization:

```python
class PricingPlugin(BasePlugin):
    async def register_services(self, registry: PluginRegistry) -> None:
        """Register pricing service for other plugins."""
        pricing_service = PricingService(self.config)
        await pricing_service.initialize()
        registry.register_service("pricing", pricing_service, "pricing")
```

### Consuming Services

Plugins declare dependencies in their manifests and access services through the registry:

```python
# In plugin manifest
manifest = PluginManifest(
    name="claude_api",
    optional_requires=["pricing"]  # Optional dependency
)

# In plugin code
def _get_pricing_service(self) -> PricingService | None:
    """Get pricing service if available."""
    if self.registry and self.registry.has_service("pricing"):
        return self.registry.get_service("pricing")
    return None
```

## Plugin Manifest Fields

### Dependency Declaration

- `provides`: List of service names this plugin provides
- `requires`: List of required service dependencies
- `optional_requires`: List of optional service dependencies

Example:
```python
PluginManifest(
    name="pricing",
    provides=["pricing"],
    requires=[],
    optional_requires=[]
)
```

## Streaming Cost Calculation

### Architecture

The streaming cost calculation is handled at the provider level through metrics collectors:

1. **Provider-specific collectors** extract token usage from streaming chunks
2. **Synchronous cost calculation** avoids async delays during streaming
3. **Metrics attached to context** for access logging

### Implementation

```python
# In streaming_metrics.py
class StreamingMetricsCollector:
    def __init__(self, request_id: str | None = None,
                 pricing_service: Any = None,
                 model: str | None = None):
        self.pricing_service = pricing_service
        self.model = model

    def process_chunk(self, chunk_str: str) -> bool:
        # Extract tokens from provider format
        # Calculate cost synchronously on final chunk
        if event_type == "message_delta" and self.pricing_service:
            cost = self.pricing_service.calculate_cost_sync(...)
            self.metrics["cost_usd"] = float(cost)
            return True  # Final chunk
```

### Key Design Decisions

1. **Synchronous calculation**: Uses `calculate_cost_sync()` to avoid async calls during streaming
2. **Provider-specific**: Each provider handles its own token extraction format
3. **Final chunk calculation**: Cost calculated when complete token counts are available
4. **Separation of concerns**: Core streaming infrastructure remains agnostic to pricing

## Migration Guide

### From Direct Container Access

Before:
```python
pricing_service = self.container.get_pricing_service()
```

After:
```python
pricing_service = self.registry.get_service("pricing")
```

### Service Availability Checks

Always check service availability for optional dependencies:
```python
if self.registry and self.registry.has_service("pricing"):
    pricing_service = self.registry.get_service("pricing")
    # Use pricing service
else:
    # Handle absence gracefully
    logger.debug("Pricing service not available")
```

## Benefits

1. **Decoupling**: Plugins don't depend on core container implementation
2. **Flexibility**: Services can be added/removed without core changes
3. **Testability**: Easy to mock services in tests
4. **Clarity**: Explicit dependency declarations in manifests
5. **Performance**: Synchronous operations where needed for streaming
