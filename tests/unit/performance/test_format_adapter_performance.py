"""Performance tests for format adapter registry and conflict resolution.

This module tests performance characteristics and validates conflict resolution
behavior under various load conditions.
"""

import pytest
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List

from ccproxy.services.adapters.format_registry import FormatAdapterRegistry
from ccproxy.core.plugins.declaration import FormatAdapterSpec, PluginManifest
from ccproxy.adapters.base import APIAdapter


class BenchmarkAdapter(APIAdapter):
    """Minimal adapter for performance testing."""
    
    def __init__(self, name: str = "benchmark"):
        self.name = name
    
    async def adapt_request(self, request_data):
        return {"benchmark": self.name}
    
    async def adapt_response(self, response_data):
        return {"benchmark": self.name}
    
    async def adapt_stream(self, stream):
        async def benchmark_stream():
            yield {"benchmark": self.name}
        return benchmark_stream()


@pytest.mark.performance
class TestFormatAdapterPerformance:
    """Performance tests for format adapter registry."""

    @pytest.fixture
    def registry(self):
        return FormatAdapterRegistry(conflict_mode="priority")

    @pytest.mark.asyncio
    async def test_large_scale_registration_performance(self, registry):
        """Test performance with large number of adapter registrations."""
        num_plugins = 100
        adapters_per_plugin = 5
        
        start_time = time.perf_counter()
        
        # Register many plugins with multiple adapters each
        for plugin_idx in range(num_plugins):
            specs = []
            for adapter_idx in range(adapters_per_plugin):
                spec = FormatAdapterSpec(
                    from_format=f"from_{plugin_idx}_{adapter_idx}",
                    to_format=f"to_{plugin_idx}_{adapter_idx}",
                    adapter_factory=lambda: BenchmarkAdapter(f"adapter_{plugin_idx}_{adapter_idx}"),
                    priority=plugin_idx
                )
                specs.append(spec)
            
            manifest = PluginManifest(
                name=f"plugin_{plugin_idx}",
                version="1.0.0",
                format_adapters=specs
            )
            
            await registry.register_from_manifest(manifest, f"plugin_{plugin_idx}")
        
        registration_time = time.perf_counter() - start_time
        
        # Finalize registry
        finalize_start = time.perf_counter()
        await registry.resolve_conflicts_and_finalize()
        finalization_time = time.perf_counter() - finalize_start
        
        total_adapters = num_plugins * adapters_per_plugin
        
        # Performance assertions
        assert registration_time < 5.0, f"Registration took {registration_time:.2f}s for {total_adapters} adapters"
        assert finalization_time < 10.0, f"Finalization took {finalization_time:.2f}s for {total_adapters} adapters"
        
        # Verify all adapters were registered
        assert len(registry._adapters) == total_adapters
        
        print(f"Performance: Registered {total_adapters} adapters in {registration_time:.3f}s")
        print(f"Performance: Finalized {total_adapters} adapters in {finalization_time:.3f}s")

    @pytest.mark.asyncio
    async def test_conflict_resolution_performance(self, registry):
        """Test performance of conflict resolution with many conflicts."""
        num_conflicts = 50
        plugins_per_conflict = 4
        
        # Create many conflicting adapters
        for conflict_idx in range(num_conflicts):
            format_pair = (f"conflict_{conflict_idx}", "target")
            
            for plugin_idx in range(plugins_per_conflict):
                spec = FormatAdapterSpec(
                    from_format=format_pair[0],
                    to_format=format_pair[1],
                    adapter_factory=lambda: BenchmarkAdapter(f"conflict_{conflict_idx}_{plugin_idx}"),
                    priority=plugin_idx * 10  # Different priorities for resolution
                )
                
                manifest = PluginManifest(
                    name=f"conflict_plugin_{conflict_idx}_{plugin_idx}",
                    version="1.0.0",
                    format_adapters=[spec]
                )
                
                await registry.register_from_manifest(manifest, f"conflict_plugin_{conflict_idx}_{plugin_idx}")
        
        # Resolve conflicts
        start_time = time.perf_counter()
        await registry.resolve_conflicts_and_finalize(enable_priority_mode=True)
        resolution_time = time.perf_counter() - start_time
        
        # Should resolve quickly even with many conflicts
        assert resolution_time < 2.0, f"Conflict resolution took {resolution_time:.2f}s for {num_conflicts} conflicts"
        
        # Verify conflicts were resolved (only one adapter per format pair)
        assert len(registry._adapters) == num_conflicts
        
        print(f"Performance: Resolved {num_conflicts} conflicts in {resolution_time:.3f}s")

    @pytest.mark.asyncio
    async def test_concurrent_registration_safety(self, registry):
        """Test thread safety of concurrent registrations."""
        num_concurrent = 20
        
        async def register_plugin(plugin_id: int):
            """Register a single plugin with adapters."""
            spec = FormatAdapterSpec(
                from_format=f"concurrent_{plugin_id}",
                to_format="target",
                adapter_factory=lambda: BenchmarkAdapter(f"concurrent_{plugin_id}"),
                priority=plugin_id
            )
            
            manifest = PluginManifest(
                name=f"concurrent_plugin_{plugin_id}",
                version="1.0.0",
                format_adapters=[spec]
            )
            
            await registry.register_from_manifest(manifest, f"concurrent_plugin_{plugin_id}")
        
        # Register plugins concurrently
        start_time = time.perf_counter()
        await asyncio.gather(*[
            register_plugin(i) for i in range(num_concurrent)
        ])
        concurrent_time = time.perf_counter() - start_time
        
        await registry.resolve_conflicts_and_finalize(enable_priority_mode=True)
        
        # Should handle concurrent access gracefully
        assert len(registry._registered_plugins) == num_concurrent
        assert len(registry._adapters) == 1  # Only one should win due to conflicts
        
        print(f"Performance: {num_concurrent} concurrent registrations in {concurrent_time:.3f}s")

    @pytest.mark.asyncio
    async def test_memory_usage_with_large_registry(self, registry):
        """Test memory efficiency with large number of adapters."""
        import sys
        
        # Measure initial memory
        initial_size = sys.getsizeof(registry._adapters) + sys.getsizeof(registry._adapter_specs)
        
        # Add many adapters
        num_adapters = 1000
        for i in range(num_adapters):
            spec = FormatAdapterSpec(
                from_format=f"memory_from_{i}",
                to_format=f"memory_to_{i}",
                adapter_factory=lambda: BenchmarkAdapter(f"memory_{i}"),
                priority=i
            )
            
            manifest = PluginManifest(
                name=f"memory_plugin_{i}",
                version="1.0.0",
                format_adapters=[spec]
            )
            
            await registry.register_from_manifest(manifest, f"memory_plugin_{i}")
        
        # Finalize
        await registry.resolve_conflicts_and_finalize()
        
        # Measure final memory
        final_size = sys.getsizeof(registry._adapters) + sys.getsizeof(registry._adapter_specs)
        memory_per_adapter = (final_size - initial_size) / num_adapters
        
        # Memory usage should be reasonable
        assert memory_per_adapter < 1000, f"Memory per adapter: {memory_per_adapter:.0f} bytes"
        
        print(f"Memory: ~{memory_per_adapter:.0f} bytes per adapter")

    def test_format_adapter_spec_creation_performance(self):
        """Test performance of creating many FormatAdapterSpec instances."""
        num_specs = 10000
        
        start_time = time.perf_counter()
        
        specs = []
        for i in range(num_specs):
            spec = FormatAdapterSpec(
                from_format=f"perf_from_{i}",
                to_format=f"perf_to_{i}",
                adapter_factory=lambda: BenchmarkAdapter(),
                priority=i,
                description=f"Performance test adapter {i}"
            )
            specs.append(spec)
        
        creation_time = time.perf_counter() - start_time
        
        # Should create specs quickly
        assert creation_time < 1.0, f"Creating {num_specs} specs took {creation_time:.2f}s"
        
        # Verify all specs are valid
        for spec in specs[:10]:  # Check first 10
            assert spec.format_pair[0].startswith("perf_from_")
            assert spec.format_pair[1].startswith("perf_to_")
        
        print(f"Performance: Created {num_specs} FormatAdapterSpecs in {creation_time:.3f}s")


@pytest.mark.performance
class TestConflictResolutionValidation:
    """Validation tests for conflict resolution behavior."""

    @pytest.fixture
    def priority_registry(self):
        return FormatAdapterRegistry(conflict_mode="priority")

    @pytest.fixture
    def fail_fast_registry(self):
        return FormatAdapterRegistry(conflict_mode="fail_fast")

    @pytest.mark.asyncio
    async def test_priority_conflict_resolution_correctness(self, priority_registry):
        """Test that priority-based conflict resolution works correctly."""
        # Create adapters with specific priorities
        priorities = [100, 10, 50, 5, 75]  # 5 should win (lowest)
        expected_winner_priority = 5
        
        for i, priority in enumerate(priorities):
            spec = FormatAdapterSpec(
                from_format="priority_test",
                to_format="target",
                adapter_factory=lambda p=priority: BenchmarkAdapter(f"priority_{p}"),
                priority=priority
            )
            
            manifest = PluginManifest(
                name=f"priority_plugin_{i}",
                version="1.0.0",
                format_adapters=[spec]
            )
            
            await priority_registry.register_from_manifest(manifest, f"priority_plugin_{i}")
        
        # Resolve conflicts
        await priority_registry.resolve_conflicts_and_finalize()
        
        # Verify the correct adapter won
        winning_spec = priority_registry._adapter_specs[("priority_test", "target")]
        assert winning_spec.priority == expected_winner_priority

    @pytest.mark.asyncio
    async def test_fail_fast_conflict_detection(self, fail_fast_registry):
        """Test that fail-fast mode properly detects and reports conflicts."""
        # Create conflicting adapters
        for i in range(3):
            spec = FormatAdapterSpec(
                from_format="fail_fast_test",
                to_format="target", 
                adapter_factory=lambda: BenchmarkAdapter(f"fail_fast_{i}"),
                priority=i * 10
            )
            
            manifest = PluginManifest(
                name=f"fail_fast_plugin_{i}",
                version="1.0.0",
                format_adapters=[spec]
            )
            
            await fail_fast_registry.register_from_manifest(manifest, f"fail_fast_plugin_{i}")
        
        # Should raise ValueError on conflicts in fail-fast mode
        with pytest.raises(ValueError, match="Format adapter conflicts detected"):
            await fail_fast_registry.resolve_conflicts_and_finalize()

    @pytest.mark.asyncio
    async def test_complex_conflict_scenario(self, priority_registry):
        """Test complex conflict scenarios with multiple overlapping conflicts."""
        # Create a complex conflict graph:
        # - Format A->B: 3 plugins compete
        # - Format A->C: 2 plugins compete  
        # - Format B->C: 2 plugins compete
        # - No conflicts for D->E
        
        conflicts = [
            ("A", "B", [10, 20, 5]),  # Plugin with priority 5 should win
            ("A", "C", [15, 8]),      # Plugin with priority 8 should win
            ("B", "C", [25, 12]),     # Plugin with priority 12 should win
            ("D", "E", [30]),         # No conflict
        ]
        
        plugin_counter = 0
        expected_winners = {}
        
        for from_fmt, to_fmt, priorities in conflicts:
            expected_winners[(from_fmt, to_fmt)] = min(priorities)
            
            for priority in priorities:
                spec = FormatAdapterSpec(
                    from_format=from_fmt,
                    to_format=to_fmt,
                    adapter_factory=lambda: BenchmarkAdapter(f"complex_{plugin_counter}"),
                    priority=priority
                )
                
                manifest = PluginManifest(
                    name=f"complex_plugin_{plugin_counter}",
                    version="1.0.0",
                    format_adapters=[spec]
                )
                
                await priority_registry.register_from_manifest(manifest, f"complex_plugin_{plugin_counter}")
                plugin_counter += 1
        
        # Resolve all conflicts
        await priority_registry.resolve_conflicts_and_finalize()
        
        # Verify all conflicts were resolved correctly
        for format_pair, expected_priority in expected_winners.items():
            winning_spec = priority_registry._adapter_specs[format_pair]
            assert winning_spec.priority == expected_priority, (
                f"Wrong winner for {format_pair}: got priority {winning_spec.priority}, "
                f"expected {expected_priority}"
            )

    @pytest.mark.asyncio
    async def test_requirement_validation_edge_cases(self, priority_registry):
        """Test edge cases in requirement validation."""
        # Plugin with circular requirements (should be handled gracefully)
        manifest1 = PluginManifest(
            name="circular1",
            version="1.0.0",
            format_adapters=[
                FormatAdapterSpec(
                    from_format="A", to_format="B",
                    adapter_factory=lambda: BenchmarkAdapter(),
                    priority=10
                )
            ],
            requires_format_adapters=[("B", "A")]
        )
        
        manifest2 = PluginManifest(
            name="circular2", 
            version="1.0.0",
            format_adapters=[
                FormatAdapterSpec(
                    from_format="B", to_format="A",
                    adapter_factory=lambda: BenchmarkAdapter(),
                    priority=10
                )
            ],
            requires_format_adapters=[("A", "B")]
        )
        
        await priority_registry.register_from_manifest(manifest1, "circular1")
        await priority_registry.register_from_manifest(manifest2, "circular2")
        
        # Should validate requirements correctly despite circular dependencies
        manifests = {"circular1": manifest1, "circular2": manifest2}
        missing = priority_registry.validate_requirements(manifests)
        
        # After both are registered, requirements should be satisfied
        assert len(missing) == 0 or all(len(reqs) == 0 for reqs in missing.values())
        
        await priority_registry.resolve_conflicts_and_finalize()
        
        # Both adapters should be available
        assert len(priority_registry._adapters) == 2