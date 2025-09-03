#!/usr/bin/env python3
"""Test script to verify HTTP client hooks are working."""

import asyncio
import sys
from pathlib import Path


# Add the project to path
sys.path.insert(0, str(Path(__file__).parent))

from ccproxy.core.http_client import HTTPClientFactory
from ccproxy.hooks.manager import HookManager
from ccproxy.hooks.registry import HookRegistry
from ccproxy.plugins.request_tracer.config import RequestTracerConfig
from ccproxy.plugins.request_tracer.hook import RequestTracerHook


async def test_http_client_hooks():
    """Test that HTTP client hooks work correctly."""
    print("🧪 Testing HTTP Client Hooks Integration")
    print("=" * 50)

    # Create hook system
    hook_registry = HookRegistry()
    hook_manager = HookManager(hook_registry)

    # Create request tracer hook
    config = RequestTracerConfig(enabled=True, log_dir="/tmp/ccproxy/test_hooks")
    tracer_hook = RequestTracerHook(config)

    # Register the hook
    hook_registry.register(tracer_hook)
    print(f"✅ Registered hook: {tracer_hook.name}")
    print(f"📋 Hook events: {[e.value for e in tracer_hook.events]}")

    # Create HTTP client with hooks
    http_client = HTTPClientFactory.create_client(
        hook_manager=hook_manager, timeout_connect=5.0, timeout_read=10.0
    )

    # Check if HTTP client has hook manager
    has_hook_manager = (
        hasattr(http_client, "hook_manager") and http_client.hook_manager is not None
    )
    print(f"🔗 HTTP client has hook manager: {has_hook_manager}")
    if has_hook_manager:
        print(f"🆔 HTTP client hook manager ID: {id(http_client.hook_manager)}")
        print(f"🆔 Created hook manager ID: {id(hook_manager)}")
        print(f"🔄 Hook managers match: {http_client.hook_manager is hook_manager}")

    # Test making an HTTP request with POST body
    print("\n🌐 Making test HTTP POST request...")
    try:
        test_data = {"message": "Hello World", "test": True, "number": 42}
        response = await http_client.post(
            "https://httpbin.org/post", json=test_data, timeout=10.0
        )
        print(f"📡 Response status: {response.status_code}")
        print(f"📦 Response size: {len(response.content)} bytes")

        # Check if trace files were created
        trace_dir = Path("/tmp/ccproxy/test_hooks")
        if trace_dir.exists():
            trace_files = list(trace_dir.glob("*.json"))
            print(f"📁 Trace files created: {len(trace_files)}")
            if trace_files:
                print("📄 Latest trace file:", trace_files[-1].name)
        else:
            print("❌ No trace directory found")

    except Exception as e:
        print(f"❌ HTTP request failed: {e}")

    # Clean up
    await http_client.aclose()
    print("\n✅ Test completed")


if __name__ == "__main__":
    asyncio.run(test_http_client_hooks())
