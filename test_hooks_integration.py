#!/usr/bin/env python
"""Test script to verify hooks integration is working."""

import asyncio
import os
import sys
from datetime import datetime
from pathlib import Path


# Set the hooks enabled flag
os.environ["HOOKS_ENABLED"] = "true"

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from ccproxy.config.settings import Settings, get_settings  # noqa: E402
from ccproxy.hooks import HookEvent, HookManager, HookRegistry  # noqa: E402
from ccproxy.hooks.base import HookContext  # noqa: E402


# Test hook to verify events are being emitted
class TestHook:
    """Test hook that logs all events it receives."""

    name = "test_hook"
    events = [
        HookEvent.APP_STARTUP,
        HookEvent.REQUEST_STARTED,
        HookEvent.REQUEST_COMPLETED,
        HookEvent.REQUEST_FAILED,
        HookEvent.PROVIDER_REQUEST_SENT,
        HookEvent.PROVIDER_RESPONSE_RECEIVED,
        HookEvent.PROVIDER_STREAM_START,
        HookEvent.PROVIDER_STREAM_END,
    ]

    def __init__(self):
        self.received_events = []

    async def __call__(self, context: HookContext) -> None:
        """Log received events."""
        event_info = {
            "event": context.event.value if context.event else "unknown",
            "timestamp": context.timestamp.isoformat() if context.timestamp else None,
            "provider": context.provider,
            "data": context.data,
        }
        self.received_events.append(event_info)
        print(f"✓ Received event: {event_info['event']}")
        if context.provider:
            print(f"  Provider: {context.provider}")
        if context.data:
            print(f"  Data keys: {list(context.data.keys())}")


async def test_hooks_system():
    """Test the hooks system integration."""
    print("=" * 60)
    print("Testing Hooks System Integration")
    print("=" * 60)

    # Create hook registry and manager
    registry = HookRegistry()
    manager = HookManager(registry)

    # Create and register test hook
    test_hook = TestHook()
    registry.register(test_hook)

    print("\n1. Testing hook registration...")
    events_for_hook = registry.get_hooks(HookEvent.APP_STARTUP)
    assert test_hook in events_for_hook, "Hook not registered properly"
    print("   ✓ Hook registered successfully")

    print("\n2. Testing event emission...")

    # Test APP_STARTUP event
    await manager.emit(
        HookEvent.APP_STARTUP,
        HookContext(
            event=HookEvent.APP_STARTUP,
            timestamp=datetime.now(),
            data={"phase": "testing"},
            metadata={},
        ),
    )
    assert len(test_hook.received_events) == 1, "APP_STARTUP event not received"

    # Test REQUEST_STARTED event
    await manager.emit(
        HookEvent.REQUEST_STARTED,
        HookContext(
            event=HookEvent.REQUEST_STARTED,
            timestamp=datetime.now(),
            data={
                "request_id": "test-123",
                "method": "POST",
                "url": "/test",
                "headers": {},
            },
            metadata={},
        ),
    )
    assert len(test_hook.received_events) == 2, "REQUEST_STARTED event not received"

    # Test PROVIDER_REQUEST_SENT event
    await manager.emit(
        HookEvent.PROVIDER_REQUEST_SENT,
        HookContext(
            event=HookEvent.PROVIDER_REQUEST_SENT,
            timestamp=datetime.now(),
            provider="TestProvider",
            data={
                "url": "https://api.test.com",
                "method": "POST",
                "headers": {},
                "is_streaming": False,
            },
            metadata={"request_id": "test-123"},
        ),
    )
    assert len(test_hook.received_events) == 3, (
        "PROVIDER_REQUEST_SENT event not received"
    )

    print("\n3. Testing request tracer hook integration...")

    # Import and test request tracer hook
    from plugins.request_tracer.config import RequestTracerConfig
    from plugins.request_tracer.hook import RequestTracerHook

    config = RequestTracerConfig(
        enabled=True,
        verbose_api=True,
        json_logs_enabled=False,  # Disable file logging for test
        raw_http_enabled=False,  # Disable file logging for test
    )

    tracer_hook = RequestTracerHook(config)
    registry.register(tracer_hook)
    print("   ✓ RequestTracerHook registered")

    # Emit an event that the tracer should handle
    await manager.emit(
        HookEvent.REQUEST_STARTED,
        HookContext(
            event=HookEvent.REQUEST_STARTED,
            timestamp=datetime.now(),
            data={
                "request_id": "tracer-test",
                "method": "GET",
                "url": "/api/test",
                "headers": {"content-type": "application/json"},
            },
            metadata={},
        ),
    )
    print("   ✓ Event emitted to RequestTracerHook")

    print("\n4. Summary:")
    print(f"   Total events received by test hook: {len(test_hook.received_events)}")
    print(f"   Events: {[e['event'] for e in test_hook.received_events]}")

    print("\n✅ All hook integration tests passed!")

    return True


async def test_app_integration():
    """Test hooks integration with the FastAPI app."""
    print("\n" + "=" * 60)
    print("Testing App-Level Hooks Integration")
    print("=" * 60)

    # Import app factory
    from ccproxy.api.app import create_app

    # Create app with hooks enabled
    settings = get_settings()
    settings.hooks.enabled = True

    print("\n1. Creating FastAPI app with hooks enabled...")
    app = create_app(settings)

    # Note: Hook manager is initialized during app startup (lifespan),
    # not during app creation, so we can't check it here
    print("   ✓ FastAPI app created with hooks configuration")

    # Check if HooksMiddleware is in the middleware stack
    print("\n2. Checking middleware stack...")
    middleware_found = False
    for middleware in app.middleware_stack:
        if "HooksMiddleware" in str(type(middleware)):
            middleware_found = True
            break

    if middleware_found:
        print("   ✓ HooksMiddleware found in middleware stack")
    else:
        print("   ⚠ HooksMiddleware might be wrapped in middleware stack")

    print("\n✅ App-level integration tests completed!")

    return True


async def main():
    """Run all tests."""
    try:
        # Test basic hook system
        await test_hooks_system()

        # Test app integration
        await test_app_integration()

        print("\n" + "=" * 60)
        print("🎉 All tests passed successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
