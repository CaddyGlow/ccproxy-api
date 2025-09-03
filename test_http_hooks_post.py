#!/usr/bin/env python3
"""Quick test to see if POST request bodies are captured in debug logs."""

import asyncio
import json

from ccproxy.core.http_client_hooks import HookableHTTPClient


async def test_post_body_in_logs():
    """Test POST request and check debug logs for body data."""
    print("🧪 Testing POST Request Body Capture in Debug Logs")
    print("=" * 55)

    # Create HTTP client without hooks first to see raw behavior
    client = HookableHTTPClient()

    test_data = {"message": "Hello World", "test": True, "number": 42}

    print(f"📤 Sending POST with JSON body: {json.dumps(test_data)}")
    print("🔍 Check the debug logs for 'data_keys' to see what's captured...")

    try:
        response = await client.post("https://httpbin.org/post", json=test_data)
        print(f"✅ Response status: {response.status_code}")

    except Exception as e:
        print(f"❌ Request failed: {e}")
    finally:
        await client.aclose()


if __name__ == "__main__":
    asyncio.run(test_post_body_in_logs())
