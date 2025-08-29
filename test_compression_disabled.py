#!/usr/bin/env python3
"""Test script to verify HTTP compression is disabled when raw HTTP tracing is enabled."""

import subprocess
import time
import os
import json
import sys

def test_compression_disabled():
    """Test that compression is disabled when raw HTTP tracing is enabled."""
    
    print("Starting server with raw HTTP tracing enabled...")
    
    # Start server with raw HTTP tracing enabled
    env = os.environ.copy()
    env.update({
        "PLUGINS__REQUEST_TRACER__ENABLED": "true",
        "PLUGINS__REQUEST_TRACER__RAW_HTTP_ENABLED": "true",
    })
    
    server = subprocess.Popen(
        ["uv", "run", "ccproxy-api", "serve", "--port", "8002"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    
    # Wait for server to start
    time.sleep(5)
    
    try:
        print("\nMaking test request...")
        # Make a test request
        result = subprocess.run(
            [
                "curl", "-X", "POST",
                "http://127.0.0.1:8002/api/v1/messages",
                "-H", "Content-Type: application/json",
                "-H", "x-api-key: test-key",
                "-H", "anthropic-version: 2023-06-01",
                "-d", json.dumps({
                    "model": "claude-3-5-sonnet-20241022",
                    "messages": [{"role": "user", "content": "Reply with just: test"}],
                    "max_tokens": 10
                })
            ],
            capture_output=True,
            text=True
        )
        
        print(f"Request status: {'SUCCESS' if result.returncode == 0 else 'FAILED'}")
        
        # Check the latest raw HTTP capture
        time.sleep(1)
        raw_dir = "/tmp/ccproxy/tracer/raw"
        if os.path.exists(raw_dir):
            files = sorted([f for f in os.listdir(raw_dir) if f.endswith("_client_response.http")],
                          key=lambda x: os.path.getmtime(os.path.join(raw_dir, x)))
            
            if files:
                latest_file = os.path.join(raw_dir, files[-1])
                print(f"\nChecking raw HTTP capture: {files[-1]}")
                
                with open(latest_file, 'r') as f:
                    content = f.read()
                    
                # Check for signs of compression
                if "content-encoding: gzip" in content.lower():
                    print("❌ FAILED: Response is compressed (content-encoding: gzip found)")
                    return False
                    
                # Check if JSON is readable
                if '"type":"message"' in content and '"role":"assistant"' in content:
                    print("✅ SUCCESS: Response is uncompressed (JSON is readable)")
                    
                    # Extract just the headers
                    headers_end = content.find("\n\n")
                    if headers_end > 0:
                        headers = content[:headers_end]
                        print("\nResponse headers:")
                        for line in headers.split("\n"):
                            if line.strip():
                                print(f"  {line}")
                    return True
                else:
                    # Check if it looks like compressed data
                    if any(ord(c) < 32 or ord(c) > 126 for c in content[100:200] if c not in '\n\r\t'):
                        print("❌ FAILED: Response appears to be compressed (binary data found)")
                        return False
                    else:
                        print("⚠️ WARNING: Could not determine compression status")
                        return None
        
        print("❌ No raw HTTP captures found")
        return False
        
    finally:
        # Stop server
        print("\nStopping server...")
        server.terminate()
        server.wait(timeout=5)

if __name__ == "__main__":
    result = test_compression_disabled()
    sys.exit(0 if result else 1)
