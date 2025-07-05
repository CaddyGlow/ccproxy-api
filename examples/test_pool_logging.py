#!/usr/bin/env python3
"""Test script to demonstrate connection pool logging."""

import asyncio
import json
import logging

import httpx


# Configure logging to show all pool-related messages
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


async def make_request(client: httpx.AsyncClient, request_num: int) -> None:
    """Make a single request to the API."""
    print(f"\n--- Request #{request_num} ---")

    data = {
        "model": "claude-3-5-sonnet-20241022",
        "messages": [{"role": "user", "content": f"What is 2 + {request_num}?"}],
        "max_tokens": 50,
    }

    try:
        response = await client.post(
            "http://localhost:8000/v1/messages",
            json=data,
            headers={"Content-Type": "application/json"},
        )

        if response.status_code == 200:
            result = response.json()
            print(f"Response #{request_num}: {result['content'][0]['text']}")
        else:
            print(f"Error in request #{request_num}: {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"Failed request #{request_num}: {e}")


async def main() -> None:
    """Run multiple requests to demonstrate pool behavior."""
    print("Starting pool logging demonstration...")
    print("=" * 60)
    print("Watch the server logs to see pool operations!")
    print("=" * 60)

    # Create HTTP client
    async with httpx.AsyncClient() as client:
        # Make sequential requests to show pool reuse
        print("\n1. Making sequential requests (should reuse connections):")
        for i in range(1, 4):
            await make_request(client, i)
            await asyncio.sleep(1)  # Small delay to see logs clearly

        # Make concurrent requests to show pool expansion
        print("\n\n2. Making concurrent requests (may create new connections):")
        tasks = [make_request(client, i) for i in range(10, 16)]
        await asyncio.gather(*tasks)

        # Wait and make another request to show reuse after concurrent load
        print("\n\n3. Making request after concurrent load (should reuse from pool):")
        await asyncio.sleep(2)
        await make_request(client, 20)

    print("\n" + "=" * 60)
    print("Demonstration complete! Check server logs for pool activity.")


if __name__ == "__main__":
    asyncio.run(main())
