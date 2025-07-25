#!/usr/bin/env python3
"""
Debug script for testing Anthropic streaming API with full request/response logging.

Usage:
    export ANTHROPIC_API_KEY="your-api-key"
    export ANTHROPIC_BASE_URL="https://api.anthropic.com"  # or your proxy URL
    python debug_anthropic_streaming.py
"""

import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx
from anthropic import Anthropic, Stream
from anthropic.types import Message, MessageStreamEvent


# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            f"anthropic_debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        ),
    ],
)
logger = logging.getLogger(__name__)

# Disable httpx info logs to reduce noise
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


class AnthropicDebugClient:
    """Debug wrapper for Anthropic client with request/response logging."""

    def __init__(self, api_key: str | None = None, base_url: str | None = None):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.base_url = base_url or os.environ.get(
            "ANTHROPIC_BASE_URL", "https://api.anthropic.com"
        )

        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable is required")

        logger.info(f"Initializing Anthropic client with base_url: {self.base_url}")

        # Create output directory for debug files
        self.debug_dir = Path(
            f"anthropic_debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        self.debug_dir.mkdir(exist_ok=True)
        logger.info(f"Debug output directory: {self.debug_dir}")

        # Track requests for file naming
        self.request_counter = 0

        # Create client with custom httpx client for debugging
        self.client = Anthropic(
            api_key=self.api_key,
            base_url=self.base_url,
            http_client=self._create_debug_http_client(),
        )

    def _create_debug_http_client(self) -> httpx.Client:
        """Create httpx client with request/response hooks for debugging."""

        def log_request(request: httpx.Request) -> None:
            self.request_counter += 1
            request_id = f"request_{self.request_counter:03d}"

            # Save request to file
            request_data = {
                "timestamp": datetime.now().isoformat(),
                "method": request.method,
                "url": str(request.url),
                "headers": dict(request.headers),
                "body": None,
            }

            if request.content:
                try:
                    request_data["body"] = json.loads(request.content)
                except Exception:
                    request_data["body"] = request.content.decode(
                        "utf-8", errors="replace"
                    )

            request_file = self.debug_dir / f"{request_id}_request.json"
            with request_file.open("w") as f:
                json.dump(request_data, f, indent=2)

            # Also store request_id on the request object for response matching
            request.extensions["request_id"] = request_id

            logger.info("=" * 80)
            logger.info(f"REQUEST [{request_id}]")
            logger.info("=" * 80)
            logger.info(f"Method: {request.method}")
            logger.info(f"URL: {request.url}")
            logger.info("Headers:")
            for name, value in request.headers.items():
                # Mask sensitive headers
                if name.lower() in ["x-api-key", "authorization"]:
                    value = value[:10] + "..." if len(value) > 10 else value
                logger.info(f"  {name}: {value}")

            if request.content:
                try:
                    body = json.loads(request.content)
                    logger.info("Body (JSON):")
                    logger.info(json.dumps(body, indent=2))
                except Exception:
                    logger.info(f"Body (raw): {request.content}")
            logger.info("=" * 80)
            logger.info(f"Request saved to: {request_file}")

        def log_response(response: httpx.Response) -> None:
            request_id = response.request.extensions.get("request_id", "unknown")

            # Save response to file
            response_data = {
                "timestamp": datetime.now().isoformat(),
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "body": None,
            }

            # For streaming responses, we'll save chunks separately
            if "text/event-stream" not in response.headers.get("content-type", ""):
                try:
                    response_data["body"] = response.json()
                except Exception:
                    response_data["body"] = response.text
            else:
                response_data["body"] = "Streaming response - see chunks file"

            response_file = self.debug_dir / f"{request_id}_response.json"
            with response_file.open("w") as f:
                json.dump(response_data, f, indent=2)

            logger.info("=" * 80)
            logger.info(f"RESPONSE [{request_id}]")
            logger.info("=" * 80)
            logger.info(f"Status: {response.status_code}")
            logger.info("Headers:")
            for name, value in response.headers.items():
                logger.info(f"  {name}: {value}")

            # For streaming responses, we'll log chunks separately
            if "text/event-stream" not in response.headers.get("content-type", ""):
                try:
                    body = response.json()
                    logger.info("Body (JSON):")
                    logger.info(json.dumps(body, indent=2))
                except Exception:
                    logger.info(f"Body (raw): {response.text[:1000]}...")
            logger.info("=" * 80)
            logger.info(f"Response saved to: {response_file}")

        return httpx.Client(
            event_hooks={
                "request": [log_request],
                "response": [log_response],
            }
        )

    def test_streaming(self) -> None:
        """Test streaming message creation with detailed logging."""

        logger.info("\n" + "=" * 80)
        logger.info("STARTING STREAMING TEST")
        logger.info("=" * 80)

        test_message = {
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 100,
            "messages": [
                {
                    "role": "user",
                    "content": "Count from 1 to 10 slowly, with a brief description for each number.",
                }
            ],
            "stream": True,
            "temperature": 0.7,
        }

        logger.info("Test message configuration:")
        logger.info(json.dumps(test_message, indent=2))

        try:
            # Create streaming message
            stream: Stream[MessageStreamEvent] = self.client.messages.create(
                **test_message
            )

            logger.info("\n" + "-" * 80)
            logger.info("STREAMING EVENTS")
            logger.info("-" * 80)

            event_count = 0
            content_chunks = []
            all_events = []

            for event in stream:
                event_count += 1
                timestamp = datetime.now().isoformat()

                # Capture event data
                event_data = {
                    "event_number": event_count,
                    "timestamp": timestamp,
                    "type": event.type,
                    "data": {},
                }

                # Log each streaming event
                logger.info(f"\n[Event {event_count}] {timestamp}")
                logger.info(f"Type: {event.type}")

                # Log event details based on type
                if hasattr(event, "message"):
                    logger.info(f"Message ID: {event.message.id}")
                    logger.info(f"Model: {event.message.model}")
                    logger.info(f"Role: {event.message.role}")
                    logger.info(f"Stop reason: {event.message.stop_reason}")
                    logger.info(f"Usage: {event.message.usage}")
                    event_data["data"]["message"] = {
                        "id": event.message.id,
                        "model": event.message.model,
                        "role": event.message.role,
                        "stop_reason": event.message.stop_reason,
                        "usage": event.message.usage.model_dump()
                        if event.message.usage
                        else None,
                    }

                if hasattr(event, "content_block"):
                    logger.info(f"Content block type: {event.content_block.type}")
                    event_data["data"]["content_block"] = {
                        "type": event.content_block.type
                    }
                    if hasattr(event.content_block, "text"):
                        logger.info(f"Content block text: {event.content_block.text}")
                        event_data["data"]["content_block"]["text"] = (
                            event.content_block.text
                        )

                if hasattr(event, "delta"):
                    logger.info(f"Delta type: {event.delta.type}")
                    event_data["data"]["delta"] = {"type": event.delta.type}
                    if hasattr(event.delta, "text"):
                        logger.info(f"Delta text: '{event.delta.text}'")
                        content_chunks.append(event.delta.text)
                        event_data["data"]["delta"]["text"] = event.delta.text
                    if hasattr(event.delta, "stop_reason"):
                        logger.info(f"Delta stop reason: {event.delta.stop_reason}")
                        event_data["data"]["delta"]["stop_reason"] = (
                            event.delta.stop_reason
                        )

                if hasattr(event, "usage"):
                    logger.info(f"Usage update: {event.usage}")
                    event_data["data"]["usage"] = (
                        event.usage.model_dump() if event.usage else None
                    )

                # Log raw event data
                logger.debug(f"Raw event: {event}")
                all_events.append(event_data)

            # Save all streaming events to file
            chunks_file = (
                self.debug_dir
                / f"request_{self.request_counter:03d}_streaming_chunks.json"
            )
            with chunks_file.open("w") as f:
                json.dump(
                    {
                        "total_events": event_count,
                        "complete_content": "".join(content_chunks),
                        "events": all_events,
                    },
                    f,
                    indent=2,
                )

            logger.info("\n" + "-" * 80)
            logger.info("STREAMING COMPLETE")
            logger.info("-" * 80)
            logger.info(f"Total events received: {event_count}")
            logger.info(f"Complete response: {''.join(content_chunks)}")
            logger.info(f"Streaming chunks saved to: {chunks_file}")

        except Exception as e:
            logger.error(f"Error during streaming test: {e}", exc_info=True)
            raise

    def test_non_streaming(self) -> None:
        """Test non-streaming message creation for comparison."""

        logger.info("\n" + "=" * 80)
        logger.info("STARTING NON-STREAMING TEST")
        logger.info("=" * 80)

        test_message = {
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "What is 2 + 2?"}],
            "stream": False,
            "temperature": 0,
        }

        logger.info("Test message configuration:")
        logger.info(json.dumps(test_message, indent=2))

        try:
            # Create non-streaming message
            response: Message = self.client.messages.create(**test_message)

            logger.info("\n" + "-" * 80)
            logger.info("RESPONSE DETAILS")
            logger.info("-" * 80)
            logger.info(f"Message ID: {response.id}")
            logger.info(f"Model: {response.model}")
            logger.info(f"Role: {response.role}")
            logger.info(f"Content: {response.content}")
            logger.info(f"Stop reason: {response.stop_reason}")
            logger.info(f"Usage: {response.usage}")

        except Exception as e:
            logger.error(f"Error during non-streaming test: {e}", exc_info=True)
            raise


def main():
    """Main function to run debug tests."""

    # Print environment info
    logger.info("Environment Configuration:")
    logger.info(
        f"ANTHROPIC_BASE_URL: {os.environ.get('ANTHROPIC_BASE_URL', 'not set')}"
    )
    logger.info(
        f"ANTHROPIC_API_KEY: {'set' if os.environ.get('ANTHROPIC_API_KEY') else 'not set'}"
    )

    try:
        # Create debug client
        client = AnthropicDebugClient()

        # Run non-streaming test first
        logger.info("\n" + "#" * 80)
        logger.info("TEST 1: NON-STREAMING REQUEST")
        logger.info("#" * 80)
        client.test_non_streaming()

        # Wait a bit between tests
        time.sleep(1)

        # Run streaming test
        logger.info("\n" + "#" * 80)
        logger.info("TEST 2: STREAMING REQUEST")
        logger.info("#" * 80)
        client.test_streaming()

        logger.info("\n" + "#" * 80)
        logger.info("ALL TESTS COMPLETED SUCCESSFULLY")
        logger.info("#" * 80)

        # List all generated files
        logger.info("\nGenerated debug files:")
        logger.info(f"Directory: {client.debug_dir}")
        for file in sorted(client.debug_dir.iterdir()):
            logger.info(f"  - {file.name}")

    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
