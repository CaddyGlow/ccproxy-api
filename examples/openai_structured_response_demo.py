#!/usr/bin/env python3
"""
OpenAI SDK Structured Response Demo

This example demonstrates using the OpenAI SDK with the Claude Code Proxy API
to show the full response JSON structure using structured logging and argparse.
"""

import argparse
import json
import logging
import sys
from typing import Optional

import openai
from pydantic import BaseModel, Field


# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


class ChatCompletionResponse(BaseModel):
    """Structured representation of OpenAI chat completion response"""

    id: str
    object: str
    created: int
    model: str
    system_fingerprint: str | None = None
    choices: list[dict]
    usage: dict


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="OpenAI SDK demo showing full response JSON structure"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default="test-key",
        help="API key for authentication (default: test-key)",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://localhost:8000/openai/v1",
        help="Base URL for the API (default: http://localhost:8000/openai/v1)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="claude-3-5-sonnet-20241022",
        help="Model to use (default: claude-3-5-sonnet-20241022)",
    )
    parser.add_argument(
        "--message",
        type=str,
        default="What is 2+2? Answer in one word.",
        help="Message to send to the model",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for response generation (default: 0.7)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=100,
        help="Maximum tokens in response (default: 100)",
    )
    parser.add_argument("--stream", action="store_true", help="Enable streaming mode")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    return parser.parse_args()


def log_request_details(args):
    """Log request configuration details"""
    logger.info("Request Configuration:")
    logger.info(f"  API Base URL: {args.base_url}")
    logger.info(f"  Model: {args.model}")
    logger.info(f"  Temperature: {args.temperature}")
    logger.info(f"  Max Tokens: {args.max_tokens}")
    logger.info(f"  Streaming: {args.stream}")
    logger.info(f"  Message: {args.message}")


def log_response_structure(response_data: dict):
    """Log the full response structure in a formatted way"""
    logger.info("Full Response JSON Structure:")
    logger.info("=" * 80)

    # Log top-level fields
    logger.info(f"ID: {response_data.get('id')}")
    logger.info(f"Object: {response_data.get('object')}")
    logger.info(f"Created: {response_data.get('created')}")
    logger.info(f"Model: {response_data.get('model')}")
    logger.info(f"System Fingerprint: {response_data.get('system_fingerprint')}")

    # Log choices
    logger.info("\nChoices:")
    for i, choice in enumerate(response_data.get("choices", [])):
        logger.info(f"  Choice {i}:")
        logger.info(f"    Index: {choice.get('index')}")
        logger.info(f"    Finish Reason: {choice.get('finish_reason')}")

        message = choice.get("message", {})
        logger.info("    Message:")
        logger.info(f"      Role: {message.get('role')}")
        logger.info(f"      Content: {message.get('content')}")

        if "tool_calls" in message:
            logger.info(f"      Tool Calls: {message.get('tool_calls')}")

    # Log usage
    usage = response_data.get("usage", {})
    logger.info("\nUsage:")
    logger.info(f"  Prompt Tokens: {usage.get('prompt_tokens')}")
    logger.info(f"  Completion Tokens: {usage.get('completion_tokens')}")
    logger.info(f"  Total Tokens: {usage.get('total_tokens')}")

    logger.info("=" * 80)


def handle_streaming_response(stream):
    """Handle streaming response and collect full structure"""
    logger.info("Streaming Response:")
    logger.info("-" * 80)

    full_content = ""
    chunks = []

    for chunk in stream:
        chunk_dict = chunk.model_dump()
        chunks.append(chunk_dict)

        if args.verbose:
            logger.debug(f"Chunk: {json.dumps(chunk_dict, indent=2)}")

        # Extract content from delta
        for choice in chunk.choices:
            if choice.delta and choice.delta.content:
                full_content += choice.delta.content
                print(choice.delta.content, end="", flush=True)

    print()  # New line after streaming
    logger.info("-" * 80)

    # Log aggregated streaming data
    logger.info(f"Total chunks received: {len(chunks)}")
    logger.info(f"Full content: {full_content}")

    if chunks and args.verbose:
        logger.info("\nFirst chunk structure:")
        logger.info(json.dumps(chunks[0], indent=2))
        logger.info("\nLast chunk structure:")
        logger.info(json.dumps(chunks[-1], indent=2))


def main(args):
    """Main function to demonstrate OpenAI SDK usage"""
    # Configure OpenAI client
    client = openai.OpenAI(api_key=args.api_key, base_url=args.base_url)

    # Log request details
    log_request_details(args)

    try:
        # Prepare messages
        messages = [{"role": "user", "content": args.message}]

        logger.info("\nSending request to API...")

        if args.stream:
            # Streaming mode
            stream = client.chat.completions.create(
                model=args.model,
                messages=messages,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                stream=True,
            )
            handle_streaming_response(stream)
        else:
            # Non-streaming mode
            response = client.chat.completions.create(
                model=args.model,
                messages=messages,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )

            # Convert to dict for detailed logging
            response_dict = response.model_dump()

            # Log the full response structure
            log_response_structure(response_dict)

            # Log raw JSON if verbose
            if args.verbose:
                logger.info("\nRaw JSON Response:")
                logger.info(json.dumps(response_dict, indent=2))

            # Extract and display the assistant's response
            assistant_message = response.choices[0].message.content
            logger.info(f"\nAssistant Response: {assistant_message}")

            # Validate response structure
            try:
                validated_response = ChatCompletionResponse(**response_dict)
                logger.info("\n✓ Response structure validated successfully")
            except Exception as e:
                logger.error(f"\n✗ Response validation failed: {e}")

    except Exception as e:
        logger.error(f"Error during API call: {e}")
        if args.verbose:
            logger.exception("Full exception details:")
        sys.exit(1)


if __name__ == "__main__":
    args = parse_arguments()

    # Set logging level based on verbosity
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    main(args)
