#!/usr/bin/env python3
"""
OpenAI SDK Reasoning Demo with o1-mini

This example demonstrates using the OpenAI SDK with the Claude Code Proxy API
to showcase reasoning/thinking mode with o1-mini model at high temperature.
"""

import argparse
import json
import logging
import sys
from typing import Optional

import openai
from pydantic import BaseModel


# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


class ReasoningResponse(BaseModel):
    """Structured representation of reasoning response"""

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
        description="OpenAI SDK demo with reasoning mode (o1-mini)"
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
        default="http://localhost:8000/api/v1",
        help="Base URL for the API (default: http://localhost:8000/openai/v1)",
    )
    parser.add_argument(
        "--model", type=str, default="o1-mini", help="Model to use (default: o1-mini)"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Solve this step by step: If a train travels at 60 mph for 2.5 hours, then at 45 mph for 1.5 hours, what is the total distance traveled? Show your reasoning.",
        help="Prompt that requires reasoning",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature for response generation (default: 1.0)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=500,
        help="Maximum tokens in response (default: 500)",
    )
    parser.add_argument(
        "--reasoning-effort",
        type=str,
        default="high",
        choices=["low", "medium", "high"],
        help="Reasoning effort level (default: high)",
    )
    parser.add_argument("--stream", action="store_true", help="Enable streaming mode")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    return parser.parse_args()


def log_configuration(args):
    """Log the configuration being used"""
    logger.info("=" * 80)
    logger.info("REASONING MODE CONFIGURATION")
    logger.info("=" * 80)
    logger.info(f"Model: {args.model}")
    logger.info(f"Temperature: {args.temperature} (high creativity)")
    logger.info(f"Reasoning Effort: {args.reasoning_effort}")
    logger.info(f"Max Tokens: {args.max_tokens}")
    logger.info(f"Streaming: {args.stream}")
    logger.info(f"API Endpoint: {args.base_url}")
    logger.info("=" * 80)


def log_reasoning_response(response_data: dict):
    """Log the reasoning response with special attention to thinking process"""
    logger.info("\n" + "=" * 80)
    logger.info("REASONING RESPONSE ANALYSIS")
    logger.info("=" * 80)

    # Basic metadata
    logger.info(f"Response ID: {response_data.get('id')}")
    logger.info(f"Model Used: {response_data.get('model')}")
    logger.info(f"Created: {response_data.get('created')}")

    # Extract reasoning content
    for i, choice in enumerate(response_data.get("choices", [])):
        message = choice.get("message", {})
        content = message.get("content", "")

        logger.info(f"\nChoice {i} Analysis:")
        logger.info(f"Finish Reason: {choice.get('finish_reason')}")

        # Check if response contains thinking/reasoning markers
        if (
            "<thinking>" in content
            or "Step" in content
            or "reasoning" in content.lower()
        ):
            logger.info("✓ Reasoning process detected in response")

        # Log the full response
        logger.info("\nFull Response:")
        logger.info("-" * 80)
        print(content)
        logger.info("-" * 80)

    # Usage statistics
    usage = response_data.get("usage", {})
    logger.info("\nToken Usage:")
    logger.info(f"  Prompt Tokens: {usage.get('prompt_tokens')}")
    logger.info(f"  Completion Tokens: {usage.get('completion_tokens')}")
    logger.info(f"  Total Tokens: {usage.get('total_tokens')}")

    # Note about high temperature effect
    logger.info(
        "\nNote: With temperature=1.0, responses will be more creative and varied."
    )
    logger.info("Run multiple times to see different reasoning approaches.")
    logger.info("=" * 80)


def handle_streaming_reasoning(stream, args):
    """Handle streaming response for reasoning mode"""
    logger.info("\n" + "=" * 80)
    logger.info("STREAMING REASONING RESPONSE")
    logger.info("=" * 80)

    full_content = ""
    chunk_count = 0

    for chunk in stream:
        chunk_count += 1

        if args.verbose:
            chunk_dict = chunk.model_dump()
            logger.debug(f"Chunk {chunk_count}: {json.dumps(chunk_dict, indent=2)}")

        # Extract and print content
        for choice in chunk.choices:
            if choice.delta and choice.delta.content:
                content = choice.delta.content
                full_content += content
                print(content, end="", flush=True)

    print()  # New line after streaming
    logger.info("-" * 80)
    logger.info(f"Total chunks received: {chunk_count}")
    logger.info(f"Total content length: {len(full_content)} characters")

    # Analyze reasoning patterns in the response
    if "<thinking>" in full_content or "step" in full_content.lower():
        logger.info("✓ Reasoning patterns detected in streamed response")

    logger.info("=" * 80)


def main(args):
    """Main function to demonstrate reasoning with o1-mini"""
    # Log configuration
    log_configuration(args)

    # Configure OpenAI client
    client = openai.OpenAI(api_key=args.api_key, base_url=args.base_url)

    try:
        # Prepare messages with reasoning-focused prompt
        messages = [{"role": "user", "content": args.prompt}]

        # Additional parameters for reasoning mode
        extra_params = {}
        if args.reasoning_effort == "high":
            # Some models support additional reasoning parameters
            extra_params["reasoning_effort"] = "high"

        logger.info(f"\nPrompt: {args.prompt}")
        logger.info("\nSending reasoning request to API...")

        if args.stream:
            # Streaming mode
            stream = client.chat.completions.create(
                model=args.model,
                messages=messages,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                stream=True,
                **extra_params,
            )
            handle_streaming_reasoning(stream, args)
        else:
            # Non-streaming mode
            response = client.chat.completions.create(
                model=args.model,
                messages=messages,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                **extra_params,
            )

            # Convert to dict for analysis
            response_dict = response.model_dump()

            # Log the reasoning response
            log_reasoning_response(response_dict)

            # Verbose mode: show raw JSON
            if args.verbose:
                logger.info("\nRaw JSON Response:")
                logger.info(json.dumps(response_dict, indent=2))

            # Validate response structure
            try:
                validated = ReasoningResponse(**response_dict)
                logger.info("\n✓ Response structure validated")
            except Exception as e:
                logger.error(f"\n✗ Validation failed: {e}")

    except Exception as e:
        logger.error(f"Error during reasoning request: {e}")
        if args.verbose:
            logger.exception("Full exception details:")

        # Provide helpful hints
        if "o1" in str(e) or "model" in str(e):
            logger.info(
                "\nHint: Make sure the o1-mini model is available in your Claude Code Proxy setup."
            )
            logger.info(
                "You may need to map it to an appropriate Claude model in your configuration."
            )

        sys.exit(1)


if __name__ == "__main__":
    args = parse_arguments()

    # Set logging level based on verbosity
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    main(args)
