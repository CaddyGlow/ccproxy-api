#!/usr/bin/env python3
"""
Anthropic SDK Tool Use Demonstration

This script demonstrates how to use tools with the Anthropic SDK,
using check-jsonschema to generate input schemas for exposed functions.
"""

import argparse
import json
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Union

import anthropic
import httpx


def setup_logging(debug: bool = False) -> None:
    """Setup logging configuration.

    Args:
        debug: Whether to enable debug logging
    """
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Set levels for external libraries
    if debug:
        logging.getLogger("httpx").setLevel(logging.DEBUG)
        logging.getLogger("anthropic").setLevel(logging.DEBUG)
    else:
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("anthropic").setLevel(logging.WARNING)


logger = logging.getLogger(__name__)


class LoggingHTTPClient(httpx.Client):
    """Custom HTTP client that logs requests and responses"""

    def request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        logger.info("=== HTTP REQUEST ===")
        logger.info(f"Method: {method}")
        logger.info(f"URL: {url}")
        logger.info(f"Headers: {kwargs.get('headers', {})}")
        if "content" in kwargs:
            try:
                content = kwargs["content"]
                if isinstance(content, bytes):
                    content = content.decode("utf-8")
                logger.info(f"Body: {content}")
            except Exception as e:
                logger.info(f"Body: <could not decode: {e}>")

        response = super().request(method, url, **kwargs)

        logger.info("=== HTTP RESPONSE ===")
        logger.info(f"Status: {response.status_code}")
        logger.info(f"Headers: {dict(response.headers)}")
        try:
            logger.info(f"Body: {response.text}")
        except Exception as e:
            logger.info(f"Body: <could not decode: {e}>")

        return response


def get_weather(location: str, unit: str = "celsius") -> dict[str, Any]:
    """
    Get current weather for a location.

    Args:
        location: The city and state/country to get weather for
        unit: Temperature unit (celsius or fahrenheit)

    Returns:
        Dictionary containing weather information
    """
    logger.info(f"Getting weather for {location} in {unit}")

    # Mock weather data for demonstration
    result = {
        "location": location,
        "temperature": 22 if unit == "celsius" else 72,
        "unit": unit,
        "condition": "sunny",
        "humidity": 65,
        "wind_speed": 10,
    }

    logger.info(f"Weather result: {result}")
    return result


def calculate_distance(
    lat1: float, lon1: float, lat2: float, lon2: float
) -> dict[str, Any]:
    """
    Calculate distance between two geographic coordinates.

    Args:
        lat1: Latitude of first point
        lon1: Longitude of first point
        lat2: Latitude of second point
        lon2: Longitude of second point

    Returns:
        Dictionary containing distance information
    """
    logger.info(f"Calculating distance between ({lat1}, {lon1}) and ({lat2}, {lon2})")

    # Simplified distance calculation for demonstration
    import math

    # Convert to radians
    lat1_r = math.radians(lat1)
    lon1_r = math.radians(lon1)
    lat2_r = math.radians(lat2)
    lon2_r = math.radians(lon2)

    # Haversine formula
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.asin(math.sqrt(a))
    distance_km = 6371 * c

    result = {
        "distance_km": round(distance_km, 2),
        "distance_miles": round(distance_km * 0.621371, 2),
        "coordinates": {
            "start": {"lat": lat1, "lon": lon1},
            "end": {"lat": lat2, "lon": lon2},
        },
    }

    logger.info(f"Distance calculation result: {result}")
    return result


def generate_json_schema_for_function(func: Any) -> dict[str, Any]:
    """
    Generate JSON schema for a function using check-jsonschema.

    Args:
        func: Function to generate schema for

    Returns:
        JSON schema dictionary
    """
    # Create a temporary Python file with the function
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        # Write function definition
        import inspect

        source = inspect.getsource(func)
        f.write(source)
        f.write("\n\n# Example usage for schema generation\n")
        f.write(f"result = {func.__name__}(")

        # Generate example parameters based on annotations
        sig = inspect.signature(func)
        example_params = []
        for param_name, param in sig.parameters.items():
            if param.annotation is str:
                example_params.append(f'{param_name}="example"')
            elif param.annotation is float:
                example_params.append(f"{param_name}=0.0")
            elif param.annotation is int:
                example_params.append(f"{param_name}=0")
            else:
                example_params.append(f"{param_name}=None")

        f.write(", ".join(example_params))
        f.write(")\n")
        f.write("print(json.dumps(result, indent=2))\n")
        temp_file = f.name

    try:
        # Generate schema based on function signature
        sig = inspect.signature(func)
        schema: dict[str, Any] = {"type": "object", "properties": {}, "required": []}

        for param_name, param in sig.parameters.items():
            prop_schema = {"type": "string"}  # default

            if param.annotation is str:
                prop_schema = {"type": "string"}
            elif param.annotation is float:
                prop_schema = {"type": "number"}
            elif param.annotation is int:
                prop_schema = {"type": "integer"}

            # Add description from docstring if available
            if func.__doc__:
                lines = func.__doc__.strip().split("\n")
                for line in lines:
                    if param_name in line and ":" in line:
                        desc = line.split(":", 1)[1].strip()
                        prop_schema["description"] = desc
                        break

            schema["properties"][param_name] = prop_schema

            # Add to required if no default value
            if param.default == inspect.Parameter.empty:
                required_list = schema["required"]
                if isinstance(required_list, list):
                    required_list.append(param_name)

        return schema

    finally:
        # Clean up temp file
        Path(temp_file).unlink(missing_ok=True)


def create_anthropic_tools() -> list[dict[str, Any]]:
    """
    Create Anthropic-compatible tool definitions with JSON schemas.

    Returns:
        List of tool definitions
    """
    tools = []

    # Get weather tool
    weather_schema = generate_json_schema_for_function(get_weather)
    tools.append(
        {
            "name": "get_weather",
            "description": "Get current weather information for a specific location",
            "input_schema": weather_schema,
        }
    )

    # Calculate distance tool
    distance_schema = generate_json_schema_for_function(calculate_distance)
    tools.append(
        {
            "name": "calculate_distance",
            "description": "Calculate the distance between two geographic coordinates",
            "input_schema": distance_schema,
        }
    )

    return tools


def handle_tool_call(tool_name: str, tool_input: dict[str, Any]) -> dict[str, Any]:
    """
    Handle tool calls by routing to appropriate functions.

    Args:
        tool_name: Name of the tool to call
        tool_input: Input parameters for the tool

    Returns:
        Tool execution result
    """
    logger.info(f"Handling tool call: {tool_name} with input: {tool_input}")

    if tool_name == "get_weather":
        result = get_weather(**tool_input)
    elif tool_name == "calculate_distance":
        result = calculate_distance(**tool_input)
    else:
        result = {"error": f"Unknown tool: {tool_name}"}
        logger.error(f"Unknown tool requested: {tool_name}")

    logger.info(f"Tool call result: {result}")
    return result


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Anthropic SDK Tool Use Demonstration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 anthropic_tools_demo.py
  python3 anthropic_tools_demo.py --debug
  python3 anthropic_tools_demo.py -d
        """,
    )
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="Enable debug logging (shows HTTP requests/responses)",
    )
    return parser.parse_args()


def main() -> None:
    """
    Main demonstration function.
    """
    args = parse_args()
    setup_logging(debug=args.debug)

    print("Anthropic SDK Tool Use Demonstration")
    print("=" * 40)
    if args.debug:
        print("Debug logging enabled")
        print("=" * 40)

    # Create tools
    tools = create_anthropic_tools()

    print("\nGenerated Tools:")
    for tool in tools:
        print(f"\n{tool['name']}:")
        print(f"  Description: {tool['description']}")
        print(f"  Schema: {json.dumps(tool['input_schema'], indent=4)}")

    # Initialize Anthropic client with custom HTTP client
    try:
        http_client = LoggingHTTPClient()
        client = anthropic.Anthropic(http_client=http_client)
        logger.info(
            "Anthropic client initialized successfully with logging HTTP client"
        )

        # Example conversation with tools
        messages: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": "What's the weather like in New York, and how far is it from Los Angeles?",
            }
        ]

        print("\n" + "=" * 40)
        print("Starting conversation with Claude...")
        print("=" * 40)

        logger.info(f"Sending request to Claude with {len(tools)} tools")
        logger.info(f"Tools available: {[tool['name'] for tool in tools]}")
        logger.info(f"Initial message: {messages[0]['content']}")

        # Log the complete request structure
        logger.info("=== REQUEST STRUCTURE ===")
        logger.info("Model: claude-3-5-sonnet-20241022")
        logger.info("Max tokens: 1000")
        logger.info(f"Messages: {json.dumps(messages, indent=2)}")
        logger.info(f"Tools: {json.dumps(tools, indent=2)}")

        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1000,
            tools=tools,
            messages=messages,
        )

        print("\nClaude's response:")
        print(f"Stop reason: {response.stop_reason}")

        # Log the complete response structure
        logger.info("=== COMPLETE RESPONSE STRUCTURE ===")
        logger.info(f"Response ID: {response.id}")
        logger.info(f"Model: {response.model}")
        logger.info(f"Stop reason: {response.stop_reason}")
        logger.info(f"Usage: {response.usage}")
        logger.info(f"Content blocks: {len(response.content)}")

        for i, content_block in enumerate(response.content):
            logger.info(f"Content block {i}: type={content_block.type}")
            if hasattr(content_block, "text"):
                logger.info(f"  Text: {content_block.text[:100]}...")
            if hasattr(content_block, "name"):
                logger.info(f"  Tool name: {content_block.name}")
            if hasattr(content_block, "input"):
                logger.info(f"  Tool input: {content_block.input}")

        # Handle the response based on stop reason
        while True:
            print(f"\nStop reason: {response.stop_reason}")

            # Show text content if any
            for content_block in response.content:
                if content_block.type == "text":
                    print(f"Text: {content_block.text}")

            # Handle different stop reasons
            if response.stop_reason == "tool_use":
                print("\nTool calls requested:")
                tool_results = []

                for content_block in response.content:
                    if content_block.type == "tool_use":
                        tool_name = content_block.name
                        tool_input = content_block.input
                        tool_use_id = content_block.id

                        print(f"\nTool: {tool_name}")
                        print(f"Input: {json.dumps(tool_input, indent=2)}")

                        # Execute the tool
                        result = handle_tool_call(tool_name, tool_input)
                        print(f"Result: {json.dumps(result, indent=2)}")

                        tool_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": tool_use_id,
                                "content": json.dumps(result),
                            }
                        )

                # Add assistant message and tool results to conversation
                messages.append({"role": "assistant", "content": response.content})
                messages.append({"role": "user", "content": tool_results})

                logger.info("=== MESSAGE HISTORY AFTER TOOL USE ===")
                for i, msg in enumerate(messages):
                    logger.info(f"Message {i}: role={msg['role']}")
                    if isinstance(msg["content"], str):
                        logger.info(f"  Content: {msg['content'][:100]}...")
                    else:
                        logger.info(f"  Content: {type(msg['content'])}")

                # Continue conversation with tool results
                logger.info("Sending follow-up request with tool results")
                response = client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=1000,
                    tools=tools,
                    messages=messages,
                )

                logger.info("=== FOLLOW-UP RESPONSE ===")
                logger.info(f"Response ID: {response.id}")
                logger.info(f"Stop reason: {response.stop_reason}")
                logger.info(f"Usage: {response.usage}")

            elif response.stop_reason in ["end_turn", "stop_sequence", "max_tokens"]:
                # Conversation is complete
                print(f"\nConversation ended with stop reason: {response.stop_reason}")
                break
            else:
                # Unknown stop reason
                print(f"\nUnknown stop reason: {response.stop_reason}")
                break

    except Exception as e:
        print(f"\nError: {e}")
        print("Make sure you have the ANTHROPIC_API_KEY environment variable set.")


if __name__ == "__main__":
    main()
