#!/usr/bin/env python3
"""
OpenAI SDK Tool Use Demonstration

This script demonstrates how to use tools with the OpenAI SDK (pointing to Claude via proxy),
using check-jsonschema to generate input schemas for exposed functions.
"""

import argparse
import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Union

import httpx
import openai
from httpx import URL
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionToolParam,
    ChatCompletionUserMessageParam,
)


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
        logging.getLogger("openai").setLevel(logging.DEBUG)
    else:
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("openai").setLevel(logging.WARNING)


logger = logging.getLogger(__name__)


class LoggingHTTPClient(httpx.Client):
    """Custom HTTP client that logs requests and responses"""

    def request(self, method: str, url: URL | str, **kwargs: Any) -> httpx.Response:
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


def create_openai_tools() -> list[ChatCompletionToolParam]:
    """
    Create OpenAI-compatible tool definitions with JSON schemas.

    Returns:
        List of tool definitions
    """
    tools: list[ChatCompletionToolParam] = []

    # Get weather tool
    weather_schema = generate_json_schema_for_function(get_weather)
    tools.append(
        ChatCompletionToolParam(
            type="function",
            function={
                "name": "get_weather",
                "description": "Get current weather information for a specific location",
                "parameters": weather_schema,
            },
        )
    )

    # Calculate distance tool
    distance_schema = generate_json_schema_for_function(calculate_distance)
    tools.append(
        ChatCompletionToolParam(
            type="function",
            function={
                "name": "calculate_distance",
                "description": "Calculate the distance between two geographic coordinates",
                "parameters": distance_schema,
            },
        )
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
        description="OpenAI SDK Tool Use Demonstration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 openai_tools_demo.py
  python3 openai_tools_demo.py --debug
  python3 openai_tools_demo.py -d
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

    print("OpenAI SDK Tool Use Demonstration")
    print("=" * 40)
    if args.debug:
        print("Debug logging enabled")
        print("=" * 40)

    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    base_url_default = "http://127.0.0.1:8000"

    if not api_key:
        logger.warning("OPENAI_API_KEY not set, using dummy key")
        os.environ["OPENAI_API_KEY"] = "dummy"
    if not base_url:
        logger.warning(f"OPENAI_BASE_URL not set, using {base_url_default}")
        os.environ["OPENAI_BASE_URL"] = base_url_default

    # Create tools
    tools = create_openai_tools()

    print("\nGenerated Tools:")
    for tool in tools:
        # Use dict access for ChatCompletionToolParam attributes
        tool_dict = tool if isinstance(tool, dict) else tool.model_dump()
        func_def = tool_dict.get("function", {})
        print(f"\n{func_def.get('name', 'Unknown')}:")
        print(f"  Description: {func_def.get('description', 'No description')}")
        print(f"  Schema: {json.dumps(func_def.get('parameters', {}), indent=4)}")

    # Initialize OpenAI client with custom HTTP client
    try:
        http_client = LoggingHTTPClient()
        client = openai.OpenAI(
            http_client=http_client,
        )
        logger.info("OpenAI client initialized successfully with logging HTTP client")

        # Example conversation with tools
        messages: list[ChatCompletionMessageParam] = [
            ChatCompletionUserMessageParam(
                role="user",
                content="What's the weather like in New York, and how far is it from Los Angeles?",
            )
        ]

        print("\n" + "=" * 40)
        print("Starting conversation with Claude via OpenAI API...")
        print("=" * 40)

        logger.info(f"Sending request to Claude with {len(tools)} tools")
        logger.info(
            f"Tools available: {[getattr(tool, 'function', {}).get('name', 'Unknown') if hasattr(tool, 'function') else tool.get('function', {}).get('name', 'Unknown') for tool in tools]}"
        )
        logger.info(f"Initial message: {getattr(messages[0], 'content', 'No content')}")

        # Log the complete request structure
        logger.info("=== REQUEST STRUCTURE ===")
        logger.info("Model: gpt-4o")
        logger.info("Max tokens: 1000")
        logger.info(f"Messages: {json.dumps(messages, indent=2)}")
        logger.info(f"Tools: {json.dumps(tools, indent=2)}")

        response = client.chat.completions.create(
            model="gpt-4o",  # Will be mapped to Claude by proxy
            max_tokens=1000,
            tools=tools,
            messages=messages,
        )

        print("\nClaude's response:")

        # Log the complete response structure
        logger.info("=== COMPLETE RESPONSE STRUCTURE ===")
        logger.info(f"Response: {response}")
        logger.info(f"Response ID: {response.id}")
        logger.info(f"Model: {response.model}")
        logger.info(f"Usage: {response.usage}")
        logger.info(f"Choices: {len(response.choices) if response.choices else 0}")

        if not response.choices:
            print("No choices in response!")
            return

        choice = response.choices[0]
        print(f"Finish reason: {choice.finish_reason}")

        logger.info(f"Choice finish reason: {choice.finish_reason}")
        logger.info(f"Message role: {choice.message.role}")
        logger.info(f"Message content: {choice.message.content}")
        if choice.message.tool_calls:
            logger.info(f"Tool calls: {len(choice.message.tool_calls)}")

        # Handle the response based on finish reason
        while True:
            print(f"\nFinish reason: {choice.finish_reason}")

            # Show text content if any
            if choice.message.content:
                print(f"Text: {choice.message.content}")

            # Handle different finish reasons
            if choice.finish_reason == "tool_calls" and choice.message.tool_calls:
                print("\nTool calls requested:")
                tool_messages = []

                for tool_call in choice.message.tool_calls:
                    tool_name = tool_call.function.name
                    tool_input = json.loads(tool_call.function.arguments)
                    tool_call_id = tool_call.id

                    print(f"\nTool: {tool_name}")
                    print(f"Input: {json.dumps(tool_input, indent=2)}")

                    # Execute the tool
                    result = handle_tool_call(tool_name, tool_input)
                    print(f"Result: {json.dumps(result, indent=2)}")

                    tool_messages.append(
                        ChatCompletionToolMessageParam(
                            role="tool",
                            content=json.dumps(result),
                            tool_call_id=tool_call_id,
                        )
                    )

                # Add assistant message and tool results to conversation
                messages.append(
                    ChatCompletionAssistantMessageParam(
                        role="assistant",
                        content=choice.message.content,
                        tool_calls=[
                            {
                                "id": tc.id,
                                "type": tc.type,
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments,
                                },
                            }
                            for tc in choice.message.tool_calls
                        ],
                    )
                )
                messages.extend(tool_messages)

                logger.info("=== MESSAGE HISTORY AFTER TOOL USE ===")
                for i, msg in enumerate(messages):
                    logger.info(f"Message {i}: role={getattr(msg, 'role', 'Unknown')}")
                    msg_content = getattr(msg, "content", None)
                    if isinstance(msg_content, str):
                        content = msg_content
                        logger.info(
                            f"  Content: {content[:100] + '...' if len(content) > 100 else content}"
                        )
                    else:
                        logger.info(f"  Content: {type(msg_content)}")

                # Continue conversation with tool results
                logger.info("Sending follow-up request with tool results")
                response = client.chat.completions.create(
                    model="gpt-4o",
                    max_tokens=1000,
                    tools=tools,
                    messages=messages,
                )

                choice = response.choices[0]
                logger.info("=== FOLLOW-UP RESPONSE ===")
                logger.info(f"Response ID: {response.id}")
                logger.info(f"Finish reason: {choice.finish_reason}")
                logger.info(f"Usage: {response.usage}")

            elif choice.finish_reason in ["stop", "length"]:
                # Conversation is complete
                print(
                    f"\nConversation ended with finish reason: {choice.finish_reason}"
                )
                break
            else:
                # Unknown finish reason
                print(f"\nUnknown finish reason: {choice.finish_reason}")
                break

    except Exception as e:
        print(f"\nError: {e}")
        print("Make sure your proxy server is running on http://127.0.0.1:8000")


if __name__ == "__main__":
    main()
