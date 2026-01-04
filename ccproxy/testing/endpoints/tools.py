"""Function-tool helpers used by the endpoint test harness."""

from __future__ import annotations

import inspect
import math
from typing import Any

import structlog


logger = structlog.get_logger(__name__)


def get_weather(location: str, unit: str = "celsius") -> dict[str, Any]:
    """Get current weather for a location."""
    logger.info("weather_request", location=location, unit=unit)
    result = {
        "location": location,
        "temperature": 22 if unit == "celsius" else 72,
        "unit": unit,
        "condition": "sunny",
        "humidity": 65,
        "wind_speed": 10,
    }
    logger.info("weather_result", result=result)
    return result


def calculate_distance(
    lat1: float, lon1: float, lat2: float, lon2: float
) -> dict[str, Any]:
    """Calculate distance between two geographic coordinates."""
    logger.info(
        "distance_calculation_start", lat1=lat1, lon1=lon1, lat2=lat2, lon2=lon2
    )

    lat1_r = math.radians(lat1)
    lon1_r = math.radians(lon1)
    lat2_r = math.radians(lat2)
    lon2_r = math.radians(lon2)

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
    logger.info("distance_calculation_result", result=result)
    return result


def calculate(expression: str) -> dict[str, Any]:
    """Perform basic arithmetic calculations."""
    logger.info("calculation_request", expression=expression)
    try:
        safe_expression = expression.replace("^", "**")
        result = eval(safe_expression)
        response = {"expression": expression, "result": result, "success": True}
    except Exception as exc:  # noqa: BLE001 - surface evaluation errors verbatim
        response = {"expression": expression, "error": str(exc), "success": False}

    logger.info("calculation_result", response=response)
    return response


def generate_json_schema_for_function(func: Any) -> dict[str, Any]:
    """Generate JSON schema for a function."""
    sig = inspect.signature(func)
    schema: dict[str, Any] = {"type": "object", "properties": {}, "required": []}

    for param_name, param in sig.parameters.items():
        prop_schema = {"type": "string"}

        if param.annotation is str:
            prop_schema = {"type": "string"}
        elif param.annotation is float:
            prop_schema = {"type": "number", "format": "float"}
        elif param.annotation is int:
            prop_schema = {"type": "integer"}

        if func.__doc__:
            lines = func.__doc__.strip().split("\n")
            for line in lines:
                if param_name in line and ":" in line:
                    desc = line.split(":", 1)[1].strip()
                    prop_schema["description"] = desc
                    break

        schema["properties"][param_name] = prop_schema

        if param.default == inspect.Parameter.empty:
            required_list = schema["required"]
            if isinstance(required_list, list):
                required_list.append(param_name)

    return schema


def handle_tool_call(tool_name: str, tool_input: dict[str, Any]) -> dict[str, Any]:
    """Handle tool calls by routing to appropriate functions."""
    logger.info("tool_call_start", tool_name=tool_name, tool_input=tool_input)

    if tool_name == "get_weather":
        result = get_weather(**tool_input)
    elif tool_name == "calculate_distance":
        numeric_args = {
            key: float(value) if isinstance(value, str) else value
            for key, value in tool_input.items()
        }
        result = calculate_distance(**numeric_args)
    elif tool_name == "calculate":
        result = calculate(**tool_input)
    else:
        result = {"error": f"Unknown tool: {tool_name}"}
        logger.error("unknown_tool_requested", tool_name=tool_name)

    logger.info("tool_call_result", result=result)
    return result


def create_openai_tools() -> list[dict[str, Any]]:
    """Create OpenAI-compatible tool definitions with JSON schemas."""
    weather_schema = generate_json_schema_for_function(get_weather)
    distance_schema = generate_json_schema_for_function(calculate_distance)
    calc_schema = generate_json_schema_for_function(calculate)

    return [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather information for a specific location",
                "parameters": weather_schema,
            },
        },
        {
            "type": "function",
            "function": {
                "name": "calculate_distance",
                "description": "Calculate the distance between two geographic coordinates",
                "parameters": distance_schema,
            },
        },
        {
            "type": "function",
            "function": {
                "name": "calculate",
                "description": "Perform basic arithmetic calculations",
                "parameters": calc_schema,
            },
        },
    ]


def create_anthropic_tools() -> list[dict[str, Any]]:
    """Create Anthropic-compatible tool definitions with JSON schemas."""
    weather_schema = generate_json_schema_for_function(get_weather)
    distance_schema = generate_json_schema_for_function(calculate_distance)
    calc_schema = generate_json_schema_for_function(calculate)

    return [
        {
            "type": "custom",
            "name": "get_weather",
            "description": "Get current weather information for a specific location",
            "input_schema": weather_schema,
        },
        {
            "type": "custom",
            "name": "calculate_distance",
            "description": "Calculate the distance between two geographic coordinates",
            "input_schema": distance_schema,
        },
        {
            "type": "custom",
            "name": "calculate",
            "description": "Perform basic arithmetic calculations",
            "input_schema": calc_schema,
        },
    ]


def create_codex_tools() -> list[dict[str, Any]]:
    """Create Codex-compatible tool definitions."""
    weather_schema = generate_json_schema_for_function(get_weather)
    distance_schema = generate_json_schema_for_function(calculate_distance)
    calc_schema = generate_json_schema_for_function(calculate)

    return [
        {
            "type": "function",
            "name": "get_weather",
            "description": "Get current weather information for a specific location",
            "parameters": weather_schema,
        },
        {
            "type": "function",
            "name": "calculate_distance",
            "description": "Calculate the distance between two geographic coordinates",
            "parameters": distance_schema,
        },
        {
            "type": "function",
            "name": "calculate",
            "description": "Perform basic arithmetic calculations",
            "parameters": calc_schema,
        },
    ]


_WEATHER_SCHEMA = generate_json_schema_for_function(get_weather)
_DISTANCE_SCHEMA = generate_json_schema_for_function(calculate_distance)
_CALC_SCHEMA = generate_json_schema_for_function(calculate)


OPENAI_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather information for a specific location",
            "parameters": _WEATHER_SCHEMA,
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_distance",
            "description": "Calculate the distance between two geographic coordinates",
            "parameters": _DISTANCE_SCHEMA,
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Perform basic arithmetic calculations",
            "parameters": _CALC_SCHEMA,
        },
    },
]


ANTHROPIC_TOOLS = [
    {
        "type": "custom",
        "name": "get_weather",
        "description": "Get current weather information for a specific location",
        "input_schema": _WEATHER_SCHEMA,
    },
    {
        "type": "custom",
        "name": "calculate_distance",
        "description": "Calculate the distance between two geographic coordinates",
        "input_schema": _DISTANCE_SCHEMA,
    },
    {
        "type": "custom",
        "name": "calculate",
        "description": "Perform basic arithmetic calculations",
        "input_schema": _CALC_SCHEMA,
    },
]


CODEX_TOOLS = [
    {
        "type": "function",
        "name": "get_weather",
        "description": "Get current weather information for a specific location",
        "parameters": _WEATHER_SCHEMA,
    },
    {
        "type": "function",
        "name": "calculate_distance",
        "description": "Calculate the distance between two geographic coordinates",
        "parameters": _DISTANCE_SCHEMA,
    },
    {
        "type": "function",
        "name": "calculate",
        "description": "Perform basic arithmetic calculations",
        "parameters": _CALC_SCHEMA,
    },
]


__all__ = [
    "handle_tool_call",
    "create_openai_tools",
    "create_anthropic_tools",
    "create_codex_tools",
    "OPENAI_TOOLS",
    "ANTHROPIC_TOOLS",
    "CODEX_TOOLS",
]
