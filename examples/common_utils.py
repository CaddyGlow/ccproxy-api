import json
import logging
import tempfile
from pathlib import Path
from typing import Any

import httpx
import structlog
from httpx import URL


def setup_logging(verbose: int = 0) -> None:
    """Setup logging configuration.
    Args:
        verbose: Verbosity level (0=WARNING, 1=INFO, 2=DEBUG)
    """
    if verbose >= 2:
        level = logging.DEBUG
    elif verbose >= 1:
        level = logging.INFO
    else:
        level = logging.WARNING

    logging.basicConfig(
        level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    logging.getLogger().setLevel(level)

    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    if verbose >= 2:
        logging.getLogger("httpx").setLevel(logging.DEBUG)
        logging.getLogger("openai").setLevel(logging.DEBUG)
        logging.getLogger("anthropic").setLevel(logging.DEBUG)
    elif verbose >= 1:
        logging.getLogger("httpx").setLevel(logging.INFO)
        logging.getLogger("openai").setLevel(logging.INFO)
        logging.getLogger("anthropic").setLevel(logging.INFO)
    else:
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("openai").setLevel(logging.WARNING)
        logging.getLogger("anthropic").setLevel(logging.WARNING)


logger = structlog.get_logger(__name__)


class LoggingSyncClient(httpx.Client):
    """Custom synchronous HTTP client that logs requests and responses"""

    def request(self, method: str, url: URL | str, **kwargs: Any) -> httpx.Response:
        logger.info("http_request_start")
        logger.info(
            "http_request_details",
            method=method,
            url=str(url),
            headers=kwargs.get("headers", {}),
        )
        if "content" in kwargs:
            try:
                content = kwargs["content"]
                if isinstance(content, bytes):
                    content = content.decode("utf-8")
                logger.info("http_request_body", body=content)
            except Exception as e:
                logger.info("http_request_body_decode_error", error=str(e))

        response = super().request(method, url, **kwargs)

        logger.info(
            "http_response_start",
            status_code=response.status_code,
            headers=dict(response.headers),
        )
        try:
            logger.info("http_response_body", body=response.text)
        except Exception as e:
            logger.info("http_response_body_decode_error", error=str(e))

        return response


class LoggingAsyncClient(httpx.AsyncClient):
    """Custom asynchronous HTTP client that logs requests and responses"""

    async def send(self, request: httpx.Request, **kwargs: Any) -> httpx.Response:
        logger.debug("http_request_start")
        logger.debug(
            "http_request_details",
            method=request.method,
            url=str(request.url),
            headers=dict(request.headers),
        )
        if hasattr(request, "content"):
            try:
                content = request.content
                if isinstance(content, bytes):
                    content = content.decode("utf-8")
                logger.debug("http_request_body", body=content)
            except Exception as e:
                logger.debug("http_request_body_decode_error", error=str(e))

        response = await super().send(request, **kwargs)

        logger.debug(
            "http_response_start",
            status_code=response.status_code,
            headers=dict(response.headers),
        )

        if not response.headers.get("content-type", "").startswith("text/event-stream"):
            try:
                logger.debug("http_response_body", body=response.text)
            except Exception as e:
                logger.debug("http_response_body_decode_error", error=str(e))
        else:
            logger.debug("http_response_body", body="[Streaming response]")

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
    logger.info(
        "distance_calculation_start", lat1=lat1, lon1=lon1, lat2=lat2, lon2=lon2
    )
    import math

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


def generate_json_schema_for_function(func: Any) -> dict[str, Any]:
    """
    Generate JSON schema for a function.
    Args:
        func: Function to generate schema for
    Returns:
        JSON schema dictionary
    """
    import inspect

    sig = inspect.signature(func)
    schema: dict[str, Any] = {"type": "object", "properties": {}, "required": []}

    for param_name, param in sig.parameters.items():
        prop_schema = {"type": "string"}

        if param.annotation is str:
            prop_schema = {"type": "string"}
        elif param.annotation is float:
            prop_schema = {"type": "number"}
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
    """
    Handle tool calls by routing to appropriate functions.
    Args:
        tool_name: Name of the tool to call
        tool_input: Input parameters for the tool
    Returns:
        Tool execution result
    """
    logger.info("tool_call_start", tool_name=tool_name, tool_input=tool_input)

    if tool_name == "get_weather":
        result = get_weather(**tool_input)
    elif tool_name == "calculate_distance":
        result = calculate_distance(**tool_input)
    else:
        result = {"error": f"Unknown tool: {tool_name}"}
        logger.error("unknown_tool_requested", tool_name=tool_name)

    logger.info("tool_call_result", result=result)
    return result
