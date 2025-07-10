"""Core constants used across the Claude Code Proxy API."""

# HTTP headers
REQUEST_ID_HEADER = "X-Request-ID"
AUTH_HEADER = "Authorization"
CONTENT_TYPE_HEADER = "Content-Type"

# API endpoints
ANTHROPIC_API_BASE_PATH = "/v1"
OPENAI_API_BASE_PATH = "/openai/v1"
CHAT_COMPLETIONS_ENDPOINT = "/chat/completions"
MESSAGES_ENDPOINT = "/messages"
MODELS_ENDPOINT = "/models"

# Default values
DEFAULT_MODEL = "claude-3-5-sonnet-20241022"
DEFAULT_MAX_TOKENS = 4096
DEFAULT_TEMPERATURE = 1.0
DEFAULT_TOP_P = 1.0
DEFAULT_STREAM = False

# Timeouts (in seconds)
DEFAULT_TIMEOUT = 30
DEFAULT_CONNECT_TIMEOUT = 10
DEFAULT_READ_TIMEOUT = 300

# Rate limiting
DEFAULT_RATE_LIMIT = 100  # requests per minute
DEFAULT_BURST_LIMIT = 10  # burst capacity

# Docker defaults
DEFAULT_DOCKER_IMAGE = "ghcr.io/anthropics/claude-cli:latest"
DEFAULT_DOCKER_TIMEOUT = 300

# File extensions
TOML_EXTENSIONS = [".toml"]
JSON_EXTENSIONS = [".json"]
YAML_EXTENSIONS = [".yaml", ".yml"]

# Configuration file names
CONFIG_FILE_NAMES = [
    ".ccproxy.toml",
    "ccproxy.toml",
    "config.toml",
    "config.json",
    "config.yaml",
    "config.yml",
]

# Environment variable prefixes
ENV_PREFIX = "CCPROXY_"
CLAUDE_ENV_PREFIX = "CLAUDE_"

# Logging levels
LOG_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

# Error messages
ERROR_MSG_INVALID_TOKEN = "Invalid or expired authentication token"
ERROR_MSG_MODEL_NOT_FOUND = "Model not found or not available"
ERROR_MSG_RATE_LIMIT_EXCEEDED = "Rate limit exceeded"
ERROR_MSG_INVALID_REQUEST = "Invalid request format"
ERROR_MSG_INTERNAL_ERROR = "Internal server error"

# Status codes
STATUS_OK = 200
STATUS_CREATED = 201
STATUS_BAD_REQUEST = 400
STATUS_UNAUTHORIZED = 401
STATUS_FORBIDDEN = 403
STATUS_NOT_FOUND = 404
STATUS_RATE_LIMITED = 429
STATUS_INTERNAL_ERROR = 500
STATUS_BAD_GATEWAY = 502
STATUS_SERVICE_UNAVAILABLE = 503

# Stream event types
STREAM_EVENT_MESSAGE_START = "message_start"
STREAM_EVENT_MESSAGE_DELTA = "message_delta"
STREAM_EVENT_MESSAGE_STOP = "message_stop"
STREAM_EVENT_CONTENT_BLOCK_START = "content_block_start"
STREAM_EVENT_CONTENT_BLOCK_DELTA = "content_block_delta"
STREAM_EVENT_CONTENT_BLOCK_STOP = "content_block_stop"

# Content types
CONTENT_TYPE_JSON = "application/json"
CONTENT_TYPE_STREAM = "text/event-stream"
CONTENT_TYPE_TEXT = "text/plain"

# Character limits
MAX_PROMPT_LENGTH = 200_000  # Maximum prompt length in characters
MAX_MESSAGE_LENGTH = 100_000  # Maximum message length
MAX_TOOL_CALLS = 100  # Maximum number of tool calls per request

# Validation patterns
EMAIL_PATTERN = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
URL_PATTERN = r"^https?://[^\s/$.?#].[^\s]*$"
UUID_PATTERN = r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
