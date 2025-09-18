"""Core constants for format identifiers and related shared values."""

# Format identifiers
FORMAT_OPENAI_CHAT = "openai.chat_completions"
FORMAT_OPENAI_RESPONSES = "openai.responses"
FORMAT_ANTHROPIC_MESSAGES = "anthropic.messages"

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
# Common upstream endpoint paths (provider APIs)
UPSTREAM_ENDPOINT_OPENAI_RESPONSES = "/responses"
UPSTREAM_ENDPOINT_OPENAI_CHAT_COMPLETIONS = "/chat/completions"
UPSTREAM_ENDPOINT_ANTHROPIC_MESSAGES = "/v1/messages"
# Additional common OpenAI-style endpoints
UPSTREAM_ENDPOINT_OPENAI_EMBEDDINGS = "/embeddings"
UPSTREAM_ENDPOINT_OPENAI_MODELS = "/models"
# GitHub Copilot internal API endpoints
UPSTREAM_ENDPOINT_COPILOT_INTERNAL_USER = "/copilot_internal/user"
UPSTREAM_ENDPOINT_COPILOT_INTERNAL_TOKEN = "/copilot_internal/v2/token"
