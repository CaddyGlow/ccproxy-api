"""Configuration constants for CCProxy."""

# Plugin System Constants
PLUGIN_HEALTH_CHECK_TIMEOUT = 10.0  # seconds
PLUGIN_SUMMARY_CACHE_TTL = 300.0  # 5 minutes
PLUGIN_SUMMARY_CACHE_SIZE = 32  # entries

# Task Scheduler Constants
DEFAULT_TASK_INTERVAL = 3600  # 1 hour in seconds

# URL Constants
CLAUDE_API_BASE_URL = "https://api.anthropic.com"
CODEX_API_BASE_URL = "https://chatgpt.com/backend-api"

# API Endpoints
CLAUDE_MESSAGES_ENDPOINT = "/v1/messages"
CODEX_RESPONSES_ENDPOINT = "/codex/responses"

# Format Conversion Patterns
OPENAI_CHAT_COMPLETIONS_PATH = "/v1/chat/completions"
OPENAI_COMPLETIONS_PATH = "/chat/completions"

# HTTP Client Configuration
HTTP_CLIENT_TIMEOUT = 120.0  # 2 minutes default timeout
HTTP_STREAMING_TIMEOUT = 300.0  # 5 minutes for streaming requests
HTTP_CLIENT_POOL_SIZE = 20  # Max connections per pool
