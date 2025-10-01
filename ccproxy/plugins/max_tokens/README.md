# Max Tokens Plugin

Automatically sets `max_tokens` in requests based on model limits when the value is missing, invalid, or exceeds the model's maximum output tokens.

## Features

- **Automatic max_tokens injection**: Adds `max_tokens` when missing from requests
- **Validation**: Fixes invalid values (negative, zero, non-integer)
- **Limit enforcement**: Prevents exceeding model-specific token limits
- **Alias-aware adjustments**: When model aliases map to provider identifiers,
  the hook realigns `max_output_tokens` to the mapped model's limit
- **Pattern matching**: Supports wildcard patterns for model variants
- **Provider filtering**: Can target specific providers

## Installation

The plugin is registered in `pyproject.toml` and wires itself through the CCProxy
hook system. When the plugin is enabled, it registers a `MaxTokensHook` handler
for the `provider.request.prepared` (the new **PREPARE_REQUEST** stage) hook,
allowing the request payload to be modified immediately before it is dispatched
upstream.

If you're integrating the service outside of the plugin runtime, register the
hook manually:

```python
from ccproxy.core.plugins.hooks import HookRegistry, HookEvent
from ccproxy.plugins.max_tokens.adapter import MaxTokensHook
from ccproxy.plugins.max_tokens.config import MaxTokensConfig
from ccproxy.plugins.max_tokens.service import TokenLimitsService

config = MaxTokensConfig()
service = TokenLimitsService(config)
service.initialize()

hook = MaxTokensHook(config, service)
registry = HookRegistry()
registry.register(hook)
# The hook will now fire for HookEvent.PROVIDER_REQUEST_PREPARED events
```

## Configuration

Configure via environment variables:

```bash
# Disable the plugin
MAX_TOKENS__ENABLED=false

# Change fallback value for unknown models
MAX_TOKENS__FALLBACK_MAX_TOKENS=2048

# Target specific providers only
MAX_TOKENS__APPLY_TO_ALL_PROVIDERS=false
MAX_TOKENS__TARGET_PROVIDERS='["claude_api", "claude_sdk"]'

# Disable modification logging
MAX_TOKENS__LOG_MODIFICATIONS=false
```

## Model Token Limits

The plugin includes token limits for:

### Claude Models
- **Claude 3.5 Sonnet** (`claude-3-5-sonnet-*`): 64,000 tokens
- **Claude 3.5 Haiku** (`claude-3-5-haiku-*`): 8,192 tokens
- **Claude 4 Sonnet** (`claude-4-sonnet-*`): 64,000 tokens
- **Claude 4 Haiku** (`claude-4-haiku-*`): 8,192 tokens
- **Claude 4 Opus** (`claude-4-opus-*`): 32,000 tokens
- **Claude 3 Opus** (`claude-3-opus-*`): 4,096 tokens
- **Claude 3 Sonnet** (`claude-3-sonnet-*`): 4,096 tokens
- **Claude 3 Haiku** (`claude-3-haiku-*`): 4,096 tokens

### OpenAI Models
- **GPT-4o Mini**: 16,384 tokens
- **GPT-4o**: 4,096 tokens
- **GPT-4 Turbo**: 4,096 tokens
- **GPT-4**: 8,192 tokens

### Wildcard Patterns
- `*sonnet*`: 64,000 tokens
- `*haiku*`: 8,192 tokens
- `*opus*`: 32,000 tokens

## How It Works

1. **Hook Invocation**: The `MaxTokensHook` listens for
   `HookEvent.PROVIDER_REQUEST_PREPARED` (the PREPARE_REQUEST stage for
   providers) within the HTTP adapter.
2. **Payload Extraction**: The hook inspects the JSON request body to locate the
   `model` along with any `max_tokens` and `max_output_tokens` values.
3. **Validation**: It determines whether tokens are missing, invalid, mapped to
   an alias with a different provider limit, or exceed the known limit.
4. **Modification**: When needed, it injects or corrects `max_tokens` and/or
   `max_output_tokens`, updating the raw payload bytes before the upstream
   request is sent.
5. **Logging**: Modifications are logged at INFO level via
   `max_tokens_modified` entries.

## Example

**Original Request:**
```json
{
  "model": "claude-3-5-sonnet-20241022",
  "messages": [{"role": "user", "content": "Hello"}]
}
```

**Modified Request** (automatic):
```json
{
  "model": "claude-3-5-sonnet-20241022",
  "messages": [{"role": "user", "content": "Hello"}],
  "max_tokens": 64000
}
```

## Limitations

- **JSON bodies**: Only requests that encode a JSON object with a `model` field
  can be modified.
- **Provider filtering**: Ensure the plugin is configured to target the provider
  that raised the hook; otherwise no change is applied.
- **Static limits**: Token limits are derived from cached pricing data or the
  fallback value (extend via custom JSON if required).

## Custom Token Limits

To add custom token limits, create a `token_limits.json` file:

```json
{
  "models": {
    "my-custom-model": {
      "max_output_tokens": 10000,
      "max_input_tokens": 50000
    }
  }
}
```

Then configure the path:

```bash
MAX_TOKENS__DEFAULT_TOKEN_LIMITS_FILE=/path/to/token_limits.json
```

## Testing

The plugin includes comprehensive tests:

```bash
# Run all tests
uv run pytest tests/plugins/max_tokens/ -v

# Run specific test suites
uv run pytest tests/plugins/max_tokens/test_service.py -v
uv run pytest tests/plugins/max_tokens/test_config.py -v
uv run pytest tests/plugins/max_tokens/test_plugin.py -v
```

## Troubleshooting

### Plugin not modifying requests

1. **Check logs**: Look for `max_tokens_modified` entries; if absent, the hook
   may not be firing.
2. **Verify provider**: Confirm the provider name is included in
   `target_providers` or `apply_to_all_providers` is true.
3. **Hook registry**: Ensure the hook registry is available—warnings about
   `max_tokens_hook_registry_unavailable` indicate registration failed.
4. **Payload shape**: Confirm the upstream request body is JSON and contains a
   `model` field.
