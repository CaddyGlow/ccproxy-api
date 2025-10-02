# Max Tokens Plugin

Automatically sets `max_tokens` in requests based on model limits when the value is missing, invalid, or exceeds the model's maximum output tokens.

## Features

- **Automatic max_tokens injection**: Adds `max_tokens` when missing from requests
- **Validation**: Fixes invalid values (negative, zero, non-integer)
- **Limit enforcement**: Prevents exceeding model-specific token limits
- **Enforce mode**: Always sets `max_tokens` to the model's maximum limit, ignoring existing values
- **Alias-aware adjustments**: When model aliases map to provider identifiers,
  the hook realigns `max_output_tokens` to the mapped model's limit
- **Pattern matching**: Supports wildcard patterns for model variants
- **Provider filtering**: Can target specific providers

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

# Enable enforce mode (always set max_tokens to model limit)
MAX_TOKENS__ENFORCE_MODE=true

# Prioritize local token_limits.json over pricing cache
MAX_TOKENS__PRIORITIZE_LOCAL_FILE=true
```

or via the config file:

```toml
[plugins.max_tokens]
enabled = true
fallback_max_tokens = 2048
apply_to_all_providers = false
target_providers = ["claude_api", "claude_sdk"]
log_modifications = true
enforce_mode = true
prioritize_local_file = true
default_token_limits_file = "/path/to/token_limits.json"
```
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

## Enforce Mode

When `enforce_mode=true`, the plugin will always set `max_tokens` to the model's maximum limit, regardless of the existing value in the request. This ensures that all requests use the maximum possible output tokens for the selected model.

**Example with enforce mode enabled:**

**Original Request:**
```json
{
  "model": "claude-3-5-sonnet-20241022",
  "max_tokens": 1000,
  "messages": [{"role": "user", "content": "Hello"}]
}
```

**Modified Request** (enforce mode):
```json
{
  "model": "claude-3-5-sonnet-20241022",
  "max_tokens": 8192,  // Set to model's maximum
  "messages": [{"role": "user", "content": "Hello"}]
}
```

**Key behaviors in enforce mode:**
- Always modifies requests to set `max_tokens` to the model's limit
- Ignores existing valid `max_tokens` values
- Still respects provider filtering (only applies to configured providers)
- Uses fallback limits for unknown models when configured
- Logs modifications with "enforced" reason

## Local File Priority

The plugin supports two modes for handling the local `token_limits.json` file:

### Default Mode (Fallback)
- **Pricing cache first**: Uses pricing cache values when available
- **Local file as fallback**: Only uses local file for models not found in pricing cache
- **Recommended**: Ensures you get the most up-to-date pricing data

```toml
[plugins.max_tokens]
prioritize_local_file = false  # Default behavior
```

### Prioritize Mode (Override)
- **Local file first**: Local file values take precedence over pricing cache
- **Pricing cache as fallback**: Only uses pricing cache for models not in local file
- **Use case**: When you want to enforce specific limits regardless of pricing cache

```toml
[plugins.max_tokens]
prioritize_local_file = true  # Local file overrides pricing cache
```

## Limitations

- **JSON bodies**: Only requests that encode a JSON object with a `model` field
  can be modified.
- **Provider filtering**: Ensure the plugin is configured to target the provider
  that raised the hook; otherwise no change is applied.
- **Static limits**: Token limits are derived from cached pricing data of LiteLLM

## Custom Token Limits

To add custom token limits, create a `token_limits.json` file:

```json
{
  "my-custom-model": {
    "max_output_tokens": 10000,
    "max_input_tokens": 50000
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
