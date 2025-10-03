# Max Tokens Plugin

Normalizes `max_tokens` fields so provider requests respect model limits.

## Highlights
- Injects or corrects `max_tokens` / `max_output_tokens` before sending requests
- Supports enforce mode, provider filtering, and alias-aware model lookups
- Pulls limits from pricing cache with optional overrides via local JSON files

## Configuration
- `MaxTokensConfig` toggles enablement, enforce mode, fallback values, and targets
- Environment variables follow the `MAX_TOKENS__*` pattern for quick overrides
- Generate defaults with `python3 scripts/generate_config_from_model.py \
  --format toml --plugin max_tokens --config-class MaxTokensConfig`

```toml
[plugins.max_tokens]
# enabled = true
# default_token_limits_file = "ccproxy/plugins/max_tokens/token_limits.json"
# fallback_max_tokens = 4096
# apply_to_all_providers = true
# target_providers = ["claude_api", "claude_sdk", "codex", "copilot"]
# require_pricing_data = false
# log_modifications = true
# enforce_mode = false
# prioritize_local_file = false

[plugins.max_tokens.modification_reasons]
# missing = "max_tokens was missing from request"
# invalid = "max_tokens was invalid or too high"
# exceeded = "max_tokens exceeded model limit"
# enforced = "max_tokens enforced to model limit (enforce mode)"
```

## Related Components
- `plugin.py`: runtime lifecycle and hook registration
- `adapter.py`: hook implementation that edits outbound payloads
- `service.py`: token limit lookup and caching helpers
