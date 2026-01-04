# Claude Shared Utilities

Shared metadata consumed by Claude provider plugins.

## Highlights
- Defines default `ModelCard` entries for common Claude model releases
- Declares mapping rules that translate OpenAI-style aliases to Claude IDs

## Configuration
- No standalone plugin configuration; these helpers are imported by
  `claude_api` and `claude_sdk` settings modules.

## Usage
- Imported during Claude plugin configuration to seed model catalogs
- Extend by editing `DEFAULT_CLAUDE_MODEL_CARDS` or mapping rules

## Related Components
- `model_defaults.py`: source of cards and alias mapping helpers
