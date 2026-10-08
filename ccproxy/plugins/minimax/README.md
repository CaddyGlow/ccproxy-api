# MiniMax provider plugin

First-class MiniMax provider for CCProxy. It exposes OpenAI- and
Anthropic-compatible endpoints backed by MiniMax's global (`minimax.io`) and
regional (`minimaxi.com`) deployments, using static API-key (Bearer)
authentication.

## Endpoints

Mounted under the `/minimax` prefix:

- `POST /minimax/v1/chat/completions` — OpenAI-compatible chat completions.
- `POST /minimax/v1/messages` — Anthropic-compatible messages, converted to the
  OpenAI chat protocol via the core format chain.
- `GET /minimax/v1/models` — configured model metadata.

## Configuration

```toml
[plugins.minimax]
enabled = true
# "global_en" (api.minimax.io) or "cn_zh" (api.minimaxi.com)
region = "global_en"
api_key = "<your-minimax-api-key>"
```

Selecting a `region` resolves the OpenAI and Anthropic base URLs and the
documentation root unless they are overridden explicitly. The default models
are `MiniMax-M3` and `MiniMax-M2.7`.
