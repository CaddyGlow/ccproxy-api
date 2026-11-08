# MiniMax provider plugin

First-class MiniMax provider for CCProxy. It exposes OpenAI- and
Anthropic-compatible endpoints backed by MiniMax's global (`minimax.io`) and
regional (`minimaxi.com`) deployments, using static API-key (Bearer)
authentication.

## Endpoints

Mounted under the `/minimax` prefix:

- `POST /minimax/v1/chat/completions` — OpenAI-compatible chat completions.
- `POST /minimax/v1/messages` — native Anthropic-compatible messages for
  `MiniMax-M3.1-Flash` and `MiniMax-M3`. Other explicitly requested models retain
  the existing conversion to the OpenAI chat protocol. An overridden OpenAI
  `base_url` also retains conversion unless a custom `anthropic_base_url` is set.
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
documentation root unless they are overridden explicitly. The default catalog
lists `MiniMax-M3.1-Flash` first, followed by `MiniMax-M3`. The `minimax` alias
resolves to Flash. Explicit legacy/custom model IDs and configured mappings or
model cards are preserved; removing a legacy model from the default catalog does
not rewrite requests for that model. Requests must still specify a model.

Flash always uses adaptive thinking. Omit thinking/effort to use upstream defaults,
or set `reasoning_effort` for Chat Completions and `output_config.effort` for
Messages to `low`, `medium`, `high`, `xhigh`, or `max` (default). Flash does not
support disabled thinking, `none` effort, `service_tier`, or `reasoning_split=false`.
The proxy does not inject these options or silently change explicit settings;
upstream validation applies. M3 supports adaptive/disabled thinking; its default
is adaptive for Chat Completions and disabled for native Messages.

Native Messages preserves the entire assistant response, including thinking and
tool blocks, when clients send it back in later turns. Image/video blocks and
custom native fields are forwarded unchanged. See the
[MiniMax Messages contract](https://platform.minimax.io/docs/api-reference/text-anthropic-api)
and [pricing](https://platform.minimax.io/docs/pricing/overview). Region selection
configures transport URLs; account/model availability is determined by MiniMax.
