#!/usr/bin/env bash
set -euo pipefail

# Convert codex_full_auth.raw to an equivalent curl request.
#
# Usage:
#   AUTH_TOKEN=... ./scripts/codex_full_auth.sh [--body path/to/body.json] [--raw path/to/codex_full_auth.raw]
#
# Notes:
# - Sensitive values (Authorization bearer, session/account IDs) are read from env vars when set,
#   otherwise they are best-effort extracted from the provided .raw file.
# - You can override the target with BASE_URL (default inferred from Host header or http://127.0.0.1:48691).
# - The script streams SSE by using curl -N and Accept: text/event-stream.

RAW_FILE_DEFAULT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)/codex_full_auth.raw"
RAW_FILE="$RAW_FILE_DEFAULT"
BODY_FILE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
  --body)
    BODY_FILE="$2"
    shift 2
    ;;
  --raw)
    RAW_FILE="$2"
    shift 2
    ;;
  -h | --help)
    grep '^#' "$0" | sed 's/^# \{0,1\}//'
    exit 0
    ;;
  *)
    echo "Unknown argument: $1" >&2
    exit 2
    ;;
  esac
done

if [[ ! -f "$RAW_FILE" ]]; then
  echo "Raw file not found: $RAW_FILE" >&2
  echo "Provide it via --raw or place codex_full_auth.raw at repo root." >&2
  exit 1
fi

# Helpers to extract header values from the raw file (case-insensitive key match)
extract_header() {
  local key="$1"
  # Normalize to lowercase, trim leading/trailing spaces
  awk -v IGNORECASE=1 -v key="$key" '
    /^[[:space:]]*$/ {exit} # stop at blank line (end of headers)
    {
      line=$0
      # Split at first ':'
      split(line, a, ":")
      hname=a[1]
      sub(/^ +| +$/, "", hname)
      if (tolower(hname) == tolower(key)) {
        sub(/^[^:]*:/, "", line)
        sub(/^ +/, "", line)
        print line
        exit
      }
    }
  ' "$RAW_FILE"
}

extract_body_to_file() {
  local outfile="$1"
  awk 'BEGIN{body=0} { if(body){print $0} else if ($0 ~ /^\r?$/) { body=1 } }' "$RAW_FILE" >"$outfile"
}

# Determine BASE_URL from env or Host header
HOST_HEADER="$(extract_header host || true)"
if [[ -n "${BASE_URL:-}" ]]; then
  BASE_URL="${BASE_URL%/}"
elif [[ -n "$HOST_HEADER" ]]; then
  BASE_URL="http://$HOST_HEADER"
else
  BASE_URL="http://127.0.0.1:48691"
fi

BASE_URL="https://chatgpt.com"

# Resolve headers from env or raw
AUTH_TOKEN="${AUTH_TOKEN:-}"
# Try ~/.codex/auth.json (or $CODEX_AUTH_JSON) if AUTH_TOKEN not set
if [[ -z "$AUTH_TOKEN" ]]; then
  CODEX_AUTH_JSON_PATH="${CODEX_AUTH_JSON:-$HOME/.codex/auth.json}"
  if [[ -f "$CODEX_AUTH_JSON_PATH" ]]; then
    if command -v jq >/dev/null 2>&1; then
      AUTH_TOKEN="$(jq -r '.tokens.access_token // empty' "$CODEX_AUTH_JSON_PATH")"
    fi
    if [[ -z "$AUTH_TOKEN" ]]; then
      AUTH_TOKEN="$(
        python3 - <<'PY'
import json, os, sys
p = os.environ.get('CODEX_AUTH_JSON', os.path.expanduser('~/.codex/auth.json'))
try:
    with open(p, 'r') as f:
        data = json.load(f)
    tok = data.get('tokens', {}).get('access_token', '')
    if tok:
        print(tok)
except Exception:
    pass
PY
      )"
    fi
  fi
fi
if [[ -z "$AUTH_TOKEN" ]]; then
  AUTH_TOKEN="$(extract_header authorization | sed -E 's/^Bearer +//I' || true)"
fi

VERSION_HEADER="${VERSION_HEADER:-$(extract_header version || echo "0.27.0") }"
OPENAI_BETA_HEADER="${OPENAI_BETA_HEADER:-$(extract_header openai-beta || echo "responses=experimental") }"
SESSION_ID_HEADER="${SESSION_ID_HEADER:-$(extract_header session_id || true)}"
CHATGPT_ACCOUNT_ID_HEADER="${CHATGPT_ACCOUNT_ID_HEADER:-$(extract_header chatgpt-account-id || true)}"
ORIGINATOR_HEADER="${ORIGINATOR_HEADER:-$(extract_header originator || echo "codex_cli_rs") }"
USER_AGENT_HEADER="${USER_AGENT_HEADER:-$(extract_header user-agent || echo "codex_cli_rs/0.27.0") }"

if [[ -z "$AUTH_TOKEN" ]]; then
  echo "Missing AUTH_TOKEN and could not extract from raw file." >&2
  echo "Set AUTH_TOKEN=... in env (without 'Bearer ')." >&2
  exit 1
fi

# Prepare body file
TMP_BODY=""
cleanup() { [[ -n "$TMP_BODY" && -f "$TMP_BODY" ]] && rm -f "$TMP_BODY"; }
trap cleanup EXIT

if [[ -z "$BODY_FILE" ]]; then
  TMP_BODY="$(mktemp)"
  extract_body_to_file "$TMP_BODY"
  BODY_FILE="$TMP_BODY"
fi

if [[ ! -s "$BODY_FILE" ]]; then
  echo "Body file is empty or missing: $BODY_FILE" >&2
  exit 1
fi

URL="$BASE_URL/backend-api/codex/responses"

set -x
curl -v -N -sS \
  -X POST "$URL" \
  -H "Authorization: Bearer $AUTH_TOKEN" \
  -H "version: $VERSION_HEADER" \
  -H "openai-beta: $OPENAI_BETA_HEADER" \
  ${SESSION_ID_HEADER:+-H "session_id: $SESSION_ID_HEADER"} \
  -H "accept: text/event-stream" \
  -H "accept-encoding: identity" \
  -H "content-type: application/json" \
  ${CHATGPT_ACCOUNT_ID_HEADER:+-H "chatgpt-account-id: $CHATGPT_ACCOUNT_ID_HEADER"} \
  -H "originator: $ORIGINATOR_HEADER" \
  -H "user-agent: $USER_AGENT_HEADER" \
  --data-binary @"$BODY_FILE"
set +x
