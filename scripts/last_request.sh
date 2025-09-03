#!/usr/bin/env bash
# Use a function - it won't visually expand
fdcat() {
  fd -t f "$@" -x sh -c 'printf "\n\033[1;34m=== %s ===\033[0m\n" "$1" && cat "$1"' _ {}
}
PATH_LOG="/tmp/ccproxy"
PATH_REQ="${PATH_LOG}/tracer/"

# Parse arguments
N=-1 # Default to last request
REQUEST_ID=""
if [[ $# -gt 0 ]]; then
  if [[ $1 =~ ^-[0-9]+$ ]]; then
    N=$1
  elif [[ $1 =~ ^[a-f0-9-]{36}$ ]]; then
    REQUEST_ID=$1
  else
    echo "Usage: $0 [-N|request_id]"
    echo "  -N: Show the Nth-to-last request (e.g., -1 for last, -2 for second-to-last)"
    echo "  request_id: Show the request with the given UUID"
    exit 1
  fi
fi

if [[ -n "$REQUEST_ID" ]]; then
  LAST_UUID="$REQUEST_ID"
else
  # Get the Nth-to-last UUID (grouped by unique UUID, preserving chronological order)
  ALL_UUIDS=$(eza -la --sort=modified "${PATH_REQ}" | grep -E '[a-f0-9-]{36}' | sed -E 's/.*([a-f0-9-]{36})_.*/\1/')
  UNIQUE_UUIDS=$(echo "$ALL_UUIDS" | awk '{if(!seen[$0]++) print}')

  if [[ $N == -1 ]]; then
    LAST_UUID=$(echo "$UNIQUE_UUIDS" | tail -1)
  else
    # Convert negative index to positive from end: -2 becomes 2nd from end, -3 becomes 3rd from end
    POS_FROM_END=$((${N#-}))
    LAST_UUID=$(echo "$UNIQUE_UUIDS" | tail -n "$POS_FROM_END" | head -1)
  fi
fi

if [[ -z "$LAST_UUID" ]]; then
  if [[ -n "$REQUEST_ID" ]]; then
    echo "No request found for UUID $REQUEST_ID"
  else
    echo "No request found for position $N"
  fi
  exit 1
fi

printf "\n\033[1;34m=== Log ===\033[0m\n"
rg -I -t log "${LAST_UUID}" ${PATH_LOG} | jq .
printf "\n\033[1;34m=== Raw ===\033[0m\n"
bat --paging never "${PATH_REQ}"*"${LAST_UUID}"*.http
printf "\n\033[1;34m=== Requests ===\033[0m\n"
bat --paging never "${PATH_REQ}"*"${LAST_UUID}"*.json
