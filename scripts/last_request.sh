#!/usr/bin/env bash
# Use a function - it won't visually expand
fdcat() {
  fd -t f "$@" -x sh -c 'printf "\n\033[1;34m=== %s ===\033[0m\n" "$1" && cat "$1"' _ {}
}
PATH_LOG="/tmp/ccproxy"
PATH_REQ="${PATH_LOG}/tracer/"
COMMAND_REQ="${PATH_LOG}/command_replay"

# Parse arguments
N=-1 # Default to last request
REQUEST_ID=""
if [[ $# -gt 0 ]]; then
  if [[ $1 =~ ^-[0-9]+$ ]]; then
    N=$1
  elif [[ $1 =~ ^[a-f0-9]{8}$ ]] || [[ $1 =~ ^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$ ]]; then
    REQUEST_ID=$1
  else
    echo "Usage: $0 [-N|request_id]"
    echo "  -N: Show the Nth-to-last request (e.g., -1 for last, -2 for second-to-last)"
    echo "  request_id: Show the request with the given 8-char hex ID or full UUID"
    exit 1
  fi
fi

if [[ -n "$REQUEST_ID" ]]; then
  LAST_UUID="$REQUEST_ID"
else
  # Get the Nth-to-last ID (grouped by unique ID, preserving chronological order)
  # Handle both 8-char hex IDs and full UUIDs
  # Extract IDs from filenames, prioritizing the file modification order
  ALL_IDS=$(eza -la --sort=modified "${PATH_REQ}" | sed -n -E '
    s/^.*[[:space:]]([a-f0-9]{8})_[0-9]{8}_[0-9]{6}_[0-9]{6}_[0-9]{6}_.*\..*$/\1/p
    s/^.*[[:space:]]([a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12})_[0-9]{8}_[0-9]{6}_[0-9]{6}_[0-9]{6}_.*\..*$/\1/p
  ')
  UNIQUE_IDS=$(echo "$ALL_IDS" | awk '{if(!seen[$0]++) print}')

  if [[ $N == -1 ]]; then
    LAST_UUID=$(echo "$UNIQUE_IDS" | tail -1)
  else
    # Convert negative index to positive from end: -2 becomes 2nd from end, -3 becomes 3rd from end
    POS_FROM_END=$((${N#-}))
    LAST_UUID=$(echo "$UNIQUE_IDS" | tail -n "$POS_FROM_END" | head -1)
  fi
fi

if [[ -z "$LAST_UUID" ]]; then
  if [[ -n "$REQUEST_ID" ]]; then
    echo "No request found for ID $REQUEST_ID"
  else
    echo "No request found for position $N"
  fi
  exit 1
fi

printf "\n\033[1;34m=== Log ===\033[0m\n"
rg -I -t log "${LAST_UUID}" ${PATH_LOG} | jq .
printf "\n\033[1;34m=== Raw ===\033[0m\n"
bat --paging never "${PATH_REQ}/"*"${LAST_UUID}"*.http
printf "\n\033[1;34m=== Requests ===\033[0m\n"
bat --paging never "${PATH_REQ}/"*"${LAST_UUID}"*.json
printf "\n\033[1;34m=== Command ===\033[0m\n"
fd ${LAST_UUID} "${COMMAND_REQ}" | xargs -I{} -- echo {}
# bat --paging never "${COMMAND_REQ}/"*"${LAST_UUID}"*.txt
