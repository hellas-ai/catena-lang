#!/usr/bin/env bash
set -euo pipefail

RUNPOD_API_URL="${RUNPOD_API_URL:-https://rest.runpod.io/v1}"

if [[ -n "${RUNPOD_ENV_FILE:-}" ]]; then
  if [[ ! -f "$RUNPOD_ENV_FILE" ]]; then
    echo "RUNPOD_ENV_FILE does not exist: $RUNPOD_ENV_FILE" >&2
    exit 2
  fi

  set -a
  # shellcheck source=/dev/null
  source "$RUNPOD_ENV_FILE"
  set +a
fi

usage() {
  cat <<'EOF'
Usage:
  scripts/runpod.sh pod-create <pod-spec.json> [container-registry-auth-id]
  scripts/runpod.sh pod-list
  scripts/runpod.sh pod-start <pod-id>
  scripts/runpod.sh pod-stop <pod-id>
  scripts/runpod.sh pod-delete <pod-id>

Optional:
  RUNPOD_ENV_FILE      Optional env file to load.

Required:
  RUNPOD_API_KEY       Runpod API key.

For pod-create:
  RUNPOD_REGISTRY_AUTH_ID
                       Required unless provided as a pod-create argument.

Notes:
  Set RUNPOD_ENV_FILE when you want the script to load variables from a file.
  Create the private GHCR registry auth manually in Runpod, then put its ID in
  RUNPOD_REGISTRY_AUTH_ID or pass it to pod-create.
EOF
}

require_env() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    echo "missing required environment variable: $name" >&2
    exit 2
  fi
}

api() {
  local method="$1"
  local path="$2"
  shift 2

  require_env RUNPOD_API_KEY
  curl --fail-with-body --silent --show-error \
    --request "$method" \
    --url "$RUNPOD_API_URL/$path" \
    --header "Authorization: Bearer $RUNPOD_API_KEY" \
    "$@"
}

json_api() {
  local method="$1"
  local path="$2"
  local file="$3"

  api "$method" "$path" \
    --header "Content-Type: application/json" \
    --data-binary "@$file"
}

command="${1:-}"
case "$command" in
  pod-create)
    spec="${2:?pod spec JSON path is required}"
    registry_auth_id="${3:-${RUNPOD_REGISTRY_AUTH_ID:-}}"
    if [[ -z "$registry_auth_id" ]]; then
      cat >&2 <<'EOF'
missing Runpod container registry auth ID

Create the private GHCR registry auth manually in Runpod, then either:
  - set RUNPOD_REGISTRY_AUTH_ID in .env
  - pass it as: scripts/runpod.sh pod-create <pod-spec.json> <registry-auth-id>
EOF
      exit 2
    fi

    tmp="$(mktemp)"
    trap 'rm -f "$tmp"' EXIT
    jq --arg id "$registry_auth_id" '.containerRegistryAuthId = $id' "$spec" > "$tmp"
    json_api POST pods "$tmp"
    ;;
  pod-list)
    api GET pods
    ;;
  pod-start)
    id="${2:?pod id is required}"
    api POST "pods/$id/start"
    ;;
  pod-stop)
    id="${2:?pod id is required}"
    api POST "pods/$id/stop"
    ;;
  pod-delete)
    id="${2:?pod id is required}"
    api DELETE "pods/$id"
    ;;
  -h|--help|help|"")
    usage
    ;;
  *)
    echo "unknown command: $command" >&2
    usage >&2
    exit 2
    ;;
esac
