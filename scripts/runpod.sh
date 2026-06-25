#!/usr/bin/env bash
set -euo pipefail

RUNPOD_API_URL="${RUNPOD_API_URL:-https://rest.runpod.io/v1}"
RUNPOD_GRAPHQL_URL="${RUNPOD_GRAPHQL_URL:-https://api.runpod.io/graphql}"

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
  scripts/runpod.sh graphql <query-file>
  scripts/runpod.sh gpu-types
  scripts/runpod.sh gpu-candidates [SECURE|COMMUNITY] [max-price]

Optional:
  RUNPOD_ENV_FILE      Optional env file to load.
  RUNPOD_API_URL       Override the Runpod REST API URL.
  RUNPOD_GRAPHQL_URL   Override the Runpod GraphQL API URL.
  RUNPOD_OUTPUT        Set to json to print full JSON responses.

Required:
  RUNPOD_API_KEY       Runpod API key.

For pod-create:
  RUNPOD_REGISTRY_AUTH_ID
                       Required unless provided as a pod-create argument.

Notes:
  Set RUNPOD_ENV_FILE when you want the script to load variables from a file.
  Create the private GHCR registry auth manually in Runpod, then put its ID in
  RUNPOD_REGISTRY_AUTH_ID or pass it to pod-create.
  pod-create injects the container registry auth ID and strips _comment fields
  from the JSON spec before sending it to Runpod.
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

pretty_json() {
  local tmp
  tmp="$(mktemp)"
  cat > "$tmp"

  if [[ -s "$tmp" ]] && jq . "$tmp" >/dev/null 2>&1; then
    jq . "$tmp"
  else
    cat "$tmp"
  fi

  rm -f "$tmp"
}

output_json() {
  [[ "${RUNPOD_OUTPUT:-}" == "json" ]]
}

pod_summary() {
  if output_json; then
    pretty_json
    return
  fi

  jq -r '
    def text($value): if $value == null or $value == "" then "-" else ($value | tostring) end;
    def ssh_port: (.portMappings["22"] // .portMappings[22] // "-");
    [
      ["id", (.id // "-")],
      ["name", (.name // "-")],
      ["status", (.desiredStatus // .runtimeStatus // "-")],
      ["gpu", (.gpu.displayName // .machine.gpuDisplayName // .machine.gpuType.displayName // "-")],
      ["vcpu", text(.vcpuCount)],
      ["ram", (if .memoryInGb then ((.memoryInGb | tostring) + "GB") else "-" end)],
      ["cost/hr", text(.costPerHr // .machine.costPerHr // .machine.currentPricePerGpu)],
      ["public ip", text(.publicIp)],
      ["ssh port", (ssh_port | tostring)]
    ]
    | .[]
    | @tsv
  ' | column -t -s $'\t'
}

pods_table() {
  if output_json; then
    pretty_json
    return
  fi

  jq -r '
    def text($value): if $value == null or $value == "" then "-" else ($value | tostring) end;
    def pods:
      if type == "array" then .
      elif .pods then .pods
      elif .data then .data
      else []
      end;
    def ssh_port: (.portMappings["22"] // .portMappings[22] // "-");
    (
      ["ID", "NAME", "STATUS", "VCPU", "RAM", "COST/HR", "IP", "SSH"],
      (pods[] | [
        text(.id),
        text(.name),
        text(.desiredStatus // .runtimeStatus),
        text(.vcpuCount),
        (if .memoryInGb then ((.memoryInGb | tostring) + "GB") else "-" end),
        text(.costPerHr // .machine.costPerHr // .machine.currentPricePerGpu),
        text(.publicIp),
        (ssh_port | tostring)
      ])
    )
    | @tsv
  ' | column -t -s $'\t'
}

graphql() {
  local query="$1"

  require_env RUNPOD_API_KEY
  jq -n --arg query "$query" '{query: $query}' \
    | curl --fail-with-body --silent --show-error \
      --request POST \
      --url "$RUNPOD_GRAPHQL_URL" \
      --header "Authorization: Bearer $RUNPOD_API_KEY" \
      --header "Content-Type: application/json" \
      --data-binary @-
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
    jq \
      --arg id "$registry_auth_id" \
      '
        .containerRegistryAuthId = $id
        | del(._comment)
      ' "$spec" > "$tmp"
    json_api POST pods "$tmp" | pod_summary
    ;;
  pod-list)
    api GET pods | pods_table
    ;;
  pod-start)
    id="${2:?pod id is required}"
    api POST "pods/$id/start" | pod_summary
    ;;
  pod-stop)
    id="${2:?pod id is required}"
    api POST "pods/$id/stop" | pod_summary
    ;;
  pod-delete)
    id="${2:?pod id is required}"
    api DELETE "pods/$id" | pod_summary
    ;;
  graphql)
    query_file="${2:?GraphQL query file is required}"
    if [[ ! -f "$query_file" ]]; then
      echo "GraphQL query file does not exist: $query_file" >&2
      exit 2
    fi

    graphql "$(cat "$query_file")" | pretty_json
    ;;
  gpu-types)
    graphql '
      query GpuTypes {
        gpuTypes {
          id
          displayName
          memoryInGb
          secureCloud
          communityCloud
          securePrice
          communityPrice
          secureSpotPrice
          communitySpotPrice
        }
      }
    ' | if output_json; then
      jq 'if .errors then . else .data.gpuTypes end'
    else
      jq -r '
        if type == "object" and has("errors") then
          ["ERROR", (.errors | map(.message) | join("; "))]
        else
          (
            ["ID", "NAME", "MEM", "SECURE", "SECURE$/HR", "COMMUNITY", "COMMUNITY$/HR"],
            (.data.gpuTypes[]
              | [
                  .id,
                  .displayName,
                  ((.memoryInGb | tostring) + "GB"),
                  (.secureCloud | tostring),
                  ((.securePrice // "-") | tostring),
                  (.communityCloud | tostring),
                  ((.communityPrice // "-") | tostring)
                ])
          )
        end
        | @tsv
      ' | column -t -s $'\t'
    fi
    ;;
  gpu-candidates)
    cloud="${2:-SECURE}"
    max_price="${3:-0.50}"
    case "$cloud" in
      SECURE|COMMUNITY) ;;
      *)
        echo "cloud must be SECURE or COMMUNITY" >&2
        exit 2
        ;;
    esac

    graphql '
      query GpuTypes {
        gpuTypes {
          id
          displayName
          memoryInGb
          secureCloud
          communityCloud
          securePrice
          communityPrice
        }
      }
    ' | jq --arg cloud "$cloud" --argjson max_price "$max_price" '
      if .errors then
        {errors}
      else
        .data.gpuTypes
        | map(
            if $cloud == "SECURE" then
              select(.secureCloud and .securePrice > 0 and .securePrice <= $max_price)
              | . + {price: .securePrice}
            else
              select(.communityCloud and .communityPrice > 0 and .communityPrice <= $max_price)
              | . + {price: .communityPrice}
            end
          )
        | sort_by(.price, -.memoryInGb, .id)
        | map({
            id,
            displayName,
            memoryInGb,
            price
          })
      end
    ' | if output_json; then
      cat
    else
      jq -r '
        if type == "object" and has("errors") then
          ["ERROR", (.errors | map(.message) | join("; "))]
        else
          (
          ["ID", "NAME", "MEM", "PRICE/HR"],
          (.[] | [
            .id,
            .displayName,
            ((.memoryInGb | tostring) + "GB"),
            (.price | tostring)
          ])
          )
        end
        | @tsv
      ' | column -t -s $'\t'
    fi
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
