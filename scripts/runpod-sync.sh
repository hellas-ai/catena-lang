#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/runpod-sync.sh <host> <ssh-key> [port]

Arguments:
  host      Runpod SSH host. Use user@host when the user is not root.
  ssh-key   SSH private key to use.
  port      SSH port. Defaults to 22.

Example:
  scripts/runpod-sync.sh root@1.2.3.4 ~/.ssh/runpod_ed25519 2222

Equivalent SSH command:
  ssh -i ~/.ssh/runpod_ed25519 -p 2222 root@1.2.3.4
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

host="${1:-}"
key="${2:-}"
port="${3:-22}"

if [[ -z "$host" || -z "$key" ]]; then
  echo "missing required arguments" >&2
  usage >&2
  exit 2
fi

if [[ ! -f "$key" ]]; then
  echo "ssh key does not exist: $key" >&2
  exit 2
fi

# Runpod exposes SSH as a host plus a mapped TCP port. Use the host shown by
# Runpod, the mapped SSH port, and the private key matching the public key
# configured on the pod/template. To open a shell directly:
#   ssh -i "$key" -p "$port" "$host"
rsync -az --delete --info=progress2 \
  -e "ssh -i $key -p $port -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new" \
  --exclude .git \
  --exclude .codex \
  --exclude target \
  --exclude .env \
  --exclude 'runpod/.env' \
  ./ \
  "$host:/workspace/catena-lang/"
