#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/runpod-sync.sh [--dry-run] <host> <ssh-key> [port]

Arguments:
  host      Runpod SSH host. Use user@host when the user is not root.
  ssh-key   SSH private key to use.
  port      SSH port. Defaults to 22.

Example:
  scripts/runpod-sync.sh root@1.2.3.4 ~/.ssh/runpod_ed25519 2222
  scripts/runpod-sync.sh --dry-run root@1.2.3.4 ~/.ssh/runpod_ed25519 2222

Equivalent SSH command:
  ssh -i ~/.ssh/runpod_ed25519 -p 2222 root@1.2.3.4
EOF
}

dry_run=()
if [[ "${1:-}" == "--dry-run" || "${1:-}" == "-n" ]]; then
  dry_run=(--dry-run --itemize-changes)
  shift
fi

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
# Sync only the files needed to run catena-lang tests on the pod. Cargo still
# needs the workspace root and member manifests, but it does not need the other
# packages' source trees when running `cargo test -p catena-lang`.
rsync -rlptDz --delete --prune-empty-dirs --info=progress2 "${dry_run[@]}" \
  -e "ssh -i $key -p $port -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new" \
  --include '/Cargo.lock' \
  --include '/Cargo.toml' \
  --include '/catena-cli/' \
  --include '/catena-cli/Cargo.toml' \
  --include '/catena-cli/src/' \
  --include '/catena-cli/src/main.rs' \
  --include '/catena-core/' \
  --include '/catena-core/Cargo.toml' \
  --include '/catena-core/src/' \
  --include '/catena-core/src/lib.rs' \
  --include '/catena-lang/' \
  --include '/catena-lang/Cargo.toml' \
  --include '/catena-lang/src/***' \
  --include '/catena-lang/tests/***' \
  --include '/catena-lang/examples/***' \
  --include '/catena-lang/stdlib/***' \
  --exclude '*' \
  ./ \
  "$host:/workspace/catena-lang/"
