#!/usr/bin/env bash
set -euo pipefail

mkdir -p /run/sshd /root/.ssh
chmod 700 /root/.ssh

# RunPod injects account SSH keys through PUBLIC_KEY for custom images.
# See: https://docs.runpod.io/pods/configuration/use-ssh#full-ssh-via-public-ip-with-key-authentication
if [[ -n "${PUBLIC_KEY:-}" ]]; then
  printf '%s\n' "$PUBLIC_KEY" > /root/.ssh/authorized_keys
fi

if [[ -f /root/.ssh/authorized_keys ]]; then
  chmod 600 /root/.ssh/authorized_keys
fi

ssh-keygen -A
exec /usr/sbin/sshd -D -e
