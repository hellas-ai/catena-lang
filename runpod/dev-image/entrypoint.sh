#!/usr/bin/env bash
set -euo pipefail

mkdir -p /run/sshd /root/.ssh
chmod 700 /root/.ssh

if [[ -f /root/.ssh/authorized_keys ]]; then
  chmod 600 /root/.ssh/authorized_keys
fi

/usr/sbin/sshd
sleep infinity
