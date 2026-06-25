# RunPod Development Image

This image is a small GPU-enabled development environment for running `catena-lang` Rust tests on RunPod.

It provides:

- CUDA 12.6 development base image with `nvcc` on `PATH`
- Rust stable with `rustfmt` and `clippy`
- C/C++ build tools used by runtime tests
- SSH server for interactive access and rsync-based code sync
- Basic debugging/editing tools such as `git`, `jq`, `ripgrep`, `tmux`, `vim`, and `nano`

It intentionally does not copy the repository source into the image. Source code is synced into `/workspace/catena-lang` after the pod starts, using `scripts/runpod-sync.sh`.

The image entrypoint writes RunPod's injected `PUBLIC_KEY` value to `/root/.ssh/authorized_keys`, then starts `sshd` in the foreground. This keeps RunPod account SSH settings as the source of truth for direct TCP SSH access.

RunPod documents this `PUBLIC_KEY` flow for custom images here:

https://docs.runpod.io/pods/configuration/use-ssh#full-ssh-via-public-ip-with-key-authentication

Build it from the repository root so the Dockerfile can copy files from this folder:

```bash
docker build -f runpod/dev-image/Dockerfile -t catena-runpod-dev .
```
