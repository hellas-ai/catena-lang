# RunPod Development Image

This image is a small GPU-enabled development environment for running `catena-lang` Rust tests on RunPod.

It provides:

- CUDA 12.6 development base image
- Rust stable with `rustfmt` and `clippy`
- C/C++ build tools used by runtime tests
- SSH server for interactive access and rsync-based code sync
- Basic debugging/editing tools such as `git`, `jq`, `ripgrep`, `tmux`, `vim`, and `nano`

It intentionally does not copy the repository source into the image. Source code is synced into `/workspace/catena-lang` after the pod starts, using `scripts/runpod-sync.sh`.

The image entrypoint starts `sshd` and then keeps the container alive.

Build it from the repository root so the Dockerfile can copy files from this folder:

```bash
docker build -f runpod/dev-image/Dockerfile -t catena-runpod-dev .
```
