# RunPod Setup

This folder contains the pieces needed to run `catena-lang` GPU/runtime tests on a RunPod pod.

## Requirements

- A RunPod API key.
- A RunPod container registry auth entry for GHCR.
- A RunPod account SSH public key in Settings.
- A matching local private key, for example `~/.ssh/runpod_ed25519`.
- The dev image built and pushed to GHCR. The pod spec references:

```text
ghcr.io/hellas-ai/catena-lang/runpod-dev:sha-4960ace
```

## Environment

Create `runpod/.env` from the example:

```bash
cp runpod/.env.runpod.example runpod/.env
```

Set:

```bash
RUNPOD_API_KEY=...
RUNPOD_REGISTRY_AUTH_ID=...
```

`RUNPOD_REGISTRY_AUTH_ID` is the RunPod registry auth object ID, not the GitHub token. The auth object should use `ghcr.io`, a GitHub username, and a PAT that can read the GHCR package.

## Image

The dev image lives in `runpod/dev-image`. It includes CUDA, Rust, build tools, SSH, and rsync.

Build locally from the repo root if needed:

```bash
docker build -f runpod/dev-image/Dockerfile -t catena-runpod-dev .
```

In normal use, GitHub Actions builds and pushes the GHCR image.

## Pod

Create a pod:

```bash
RUNPOD_ENV_FILE=runpod/.env ./scripts/runpod.sh pod-create runpod/pod.catena-dev.example.json
```

List pods:

```bash
RUNPOD_ENV_FILE=runpod/.env ./scripts/runpod.sh pod-list
```

Check available low-cost Secure Cloud GPU candidates:

```bash
RUNPOD_ENV_FILE=runpod/.env ./scripts/runpod.sh gpu-candidates SECURE 0.50
```

## SSH And Sync

Use the direct TCP port that RunPod maps to container port `22`.

If RunPod shows:

```text
213.173.99.23:14671 -> 22
```

sync with:

```bash
scripts/runpod-sync.sh root@213.173.99.23 ~/.ssh/runpod_ed25519 14671
```

SSH directly with:

```bash
ssh -i ~/.ssh/runpod_ed25519 -p 14671 root@213.173.99.23
```

## Tests

On the pod:

```bash
cd /workspace/catena-lang
CATENA_GPU_DIALECT=cuda cargo test -p catena-lang --features runtime-tests --test runtime
```
