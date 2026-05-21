#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -n "${CATENA_CMD:-}" ]]; then
  read -r -a CATENA <<< "$CATENA_CMD"
else
  CATENA=(cargo run -q -p catena-cli --)
fi
COMMON=(stdlib/core.hex stdlib/gpu.hex)

run_catena() {
  echo "+ ${CATENA[*]} $*"
  # shellcheck disable=SC2068
  ${CATENA[@]} "$@"
}

run_catena_quiet() {
  echo "+ ${CATENA[*]} $*"
  # shellcheck disable=SC2068
  ${CATENA[@]} "$@" >/dev/null
}

echo "Checking top-level examples"
for example in examples/*.hex; do
  run_catena check "${COMMON[@]}" "$example"
done

echo "Compiling core/control examples"
run_catena_quiet compile "${COMMON[@]}" examples/user-program.hex \
  --emit structured-ir \
  --theory control \
  --entry user.u32.identity
run_catena_quiet compile "${COMMON[@]}" examples/user-program.hex \
  --emit structured-ir \
  --theory data \
  --entry user.u32.inc-unless-max

echo "Compiling CUDA examples"
run_catena_quiet compile "${COMMON[@]}" examples/fill-one-array.hex \
  --emit cuda \
  --theory data \
  --entry user.f32.fill-one
run_catena_quiet compile "${COMMON[@]}" examples/shared-memory.hex \
  --emit cuda \
  --theory data \
  --entry user.f32.shared-one
run_catena_quiet compile "${COMMON[@]}" examples/static-shared-memory.hex \
  --emit cuda \
  --theory data \
  --entry user.f32.static-shared-one
run_catena_quiet compile "${COMMON[@]}" examples/two-shared-two-global.hex \
  --emit cuda \
  --theory data \
  --entry user.f32.two-shared-two-global

echo "Compiling static CUDA shared-memory variants"
run_catena_quiet compile "${COMMON[@]}" examples/static-shared-memory.hex \
  --emit cuda \
  --theory data \
  --entry user.f32.static-shared-one \
  --cuda-static tile_rows=16 \
  --cuda-static tile_cols=16
run_catena_quiet compile "${COMMON[@]}" examples/two-shared-two-global.hex \
  --emit cuda \
  --theory data \
  --entry user.f32.two-shared-two-global \
  --cuda-static tile_rows=8 \
  --cuda-static tile_cols=16

echo "Examples passed"
