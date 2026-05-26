#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -n "${CATENA_CMD:-}" ]]; then
  read -r -a CATENA <<< "$CATENA_CMD"
else
  CATENA=(cargo run -q -p catena-cli --)
fi
COMMON=(stdlib/core.hex stdlib/gpu.hex stdlib/gpu.proof.hex)

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

echo "Checking puzzle examples"
for puzzle in examples/puzzles/*.hex; do
  run_catena check "${COMMON[@]}" "$puzzle"
done

echo "Compiling core/control examples"
run_catena_quiet compile "${COMMON[@]}" examples/user-program.hex \
  --emit structured-ir \
  --theory control \
  --entry user.u32.identity \
  --no-proof
run_catena_quiet compile "${COMMON[@]}" examples/user-program.hex \
  --emit structured-ir \
  --theory data \
  --entry user.u32.inc-unless-max \
  --no-proof

echo "Compiling CUDA examples"
run_catena_quiet compile "${COMMON[@]}" examples/fill-one-array.hex \
  --emit cuda \
  --theory data \
  --entry user.f32.fill-one \
  --proof examples/fill-one-array.proof.hex
run_catena_quiet compile "${COMMON[@]}" examples/shared-memory.hex \
  --emit cuda \
  --theory data \
  --entry user.f32.shared-one \
  --no-proof
run_catena_quiet compile "${COMMON[@]}" examples/static-shared-memory.hex \
  --emit cuda \
  --theory data \
  --entry user.f32.static-shared-one \
  --no-proof
run_catena_quiet compile "${COMMON[@]}" examples/two-shared-two-global.hex \
  --emit cuda \
  --theory data \
  --entry user.f32.two-shared-two-global \
  --no-proof

echo "Compiling CUDA puzzle examples"
run_catena_quiet compile "${COMMON[@]}" examples/puzzles/map.hex \
  --emit cuda \
  --theory data \
  --entry user.f32.map-add-ten \
  --proof examples/puzzles/map.proof.hex
run_catena_quiet compile "${COMMON[@]}" examples/puzzles/zip.hex \
  --emit cuda \
  --theory data \
  --entry user.f32.zip-add \
  --no-proof
run_catena_quiet compile "${COMMON[@]}" examples/puzzles/map-square-2d.hex \
  --emit cuda \
  --theory data \
  --entry user.f32.map-square-2d-add-ten \
  --no-proof
run_catena_quiet compile "${COMMON[@]}" examples/puzzles/map-square-2d.hex \
  --emit cuda \
  --theory data \
  --entry user.f32.map-square-2d-block-add-ten \
  --no-proof
run_catena_quiet compile "${COMMON[@]}" examples/puzzles/broadcast.hex \
  --emit cuda \
  --theory data \
  --entry user.f32.broadcast-add \
  --no-proof
run_catena_quiet compile "${COMMON[@]}" examples/puzzles/broadcast.hex \
  --emit cuda \
  --theory data \
  --entry user.f32.broadcast-add-singleton-matrix-inputs \
  --proof examples/puzzles/broadcast.proof.hex

echo "Compiling static CUDA shared-memory variants"
run_catena_quiet compile "${COMMON[@]}" examples/static-shared-memory.hex \
  --emit cuda \
  --theory data \
  --entry user.f32.static-shared-one \
  --cuda-static tile_rows=16 \
  --cuda-static tile_cols=16 \
  --no-proof
run_catena_quiet compile "${COMMON[@]}" examples/two-shared-two-global.hex \
  --emit cuda \
  --theory data \
  --entry user.f32.two-shared-two-global \
  --cuda-static tile_rows=8 \
  --cuda-static tile_cols=16 \
  --no-proof

echo "Examples passed"
