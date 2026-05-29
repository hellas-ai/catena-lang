#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODE="${1:-check}"
case "$MODE" in
  check | update) ;;
  *)
    echo "usage: $0 [check|update]" >&2
    exit 2
    ;;
esac

if [[ -n "${CATENA_CMD:-}" ]]; then
  read -r -a CATENA <<< "$CATENA_CMD"
else
  CATENA=(cargo run -q -p catena-cli --)
fi
COMMON=(stdlib/core.hex stdlib/gpu.hex stdlib/gpu.proof.hex)
SNAPSHOTS="$ROOT/tests/e2e/snapshots"
ACTUAL="$ROOT/target/e2e/actual"

run_catena() {
  echo "+ ${CATENA[*]} $*"
  "${CATENA[@]}" "$@"
}

run_snapshot() {
  local output="$1"
  shift

  mkdir -p "$(dirname "$ACTUAL/$output")"
  echo "+ ${CATENA[*]} $* --output target/e2e/actual/$output"
  "${CATENA[@]}" "$@" --output "$ACTUAL/$output"
}

echo "Checking top-level examples"
for example in examples/*.hex; do
  run_catena check "${COMMON[@]}" "$example"
done

echo "Checking puzzle examples"
for puzzle in examples/puzzles/*.hex; do
  run_catena check "${COMMON[@]}" "$puzzle"
done

rm -rf "$ACTUAL"
mkdir -p "$ACTUAL"

echo "Compiling core/control examples"
run_snapshot compile/user-u32-identity.structured-ir \
  compile "${COMMON[@]}" examples/user-program.hex \
  --emit structured-ir \
  --theory control \
  --entry user.u32.identity \
  --no-proof
run_snapshot compile/user-u32-inc-unless-max.structured-ir \
  compile "${COMMON[@]}" examples/user-program.hex \
  --emit structured-ir \
  --theory data \
  --entry user.u32.inc-unless-max \
  --no-proof

echo "Compiling CUDA examples"
run_snapshot compile/fill-one-array.cuda \
  compile "${COMMON[@]}" examples/fill-one-array.hex \
  --emit cuda \
  --theory data \
  --entry user.f32.fill-one \
  --proof examples/fill-one-array.proof.hex
run_snapshot compile/shared-memory.cuda \
  compile "${COMMON[@]}" examples/shared-memory.hex \
  --emit cuda \
  --theory data \
  --entry user.f32.shared-one \
  --no-proof
run_snapshot compile/static-shared-memory.cuda \
  compile "${COMMON[@]}" examples/static-shared-memory.hex \
  --emit cuda \
  --theory data \
  --entry user.f32.static-shared-one \
  --no-proof
run_snapshot compile/two-shared-two-global.cuda \
  compile "${COMMON[@]}" examples/two-shared-two-global.hex \
  --emit cuda \
  --theory data \
  --entry user.f32.two-shared-two-global \
  --no-proof

echo "Compiling CUDA puzzle examples"
run_snapshot compile/map.cuda \
  compile "${COMMON[@]}" examples/puzzles/map.hex \
  --emit cuda \
  --theory data \
  --entry user.f32.map-add-ten \
  --proof examples/puzzles/map.proof.hex
run_snapshot compile/zip.cuda \
  compile "${COMMON[@]}" examples/puzzles/zip.hex \
  --emit cuda \
  --theory data \
  --entry user.f32.zip-add \
  --no-proof
run_snapshot compile/map-square-2d.cuda \
  compile "${COMMON[@]}" examples/puzzles/map-square-2d.hex \
  --emit cuda \
  --theory data \
  --entry user.f32.map-square-2d-add-ten \
  --no-proof
run_snapshot compile/map-square-2d-block.cuda \
  compile "${COMMON[@]}" examples/puzzles/map-square-2d.hex \
  --emit cuda \
  --theory data \
  --entry user.f32.map-square-2d-block-add-ten \
  --no-proof
run_snapshot compile/broadcast.cuda \
  compile "${COMMON[@]}" examples/puzzles/broadcast.hex \
  --emit cuda \
  --theory data \
  --entry user.f32.broadcast-add \
  --no-proof
run_snapshot compile/broadcast-singleton-matrix-inputs.cuda \
  compile "${COMMON[@]}" examples/puzzles/broadcast.hex \
  --emit cuda \
  --theory data \
  --entry user.f32.broadcast-add-singleton-matrix-inputs \
  --proof examples/puzzles/broadcast.proof.hex

echo "Compiling static CUDA shared-memory variants"
run_snapshot compile/static-shared-memory-tile-16x16.cuda \
  compile "${COMMON[@]}" examples/static-shared-memory.hex \
  --emit cuda \
  --theory data \
  --entry user.f32.static-shared-one \
  --cuda-static tile_rows=16 \
  --cuda-static tile_cols=16 \
  --no-proof
run_snapshot compile/two-shared-two-global-tile-8x16.cuda \
  compile "${COMMON[@]}" examples/two-shared-two-global.hex \
  --emit cuda \
  --theory data \
  --entry user.f32.two-shared-two-global \
  --cuda-static tile_rows=8 \
  --cuda-static tile_cols=16 \
  --no-proof

if [[ "$MODE" == "update" ]]; then
  rm -rf "$SNAPSHOTS"
  mkdir -p "$(dirname "$SNAPSHOTS")"
  cp -R "$ACTUAL" "$SNAPSHOTS"
  echo "Updated e2e snapshots in tests/e2e/snapshots"
else
  if [[ ! -d "$SNAPSHOTS" ]]; then
    echo "Missing e2e snapshots. Run \`make e2e-update\` to create them." >&2
    exit 1
  fi

  if ! diff -ru "$SNAPSHOTS" "$ACTUAL"; then
    echo
    echo "E2E snapshots differ." >&2
    echo "Fix the compiler output or run \`make e2e-update\` and commit the snapshot changes." >&2
    exit 1
  fi

  echo "E2E snapshots match"
fi

echo "Examples passed"
