# Explicit e2e cases. The harness intentionally does not discover tests.

test_case "check fill-one-array" \
  --command check \
  --input "examples/fill-one-array.hex"
test_case "check fill-one-array proof" \
  --command check \
  --input "examples/fill-one-array.proof.hex"
test_case "check shared-memory" \
  --command check \
  --input "examples/shared-memory.hex"
test_case "check static-shared-memory" \
  --command check \
  --input "examples/static-shared-memory.hex"
test_case "check two-shared-two-global" \
  --command check \
  --input "examples/two-shared-two-global.hex"
test_case "check user-program" \
  --command check \
  --input "examples/user-program.hex"
test_case "check broadcast" \
  --command check \
  --input "examples/puzzles/broadcast.hex"
test_case "check broadcast proof" \
  --command check \
  --input "examples/puzzles/broadcast.proof.hex"
test_case "check map-square-2d" \
  --command check \
  --input "examples/puzzles/map-square-2d.hex"
test_case "check map" \
  --command check \
  --input "examples/puzzles/map.hex"
test_case "check map proof" \
  --command check \
  --input "examples/puzzles/map.proof.hex"
test_case "check zip" \
  --command check \
  --input "examples/puzzles/zip.hex"

test_case "compile user-u32-identity" \
  --command compile \
  --input "examples/user-program.hex" \
  --expected "compile/user-u32-identity.structured-ir" \
  -- \
  --emit structured-ir \
  --theory control \
  --entry user.u32.identity \
  --no-proof
test_case "compile user-u32-inc-unless-max" \
  --command compile \
  --input "examples/user-program.hex" \
  --expected "compile/user-u32-inc-unless-max.structured-ir" \
  -- \
  --emit structured-ir \
  --theory data \
  --entry user.u32.inc-unless-max \
  --no-proof

test_case "compile fill-one-array" \
  --command compile \
  --input "examples/fill-one-array.hex" \
  --expected "compile/fill-one-array.cuda" \
  -- \
  --emit cuda \
  --theory data \
  --entry user.f32.fill-one \
  --proof examples/fill-one-array.proof.hex
test_case "compile shared-memory" \
  --command compile \
  --input "examples/shared-memory.hex" \
  --expected "compile/shared-memory.cuda" \
  -- \
  --emit cuda \
  --theory data \
  --entry user.f32.shared-one \
  --no-proof
test_case "compile static-shared-memory" \
  --command compile \
  --input "examples/static-shared-memory.hex" \
  --expected "compile/static-shared-memory.cuda" \
  -- \
  --emit cuda \
  --theory data \
  --entry user.f32.static-shared-one \
  --no-proof
test_case "compile two-shared-two-global" \
  --command compile \
  --input "examples/two-shared-two-global.hex" \
  --expected "compile/two-shared-two-global.cuda" \
  -- \
  --emit cuda \
  --theory data \
  --entry user.f32.two-shared-two-global \
  --no-proof

test_case "compile map" \
  --command compile \
  --input "examples/puzzles/map.hex" \
  --expected "compile/map.cuda" \
  -- \
  --emit cuda \
  --theory data \
  --entry user.f32.map-add-ten \
  --proof examples/puzzles/map.proof.hex
test_case "compile zip" \
  --command compile \
  --input "examples/puzzles/zip.hex" \
  --expected "compile/zip.cuda" \
  -- \
  --emit cuda \
  --theory data \
  --entry user.f32.zip-add \
  --no-proof
test_case "compile map-square-2d" \
  --command compile \
  --input "examples/puzzles/map-square-2d.hex" \
  --expected "compile/map-square-2d.cuda" \
  -- \
  --emit cuda \
  --theory data \
  --entry user.f32.map-square-2d-add-ten \
  --no-proof
test_case "compile map-square-2d-block" \
  --command compile \
  --input "examples/puzzles/map-square-2d.hex" \
  --expected "compile/map-square-2d-block.cuda" \
  -- \
  --emit cuda \
  --theory data \
  --entry user.f32.map-square-2d-block-add-ten \
  --no-proof
test_case "compile broadcast" \
  --command compile \
  --input "examples/puzzles/broadcast.hex" \
  --expected "compile/broadcast.cuda" \
  -- \
  --emit cuda \
  --theory data \
  --entry user.f32.broadcast-add \
  --no-proof
test_case "compile broadcast singleton matrix inputs" \
  --command compile \
  --input "examples/puzzles/broadcast.hex" \
  --expected "compile/broadcast-singleton-matrix-inputs.cuda" \
  -- \
  --emit cuda \
  --theory data \
  --entry user.f32.broadcast-add-singleton-matrix-inputs \
  --proof examples/puzzles/broadcast.proof.hex

test_case "compile static-shared-memory tile 16x16" \
  --command compile \
  --input "examples/static-shared-memory.hex" \
  --expected "compile/static-shared-memory-tile-16x16.cuda" \
  -- \
  --emit cuda \
  --theory data \
  --entry user.f32.static-shared-one \
  --cuda-static tile_rows=16 \
  --cuda-static tile_cols=16 \
  --no-proof
test_case "compile two-shared-two-global tile 8x16" \
  --command compile \
  --input "examples/two-shared-two-global.hex" \
  --expected "compile/two-shared-two-global-tile-8x16.cuda" \
  -- \
  --emit cuda \
  --theory data \
  --entry user.f32.two-shared-two-global \
  --cuda-static tile_rows=8 \
  --cuda-static tile_cols=16 \
  --no-proof
