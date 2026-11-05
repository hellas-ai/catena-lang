# BF16 WMMA simulation and validation

Set `CATENA_WMMA_EMULATE=1` when running a model to simulate the existing
`wmma.mma-bf16-f32` primitive on HIP or CUDA. Only the exact value `1` enables
simulation; unset, `0`, and other values select the existing native path.
CUDA and unsupported HIP targets still trap if native WMMA is selected.

Generated C++ contains both implementations and native/emulated specializations
of each kernel containing MMA. Generated host code reads the variable at **each
affected kernel launch** and selects a specialization. The choice is compile-time
constant inside the kernel: native kernels have no reachable simulator call or
runtime MMA branch, avoiding spill overhead from the simulator's call ABI.
Kernels without MMA are not specialized. Changing the variable between calls
affects an already-loaded artifact; no recompilation or cache clearing is needed.
Environment changes must be coordinated by the caller if multiple threads launch
kernels. Public Hex types,
fragment layout, and the runtime ABI are unchanged.

WMMA requires complete, converged groups of 32 lanes. A/B fragments in lanes
0–15 must be duplicated in lanes 16–31. Each lane owns eight accumulator
values: element `i` of lane `l` is output row `l/16 + 2*i`, column `l%16`.
The simulator gathers A fragments with width-32 shuffles and uses local B
fragments, including for multidimensional blocks and multiple groups per block.
Both HIP and CUDA use the same portable GPU implementation, with no native
dot2/MMA shortcut or host round trip. Eight independent output accumulators
advance together, exposing instruction-level parallelism while preserving the
eight ordered pair updates within each output. A fixed register window advances
input pairs without dynamic indexing; the output loop is unrolled, while the
pair loop stays rolled to limit instruction-cache pressure.

The finite path uses bounded 32-bit GPU integer operations, followed by a native
round-to-nearest integer-to-F32 conversion and an exponent adjustment in bits.
It preserves the exact underflow cutoff before conversion. Nonfinite handling
is kept out of the inlined finite path. The original general 64-bit integer
implementation is retained only as a test oracle in `reference64.h`.

Performance relative to native WMMA or a conventional SIMT matmul requires
measurement on the target GPU; portability and bitwise agreement alone do not
establish that the fallback is fast. There is no benchmark result for this
optimized implementation yet.

`arithmetic.h` preserves the standalone `rdna35_wmma/rdna35_wmma.hip` BF16
model: eight ordered dot2 rounds, signed corrections including negative zeros,
input subnormal flushing, the measured underflow cutoff, explicit nearest-even
rounding, and exact infinity/NaN selection. It has no external source dependency.
The original model was validated on gfx1151; other RDNA3/3.5 targets need their
own strict hardware run. Matching this primitive alone does not establish
whole-model agreement if other operations differ between backends.

## Numerical tests

The numerical stress tests, corpus generators, references, and fixtures live
here. Rust builds a self-contained C++ validator containing the actual
production code; its correctness checks compare numerical results. GPU runs
additionally use the real primitive lowerer and generated host launch path, with an independent
native WMMA oracle on AMD. Both backends exercise the same portable device
implementation. CPU results are compared against both the original integer64
oracle and a separate FP64 alignment/rounding implementation. Additional
scalar sweeps compare the optimized rounding against integer64 at halfway,
carry, underflow, and overflow boundaries and across seeded random values.

Run the default CPU stress test (requires a C++17 compiler, no GPU/SDK):

```sh
cargo test --offline --locked -p catena-lang --lib wmma_sim -- --nocapture
```

Run strict native/simulated validation on AMD:

```sh
HIPCC=/opt/rocm/bin/hipcc \
CATENA_WMMA_TEST_ARCH=gfx1151 \
CATENA_WMMA_TEST_DIR=/tmp/catena-wmma-gfx1151 \
cargo test --offline --locked -p catena-lang --lib --features runtime-tests \
  wmma_sim::tests::hip_stress -- --ignored --nocapture
```

Use `gfx1100` (or the actual supported gfx110x/gfx115x target) and a separate
output directory for RDNA3 validation. Explicit GPU runs fail when the GPU or
SDK is unavailable; they do not silently skip validation.

For NVIDIA, select the CUDA test:

```sh
CATENA_WMMA_TEST_ARCH=sm_80 \
CATENA_WMMA_TEST_DIR=/tmp/catena-wmma-cuda \
cargo test --offline --locked -p catena-lang --lib --features runtime-tests \
  wmma_sim::tests::cuda_stress -- --ignored --nocapture
```

CUDA compares simulation against CPU integer and FP64 results, and reports
`hardware_verified: false` because it cannot check native RDNA WMMA.

Each run includes 1,000 tiles per random family by default, plus all directed
suites, regardless of the random trial count:

- Fixed cancellation, signed-zero, subnormal, underflow-cutoff, and NaN goldens.
- Every k selection and every output lane, with independently known results.
- Seeded bounded, uniform raw-bit, cancellation, and exponent-boundary tiles.
- All 485,537 representative special-value combinations, packed onto diagonals;
  all off-diagonal cross-combinations are also checked.
- All 353,736 normal-boundary cases and every BF16 encoding in all four
  multiplicand roles.
- Repeated dependent WMMA operations with raw-bit and finite inputs and nonzero
  accumulators; partial final blocks, multiple waves, and 1D/2D/3D blocks.

GPU runs compare raw output bits with no NaN relaxation. On AMD, every batch
also toggles unset, `1`, `0`, invalid strings, empty, `01`, and `true` in the
same compiled module, returning to `1` repeatedly and checking all three block
shapes in each mode. The simulator must match both CPU results and the independent native hardware oracle. Counts in reports are
unique output positions; additional mode/geometry checks are not added again.

Configuration is inherited by the test subprocess:

| Variable | Default / purpose |
| --- | --- |
| `CXX`, `HIPCC`, `NVCC` | `c++`, `hipcc`, `nvcc` compiler executable |
| `CATENA_WMMA_TEST_ARCH` | `gfx1151` for HIP, `sm_80` for CUDA |
| `CATENA_WMMA_TEST_TRIALS` | `1000`; random tiles per family (`0` keeps directed suites) |
| `CATENA_WMMA_TEST_SEED` | `20261028`; deterministic SplitMix64 seed |
| `CATENA_WMMA_TEST_BATCH` | `128`; tiles per batch, 1–4096 |
| `CATENA_WMMA_TEST_DIR` | Temporary directory; set to preserve source, executable, JSON and failing tiles |
| `CATENA_WMMA_TEST_REPLAY` | Absolute path to a saved failing tile |
| `CATENA_WMMA_TEST_BUILD_ONLY` | If present, compile the GPU harness without running it; **not hardware validation** |

Use `CATENA_WMMA_TEST_TRIALS=100000` for the standalone's extended stress run.
Reports record source SHA-256, compiler, device/architecture, seeds, suite and
NaN counts, and pass/fail status. The first mismatch saves `wmma-failure.txt`
with complete A/B/C bits and recurrence count. Replay it by setting
`CATENA_WMMA_TEST_REPLAY` while keeping `CATENA_WMMA_TEST_DIR` set.
The preserved executable can also be run directly with `--trials`, `--seed`,
`--batch`, `--report`, and `--replay` arguments.

CPU-only and build-only runs do not establish native hardware equivalence.
