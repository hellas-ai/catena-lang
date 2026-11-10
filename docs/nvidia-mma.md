# NVIDIA BF16 MMA

The default stdlib exposes native NVIDIA warp MMA for CUDA `sm_80+`:
`mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`. The three fragment types
are distinct device values, each wrapped in `val` at the Hex interface.

| Operation | Inputs | Result |
| --- | --- | --- |
| `mma.m16n8k16.bf16.a.pack` | Four `bf16x2` | `mma.m16n8k16.bf16.a` |
| `mma.m16n8k16.bf16.b.pack` | Two `bf16x2` | `mma.m16n8k16.bf16.b` |
| `mma.m16n8k16.f32.acc.pack` | Four `f32` | `mma.m16n8k16.f32.acc` |
| `mma.m16n8k16.f32.acc.zero` | Unit | `mma.m16n8k16.f32.acc` |
| `mma.m16n8k16.f32.acc.get` | Accumulator, `u32` index | `f32` |
| `mma.m16n8k16.bf16-f32` | A, B, accumulator | Accumulator |

Accumulator access checks `index < 4`, including dynamic indices. Host execution
and HIP compilation are rejected. CUDA device compilation below `sm_80` reports
an error. There is no software fallback. Explicit PTX preserves MMA arithmetic
when surrounding compilation uses `--fmad=false`. The existing AMD WMMA and its
emulator are separate primitives; `CATENA_WMMA_EMULATE` does not affect this API.

Packing preserves PTX operand register order. `bf16x2.pack` places its first
argument in the low 16 bits. For lane `l`, define `g=l/4` and `t=l%4`:

| Fragment/register | Matrix coordinate of low BF16 element |
| --- | --- |
| A/0 | `(g, 2*t)` |
| A/1 | `(g+8, 2*t)` |
| A/2 | `(g, 2*t+8)` |
| A/3 | `(g+8, 2*t+8)` |
| B/0 | `(2*t, g)` |
| B/1 | `(2*t+8, g)` |

The high BF16 element advances K by one. Accumulator element `i` corresponds to
row `g+8*(i/2)` and column `2*t+i%2`. See NVIDIA's
[fragment layout documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-fragment-mma-16816-float).

Callers must ensure all 32 lanes participate convergently in each MMA. Compiler
structural validation does not prove convergence. Shared-memory staging requires
the caller's usual synchronization barriers. Arithmetic has no cross-vendor
bitwise reproducibility guarantee.

## Validation

Local checks (Rust and a C++ compiler, without CUDA or a GPU):

```sh
cargo test --offline --locked -p catena-lang --lib nvidia_mma
cargo test --offline --locked -p catena-lang --test nvidia_mma
```

These cover declarations through a real launch, type mapping, pack order,
accumulator bounds, structural validation, backend rejection, PTX source and
conditional support emission. The CPU fragment check only executes packing and
access, not MMA arithmetic. `build.rs` fingerprints the new helpers automatically
because it hashes every source file under `src` and `stdlib`.

CUDA toolkit checks, without a GPU:

```sh
cargo test --offline --locked -p catena-lang --lib nvidia_mma::tests::cuda_toolkit -- --ignored --nocapture
CATENA_MMA_BUILD_ONLY=1 cargo test --offline --locked -p catena-lang --lib nvidia_mma::tests::cuda_hardware -- --ignored --nocapture
```

The first checks generated CUDA for `sm_80` with `--fmad=true/false`, inspects PTX,
and verifies the `sm_75` diagnostic. The second compiles the numerical harness,
checks SASS for `HMMA` using `cuobjdump`, and prints ptxas register/spill usage.
Build-only success does not establish hardware correctness or performance.

External NVIDIA `sm_80+` GPU execution (also requires `nvcc` and `cuobjdump`):

```sh
cargo test --offline --locked -p catena-lang --lib nvidia_mma::tests::cuda_hardware -- --ignored --nocapture
```

Unset `CATENA_MMA_BUILD_ONLY` for execution. The harness uses production helpers
with all 2,048 directed one-hot row/K/column combinations, dense BF16 matrices,
zero/nonzero initial accumulators, one/three K steps, four warps per block and a
partial final block. Results use a double reference and absolute/relative
tolerances. These compiler and harness checks do not measure model performance;
representative model kernels still require their own profiling.
