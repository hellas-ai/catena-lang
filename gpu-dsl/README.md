# GPU DSL prototype in Lean

This is a small typed, CUDA/HIP-shaped DSL for GPU kernels. It is deliberately
an embedded kernel description rather than a GPU backend: the next iteration
can interpret or compile `KernelM` while keeping its typed interface.

## Entry points

- `Main.lean` is the executable entry point.
- `GpuDsl.lean` is the public library entry point.
- `GpuDsl/Matrix.lean` contains spaces, layouts, pure tensor closures, and
  buffer views.
- `GpuDsl/Launch.lean` contains scheduling, the dependent kernel-body
  interface, and the opaque `launch` primitive.
- `GpuDsl/Shared.lean` contains block-scoped shared-memory specifications,
  allocated states, and proof-indexed lookup.
- `GpuDsl/Matmul.lean` contains one reusable tiled matmul kernel plus perfect
  and predicated launch configurations.
- `PRIMITIVES.md` extracts a language-independent, global-memory-only
  `Kernel`/`launch` specification sufficient for untiled matrix multiplication.

Concrete launch examples live only in `Main.lean`, keeping launch policy and
example dimensions out of the DSL library.

## Logical tensors and physical buffers

A `Buffer space length a` is always a linear chunk of memory. Its element type
and length are compile-time indices. A `Tensor logicalSpace a` is not storage:
it is a pure closure from a bounded logical index to a value:

```lean
Index logicalSpace -> a
```

`Space` currently supports 1D, 2D, and 3D domains. `Layout` maps an index in
one of those spaces to a safe `Fin length` linear offset. The library provides
linear, row-major 2D, column-major 2D, and row-major 3D layouts; callers can
define other layouts by supplying the same bounded mapping.

For example, neither conversion below allocates or copies memory:

```lean
let linear := Buffer.asVector buffer
let rowMajor := Tensor.reshape linear Layout.rowMajor2D
let columnMajor := Tensor.reshape linear Layout.columnMajor2D
```

`Tensor.map`, `Tensor.zipWith`, `Tensor.reindex`, and `Matrix.transpose`
compose closures lazily. A global-buffer view has element type `Value a`,
representing a staged global load expression; it still carries no launch
configuration. Shared-memory reads and stores remain explicit kernel
statements.

`tiledMatmulKernel` is generic over its scalar type. It requires only zero,
addition, and multiplication; `Float` is chosen by the launch examples rather
than by the kernel.

Run the type checker and demo from this directory:

```sh
lake build
lake exe gpu-dsl-demo
```

The core design makes several future proof obligations natural:

- `Buffer space n a` records address space and the extent of linear storage.
- `Tensor space a` represents data as a composable pure closure, independently
  of launch configuration and physical layout.
- Buffer reads accept only `Fin n`, so out-of-bounds access cannot be expressed.
- `Thread config blockState state` is a reference supplied by `launch`; its
  dependent type carries the current block and launch-created state types.
- `KernelM ownershipPlan scalar thread before after a` records only the
  synchronization trace transition. The fixed launch ownership plan is part
  of its type; ownership and initialization evidence remain explicit tokens.
- A `ScheduleCertificate` assigns every logical output index to exactly one
  `ThreadId`. A thread may own zero, one, or many output cells.
- `OwnedOutputs` gives each physical thread the complete, duplicate-free list
  of logical outputs assigned to it. `launch` invokes the body exactly once per
  thread, and the body returns one value for every item in that list.
- `Kernel.requirements` records uniform launch facts such as
  `block = tile × tile × 1`; `launch` accepts their proof before any thread
  executes.
- `Kernel.shared` declares block-scoped allocations. `launch` creates the
  matching `SharedState`, and kernels retrieve buffers through total
  `SharedState.get block "name"` calls whose `SharedRef` membership proofs are
  synthesized. The declared name is retained in the handle's type,
  distinguishing allocations such as `SharedBuffer block "tileA" ...` and
  `SharedBuffer block "tileB" ...`.
- `Kernel.sharedOwner` assigns every logical shared cell to a block thread.
  The unimplemented `launch` primitive realizes that plan and supplies each
  thread with a checked `LaunchOwnership` grant. The kernel uses it once,
  before its loop, to obtain its initial exclusive `Owns` tokens.
  `KernelM.writeShared` requires `Owns` and returns a `Wrote` token.
  `KernelM.readShared` requires both `Defined` and a shared `ReadShare`; the
  share remains valid for every read at that trace.
- `KernelM.syncBlock` is the collective barrier rule. It lifts a fact supplied
  by the current thread to the same fact for every thread in the block and
  applies its `SyncSpec`. Advancing the trace resets every old permission. A
  spec may grant either one or more `ReadShare`s or one or more `Owns` tokens,
  never a mixture. Every ownership grant must prove that the fixed launch plan
  assigns the logical cell to the current thread and that its layout is
  injective.
- `LaunchFacts` exposes those launch guarantees to kernel proofs: every
  physical thread is invoked exactly once and completes the kernel's exact
  common synchronization trace. These facts are supplied by `launch` and must
  hold for any fair GPU implementation; the matmul body does not use them yet.
- Matmul's launch ownership plan assigns logical cell `(row,column)` to thread
  `(column,row)` for both shared tiles.
- Coverage is proved at each matmul read by choosing its writer thread. A write
  by that thread produces `Defined` evidence immediately after `tileLoaded`.
- In matmul, `tileLoaded` resets `Owns` and grants `ReadShare` while publishing
  `Wrote` as `Defined`; `tileConsumed` resets those read shares and directly
  grants the next round's exclusive `Owns` tokens.
- `KernelM.foldFinD` represents a statically bounded loop with an ordinary
  accumulator, a separate trace-indexed resource, and an exact
  synchronization-trace transition.

The launch surface is intentionally close to CUDA/HIP:

```lean
let kernel := tiledMatmulKernel tile tilePositive a b
launch config outputBuffer outputLayout certificate kernel requirementsProof
```

`launch` is an axiom here: this project specifies its dependent interface but
does not provide an implementation. A backend creates block/thread state,
constructs thread references and complete owned-output lists, stores returned
values through the output layout, and uses the supplied `Kernel.requirements`
proof.
