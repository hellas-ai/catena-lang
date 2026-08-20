# GPU DSL prototype in Lean

This is a small typed, CUDA/HIP-shaped DSL for GPU kernels. It is deliberately
an embedded kernel description rather than a GPU backend: the next iteration
can interpret or compile `KernelM` while keeping its typed interface.

## Entry points

- `Main.lean` is the executable entry point.
- `GpuDsl.lean` is the public library entry point.
- `GpuDsl/Matrix.lean` contains spaces, layouts, and pure tensor closures.
- `GpuDsl/Launch.lean` contains scheduling, the dependent kernel-body
  interface, and the opaque `launch` primitive.
- `GpuDsl/Shared.lean` contains block-scoped shared-memory specifications,
  allocated states, and proof-indexed lookup.
- `GpuDsl/Matmul.lean` contains one reusable tiled matmul kernel plus perfect
  and predicated launch configurations.
- `PRIMITIVES.md` presents the language-independent kernel, global-permission,
  shared-memory, and barrier interfaces in Catena-style notation.

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

`Tensor.map`, `Tensor.zipWith`, `Tensor.reindex`, and `Matrix.transpose`
compose closures lazily. Global-buffer reads are deliberately not pure tensor
views: `KernelM.readGlobal` requires a launch-created permission tied to the
actual buffer. Global and shared memory operations are explicit kernel
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
  Each handle also carries an `AllocationId`, which is used when comparing
  physical cells across different fields of a kernel's input record.
- `Tensor space a` represents data as a composable pure closure, independently
  of launch configuration and physical layout.
- Buffer reads accept only `Fin n`, so out-of-bounds access cannot be expressed.
- `Thread config blockState state` is a reference supplied by `launch`; its
  dependent type carries the current block and launch-created state types.
- `KernelM ownershipPlan scalar thread before after a` records only the
  synchronization trace transition. The fixed launch ownership plan is part
  of its type; ownership and initialization evidence remain explicit tokens.
- `Kernel Inputs SharedScalar` accepts an arbitrary input record rather than a
  privileged list of output cells. Buffers, layouts, scalars, and other
  resources can all be ordinary fields of `Inputs`.
- `Kernel.globalAccess` is a closed plan over the actual buffers in `Inputs`.
  `launch` interprets it as per-thread `GlobalReadShare` and
  `OwnedGlobalCells` permissions. `KernelM.readGlobal` and
  `KernelM.writeGlobal` require those permissions. Read grants carry a region,
  which may depend on the physical thread, and every read must prove that its
  bounded index belongs to that thread's region.
- `Kernel.globalAccessRaceFree` proves the whole global plan spatially safe:
  one physical allocation-and-offset cell has at most one writer, and a written
  cell has no readers. Reads and writes in other regions remain independent.
- An owned global resource includes the complete, duplicate-free list of cells
  assigned to the current `ThreadId`. This generalizes the former
  output-specific `OwnedOutputs` mechanism.
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
- Matmul's global plan grants each thread only the exact `A` and `B` cells it
  loads across the tile loop, plus ownership of its scheduled output cells.
  Its input record proves that the output allocation does not alias `A` or `B`.
- Coverage is proved at each matmul read by choosing its writer thread. A write
  by that thread produces `Defined` evidence immediately after `tileLoaded`.
- In matmul, `tileLoaded` resets `Owns` and grants `ReadShare` while publishing
  `Wrote` as `Defined`; `tileConsumed` resets those read shares and directly
  grants the next round's exclusive `Owns` tokens.
- `KernelM.foldFinD` represents a statically bounded loop with an ordinary
  accumulator and a separate trace-indexed invariant. The initial invariant
  and every callback result have the same shape; matmul's invariant contains
  exactly the two `Owns` tokens needed by the next tile iteration.

The launch surface is intentionally close to CUDA/HIP:

```lean
let inputs : MatmulInputs rows inner cols α :=
  { a, aLayout, b, bLayout, output }
let kernel := tiledMatmulKernel tile tilePositive
launch config kernel inputs requirementsProof
```

`launch` is an axiom here: this project specifies its dependent interface but
does not provide an implementation. A backend creates block/thread state,
constructs thread references, interprets the global and shared access plans,
and uses the supplied `Kernel.requirements` proof. The kernel performs explicit
permission-checked global writes and returns `Unit`.
