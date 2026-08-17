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
- `GpuDsl/Correctness.lean` gives the direct ownership lemmas and structural
  totality/determinism argument used by the typed DSL.

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
- `WriteOwnership` maps every shared index to exactly one physical block
  thread. One thread may own any number of indices.
- `KernelM.initializeShared` records a `SharedWritePhase`: each physical thread
  writes the cells assigned to it immediately before the matching barrier.
- `Thread config blockState state` is a reference supplied by `launch`; its
  dependent type carries the current block and launch-created state types.
- `KernelM thread before after a` ties statements to a thread and records their
  exact synchronization-trace transition.
- A `ScheduleCertificate` assigns every logical output index to exactly one
  `ThreadId`. A thread may own zero, one, or many output cells.
- `Scheduled` gives one kernel invocation an optional work item and proves
  ownership. The launch implementation may supply repeated work items when a
  thread owns multiple cells, or `none` when it owns none.
- `Kernel.requirements` records uniform launch facts such as
  `block = tile × tile × 1`; `launch` accepts their proof before any thread
  executes.
- `Kernel.shared` declares block-scoped allocations. `launch` creates the
  matching `SharedState`, and kernels retrieve buffers through total
  `SharedState.get` calls carrying `SharedRef` membership proofs.
- `KernelM.syncBlock` is independent of shared memory and proves that every
  physical thread crossed the barrier. Combining total ownership, a write phase
  at the barrier's exact pre-trace, and this synchronization proves that every
  shared cell is defined afterward. It also appends the named barrier to the
  trace, and `Kernel.body` requires one final trace for every thread.
- `KernelM.readShared` requires a `DefinedAt` proof for the exact shared-memory
  index and synchronization trace being read. Shared reads cannot be hidden
  inside a pure `Value`.
- `TileLoopState trace` carries one block-level `SharedWritableAt` capability at
  the top of each tile iteration. Fresh shared memory supplies the initial
  capability, and both tile buffers share it.
- The `readShared` commands are sequenced before `.tileConsumed`. That barrier
  proves every block thread completed its preceding reads and restores the
  block-level writable capability. Consequently, the next iteration cannot
  start its write phase without the second barrier.
- `KernelM.foldFinD` supports this invariant by allowing its loop-carried state
  type to depend on the synchronization trace.

The launch surface is intentionally close to CUDA/HIP:

```lean
let kernel := tiledMatmulKernel tile tilePositive a b
launch config outputBuffer outputLayout certificate kernel requirementsProof
```

`launch` is an axiom here: this project specifies its dependent interface but
does not provide an implementation. A backend creates block/thread state,
constructs thread references and scheduled work items, stores returned values
through the output layout, and uses the supplied `Kernel.requirements` proof.
