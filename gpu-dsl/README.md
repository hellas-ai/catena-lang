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
compose closures lazily. A buffer view has element type `Value a`, representing
a staged load expression; it still carries no launch configuration. Stores
remain explicit kernel statements on a physical `Buffer`.

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
- `KernelM.storeShared` additionally requires an `ExclusiveWrite` certificate:
  an injective block-lane assignment proving that the current thread owns the
  destination index. Shared handles retain their originating block identity.
- `Thread config blockState state` is a reference supplied by `launch`; its
  dependent type carries the current block and launch-created state types.
- `KernelM thread a` ties statements to the thread executing them.
- A `ScheduleCertificate` assigns every logical output index to exactly one
  `ThreadId`. A thread may own zero, one, or many output cells.
- `Scheduled` gives one kernel invocation an optional work item and proves
  ownership. The launch implementation may supply repeated work items when a
  thread owns multiple cells, or `none` when it owns none.
- `KernelM.require` records the uniform `block.x = block.y = tile` assumption in
  the DSL body rather than constraining the reusable kernel's signature.
- `Kernel.shared` declares block-scoped allocations. `launch` creates the
  matching `SharedState`, and kernels retrieve buffers through total
  `SharedState.get` calls carrying `SharedRef` membership proofs.
- Barriers are currently explicit operations. A future indexed effect can add
  phase/uniformity information to `KernelM` and prove barrier convergence.

The launch surface is intentionally close to CUDA/HIP:

```lean
launch config outputBuffer outputLayout certificate
  (tiledMatmulKernel tile a b)
```

`launch` is an axiom here: this project specifies its dependent interface but
does not provide an implementation. A backend creates block/thread state,
constructs thread references and scheduled work items, stores returned values
through the output layout, and discharges uniform `KernelM.require` conditions.
