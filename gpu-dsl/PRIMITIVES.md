# Language-independent GPU launch specification

This note extracts the launch model from `GpuDsl` without depending on Lean,
Catena, CUDA, HIP, or another implementation language. It specifies only the
types and contracts needed to launch a simple matrix-multiplication kernel.

Shared memory, barriers, synchronization traces, and tiling are intentionally
out of scope. Values, global-buffer reads, arithmetic, and layouts are assumed
to be supplied by the surrounding language.

The signatures use `⊗` for product and `⊸` for a linear map, following the
notation used by the exploratory Catena feature notes.

## Launch geometry

```text
Dim3 := {
  x : U64,
  y : U64,
  z : U64
}

LaunchConfig := {
  grid  : Dim3,
  block : Dim3
}

Coord(D : Dim3) := {
  x : Ix(D.x),
  y : Ix(D.y),
  z : Ix(D.z)
}

ThreadId(C : LaunchConfig) := {
  block-index  : Coord(C.grid),
  thread-index : Coord(C.block)
}
```

`Coord(D)` is bounded by construction. A `ThreadId(C)` therefore always names
a physical thread that exists in launch configuration `C`.

## Output schedule

```text
Schedule(C : LaunchConfig, S : Space) := {
  owner : Ix(S) ⊸ ThreadId(C)
}
```

`S` is the logical output space. For matmul with an `M × N` result,
`S = Mat(M, N)`.

Because `owner` is a total function, every logical output has exactly one
physical owner. A physical thread may own zero, one, or several outputs.

## Kernel

```text
Kernel(S : Space, T : Type) := {
  Requirements : LaunchConfig ⊸ Proposition,

  body : ∀ C : LaunchConfig.
    (|- Requirements(C))
    ⊗ (thread : ThreadId(C))
    ⊗ Ix(S)
    ⊸ Val(T)
}
```

The body receives the current physical thread and the logical output assigned
to that invocation. It returns one staged value for that output; it does not
choose the destination index.

`Requirements` contains uniform facts required by every invocation, such as a
particular block shape. A kernel is polymorphic over launch configurations and
may only use a configuration after receiving proof that its requirements hold.

`Kernel` does not contain a schedule. The same kernel can be launched with any
schedule satisfying the launch contract.

## Launch primitive

```text
gpu.launch[S, N, T] :
  (config : LaunchConfig)
  ⊗ Bufferᵍˡᵒᵇᵃˡ(N, T)
  ⊗ Layout(S, N)
  ⊗ (schedule : Schedule(config, S))
  ⊗ (kernel : Kernel(S, T))
  ⊗ (|- kernel.Requirements(config))
  ⊸ Launch(N, T)
```

`Bufferᵍˡᵒᵇᵃˡ(N, T)` is an opaque global output buffer, and
`Layout(S, N)` maps logical output indices to distinct physical buffer cells.
Their allocation and ownership model belongs to the surrounding language.
`Launch(N, T)` is an opaque handle or effect witnessing that the launch was
submitted.

The implementation of `gpu.launch` must:

1. Enumerate every logical index `i` in `S` exactly once.
2. Set the current physical thread to `schedule.owner(i)`.
3. Invoke `kernel.body(requirements-proof, thread, i)`.
4. Store the result at `layout(i)`.
5. Respect the surrounding language's buffer lifetime and execution-ordering
   rules.

A backend may invoke a thread's body repeatedly when that thread owns several
outputs; threads with no outputs need not be invoked. The order of different
logical outputs is unspecified. Since each output has one owner and the output
layout is injective, scheduling cannot create write races.

## Untiled matmul instance

Let the inputs be `A : M × K` and `B : K × N`, and let the logical output
space be `S = Mat(M, N)`. Choose a positive block shape, for example:

```text
config.grid  = (ceil-div(N, tile), ceil-div(M, tile), 1)
config.block = (tile, tile, 1)
```

The schedule maps output `(row, column)` to:

```text
block-index  = (column / tile, row / tile, 0)
thread-index = (column % tile, row % tile, 0)
```

The kernel requirement is:

```text
Requirements(config) :=
  config.block.x = tile
  ∧ config.block.y = tile
  ∧ config.block.z = 1
```

For output `(row, column)`, the body returns the staged scalar expression:

```text
A[row, 0] × B[0, column]
  + A[row, 1] × B[1, column]
  + ...
  + A[row, K - 1] × B[K - 1, column]
```

Terms are accumulated in ascending `k` order; the empty sum is zero. Input
access and construction of the scalar expression use facilities from the
surrounding language rather than additional launch primitives.

This is sufficient to run a global-memory-only matmul. Shared-memory tiling can
later extend `Kernel` with per-block state, no-work invocations, and
synchronization while leaving `Schedule` and the external `gpu.launch` shape
largely intact.
