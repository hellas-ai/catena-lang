# Context-Based Parallelism

Catena's parallel operations describe work without making CUDA grids, blocks, or threads part of the generic parallel dialect. CUDA is the first implemented backend and provides the concrete example in this document.

This document describes the current implementation. Some types are written in friendlier pseudocode instead of their complete Hex spelling.

## Goals

- Keep the execution topology separate from the logical output being computed.
- Allow backends to define their own topology levels and dimensions.
- Use the same parallel operations for launching work and for cooperation
  between workers that are already running.
- Represent shared storage and its synchronization explicitly.
- Keep backend-specific coordinate calculations in the backend dialect.

## The execution model

The abstract model describes how a hierarchically organized worker population
executes logical work. It separates three questions:

1. How are physical workers organized?
2. Which logical work item is assigned to each worker?
3. How do workers cooperate through storage and synchronization?

A backend interprets this structure using its own execution mechanism. The
same abstract worker could become a GPU thread, a CPU pool member, a SIMD lane,
or one iteration of a sequential interpreter.

The useful vocabulary is:

- **Topology**: the complete organization of execution, such as a GPU grid of
  blocks.
- **Level**: one layer of that topology, such as grid or block.
- **Worker**: one physical execution instance.
- **Group**: workers that can cooperate using operations supported at that
  level, such as a CUDA thread block.
- **Worker index**: a worker's coordinate at the selected level.
- **Logical shape**: the shape of the problem or result.
- **Physical distribution**: the number and organization of available
  workers.
- **Logical distribution**: the assignment of logical work items to physical
  workers.
- **Task**: the computation run through a worker population.
- **Work item**: one logical unit of the task, such as one output cell.
- **Host context**: a handle used by the caller to allocate output storage and
  submit work.
- **Worker context**: the handle available while a task is running. It carries
  the current worker index and the storage layout available at that level.

The topology is not the logical shape of the data. The simplest distribution
assigns one logical item to every physical worker, but this is only one case.
More workers may require predication; fewer workers may require each worker to
process several items.

For a perfectly tiled `n x m` matrix result, CUDA uses this physical
organization:

| Population                   | Coordinates                            |         Count |
| ---------------------------- | -------------------------------------- | ------------: |
| grid workers                 | `(x, y)` in `(m, n)`                   |       `n * m` |
| workers in the current block | `(local-x, local-y)` in `(tile, tile)` | `tile * tile` |

The logical result uses `(row, column)` in `(n, m)`. CUDA X corresponds to the
column and CUDA Y to the row. Keeping the two descriptions separate makes
this mapping explicit.

### CUDA example

For CUDA, the abstract terms map to the current implementation like this:

- topology level: `gpu.grid.2d` or `gpu.block.2d`;
- worker: one GPU thread;
- grid task: the launched kernel;
- work item: one flat output-buffer element;
- host allocation: global device memory;
- block-worker allocation: dynamic shared memory;
- host await: device synchronization;
- block await or release: `__syncthreads()`.

A compact outline of perfectly tiled matrix multiplication is:

```text
layout = empty
layout = gpu.shared(tile * tile, layout) # B
layout = gpu.shared(tile * tile, layout) # A is allocated first

grid = gpu.grid.2d(m / tile, n / tile, tile, tile)
host = parallel.host(grid, layout)
(host, C) = parallel.host.allocate(host, n * m)

(host, pending-C) = parallel.materialize(host, C,
  \(grid-worker, output-index) ->
    (block-worker, available) =
        gpu.current-block(grid-worker, tile, tile)
    (block-worker, available, A-tile) =
        parallel.allocate(block-worker, available, tile * tile)
    (block-worker, available, B-tile) =
        parallel.allocate(block-worker, available, tile * tile)
    parallel.shared.finish(available)

    # Refill the same A and B slots for every K tile.
    tiled-cell-dot(block-worker, A-tile, B-tile, output-index))

C = parallel.await-one(host, pending-C)
```

The detailed task and its two synchronization points are shown below.

## Host and worker contexts

The generic parallel dialect has two execution contexts:

```text
host<level, workers, groups, storage>
worker<level, workers, groups, index, storage>
```

A `host` submits work to a population of workers. A `worker` represents one running member of such a population. Both carry:

- `level`: the topology level;
- `workers`: the shape or description of the workers at that level;
- `groups`: how those workers are grouped;
- `storage`: the storage layout available to them.

A worker additionally carries its topology-shaped `index`. For a two
dimensional level this can be a product such as `Ix n * Ix m`.

The parallel dialect does not prescribe a particular shape language or a fixed set of topology levels. A backend dialect constructs suitable values for them.

`parallel.host` builds only the host context:

```text
parallel.host(level, workers, groups, parameters, shared-layout)
    -> host<level, workers, groups, shared-layout>
```

It does not create a worker value. The backend creates workers when `parallel.materialize` starts executing the task.

The generic worker operations are:

```text
parallel.worker.index(worker)
    -> (worker, index)

parallel.worker(parent, child-level, child-workers, child-groups, child-index)
    -> (child-worker, available-storage)
```

`parallel.worker` preserves the parent's storage layout and creates its unique
initial storage capability. It is the generic operation used after a backend
has calculated the index at a child level.

## GPU topology

The GPU dialect currently defines two two-dimensional levels:

```text
gpu.grid.2d<grid-x, grid-y, block-x, block-y>
gpu.block.2d<block-x, block-y>
```

The constructors preserve both type-level dimensions and the runtime values needed by CUDA/HIP lowering:

```text
gpu.grid.2d(grid-x, grid-y, block-x, block-y)
    -> (level, workers, groups, launch-parameters)

gpu.block.2d(block-x, block-y)
    -> (level, workers, groups, dimensions)
```

For example, a perfectly tiled `n x m` result uses:

```text
grid-x  = m / tile-size
grid-y  = n / tile-size
block-x = tile-size
block-y = tile-size
```

This produces `n * m` grid workers, one for every output cell. CUDA's X
coordinate follows matrix columns and Y follows rows.

Inside a kernel, `gpu.current-block` maps the current grid worker to a worker whose index is relative to the current block:

```text
gpu.current-block(grid-worker, block-x, block-y)
    -> (block-worker, available-shared-storage)
```

Conceptually it computes:

```text
local-x = global-x % block-x
local-y = global-y % block-y
local-index = (local-x, local-y)
```

The coordinate calculation is GPU-specific, but the resulting child worker
and its initial storage capability are built together with the generic
`parallel.worker` operation.

## Shared storage

Shared memory is described before the host context is created. Its type is a list of allocation slots:

```text
shared.empty
shared.slot<length, element-type, rest>
```

Start with an empty layout and add as many GPU shared slots as required:

```text
empty = parallel.shared.empty()
one   = gpu.shared(n, empty)
two   = gpu.shared(m, one)
```

`gpu.shared` currently adds an `f32` slot. It prepends the slot, so the most recently added slot is allocated first:

```text
gpu.shared(n, shared<Rest>)
    -> shared<slot<n, f32, Rest>>
```

When source allocation order should be A followed by B, construct the layout in reverse:

```text
empty    = parallel.shared.empty()
b-layout = gpu.shared(b-count, empty)
layout   = gpu.shared(a-count, b-layout)
```

The runtime value of the layout is its required byte count. CUDA/HIP lowering adds `length * sizeof(float)` for every call to `gpu.shared` and passes the total as the dynamic shared-memory size of the kernel launch.

Within the kernel, constructing the block worker produces a linear capability
for the complete layout. `parallel.allocate` consumes its next slot:

```text
(worker, available<slot<n, f32, Rest>>, n)
    --parallel.allocate-->
(worker, available<Rest>, writable(buf cap.own n f32))
```

The length argument must agree with the length recorded in the slot type. Once all slots have been allocated, only this operation accepts the remaining capability:

```text
parallel.shared.finish(available<shared.empty>)
```

This makes it impossible to allocate more logical slots than the layout
declares. It does not prove that the requested byte count fits a particular GPU's hardware limit; the CUDA/HIP runtime still validates the launch.

Global output storage is separate. It is allocated through the host:

```text
parallel.host.allocate(host, n)
    -> (host, writable(buf cap.own n f32))
```

CUDA/HIP lowers this operation to device allocation. Block-worker allocation instead returns pointers at successive offsets in the kernel's dynamic shared arena.

## Materialization and synchronization

`parallel.materialize` fills an existing writable buffer:

```text
parallel.materialize(context, writable-buffer,
    task : (task-worker, Ix output-count) => f32)
  -> (context, pending-buffer)
```

The context determines how the task executes:

| Context      | CUDA/HIP lowering                                            |
| ------------ | ------------------------------------------------------------ |
| grid host    | launch a kernel                                              |
| block worker | let the current GPU thread produce one shared-buffer element |

The task receives the worker separately from its logical materialization
index. The current implementation intentionally keeps physical worker count and logical output count independent. It does not yet carry a general proof or policy explaining how different counts are distributed.

A writable buffer becomes `pending` after materialization. Synchronization
makes its contents available:

```text
parallel.await(context, pending-a, pending-b)
    -> (context, ready-a, ready-b)

parallel.await-one(context, pending)
    -> (context, ready)
```

For a block worker, `parallel.await` emits one block barrier for both buffers.
For a grid host, `parallel.await-one` waits for the launched kernel. Ready
buffers are currently represented as bare buffer values rather than with a
separate `readable` modifier.

Before a loop overwrites shared buffers, all workers must also have finished
reading their previous contents:

```text
parallel.release(block-worker) -> block-worker
```

On CUDA/HIP this is another block barrier. The synchronization is required for
correctness even though the current type does not enforce it.

### Shared allocation and loop-carried release

The shared arena is allocated once per block and split into the declared slots
before the K-tile loop. The same pointers are then reused on every iteration.

Each generation of a shared slot should follow this state cycle:

```text
             materialize              await (sync)
writable -----------------> pending -----------------> readable
    ^                                                      |
    +---------------- collective release (sync) -----------+
```

`await` supplies the producer-to-consumer barrier: every cooperative write is
visible before any worker reads the tile. `release` supplies the other side:
every read finishes before a faster worker overwrites the slot in the next
iteration.

The implementation emits both barriers, but the types enforce only the first
half. Ready buffers are bare values, and `parallel.release` currently consumes
only the worker:

```text
parallel.release(block-worker) -> block-worker
```

Ideally the loop would carry the buffer states explicitly. A state-carrying
fold could use:

```text
S = accumulator
  * writable A-tile
  * writable B-tile

fold : S * (S * Ix n => S) * n -> S
```

Then the loop body could return `S` only after converting both readable tiles
back to writable tiles through a collective release. The current `reduce`
does not have this shape: it takes an independent producer `Ix n => A` and
cannot thread the tile handles through the loop backedge. Consequently, the
tiled-matmul Hex contains an explicit `parallel.release`, with a TODO noting
that omitting it would introduce a race even though the program would still
type-check.

The final iteration does not need a release if its shared slots will not be
reused. A future state-carrying combinator may therefore need to distinguish
the loop backedge state from the final result.

`parallel.materializec` is the closure-converted form used later in the
compiler pipeline. It replaces the source task closure with its explicit
environment and function pointer; it does not change the parallel semantics.

## Tiled matrix multiplication

The current example computes perfectly tiled, row-major matrix
multiplication:

```text
A : (Ix n * Ix k) => f32
B : (Ix k * Ix m) => f32
C : buf cap.own (n * m) f32
```

It assumes that `n`, `k`, and `m` are divisible by `tile-size`. Rectangular
matrices are supported; `n`, `k`, and `m` need not be equal.

A friendly version of the host setup is:

```text
tile-count = tile-size * tile-size

empty         = parallel.shared.empty()
b-only-layout = gpu.shared(tile-count, empty)
shared-layout = gpu.shared(tile-count, b-only-layout)

grid = gpu.grid.2d(
    m / tile-size,
    n / tile-size,
    tile-size,
    tile-size)

host = parallel.host(grid, shared-layout)
(host, C-writable) = parallel.host.allocate(host, n * m)

(host, C-pending) = parallel.materialize(
    host,
    C-writable,
    tiled-matmul-cell-kernel)

C = parallel.await-one(host, C-pending)
```

The backend creates a grid worker for every output element. The kernel turns
the flat materialization index into `(output-row, output-col)`, then scopes its
physical worker to the current block:

```text
(block-worker, available) =
    gpu.current-block(grid-worker, tile-size, tile-size)

(block-worker, available, A-tile) =
    parallel.allocate(block-worker, available, tile-count)

(block-worker, available, B-tile) =
    parallel.allocate(block-worker, available, tile-count)

parallel.shared.finish(available)
```

The two buffers are allocated once and reused for every K tile:

```text
acc = 0

for k-tile in 0 .. k / tile-size:
    pending-A = parallel.materialize(block-worker, A-tile,
        \local-index ->
            A[output-row,
              k-tile * tile-size + local-index.col])

    pending-B = parallel.materialize(block-worker, B-tile,
        \local-index ->
            B[k-tile * tile-size + local-index.row,
              output-col])

    (block-worker, A-ready, B-ready) =
        parallel.await(block-worker, pending-A, pending-B)

    for inner in 0 .. tile-size:
        acc += A-ready[local-row, inner]
             * B-ready[inner, local-col]

    block-worker = parallel.release(block-worker)

return acc
```

The first barrier makes both cooperative loads visible. The release barrier
prevents a faster worker from starting the next iteration and overwriting a
tile while another worker is still reading it.

CUDA lowering declares one dynamic shared arena per block, gives every worker
in that block the same base pointer, and assigns A and B distinct offsets. The
grid launch receives `2 * tile-size * tile-size * sizeof(float)` shared bytes.

## Non-perfect tiling

The current tiled-matmul implementation assumes perfect tiling. Non-perfect
tiling explains why physical worker indices and logical output indices must
remain separate.

For example:

```text
A: 70 x 37
B: 37 x 50
C: 70 x 50
tile: 16 x 16
```

The result needs `ceil(70 / 16) x ceil(50 / 16) = 5 x 4` blocks. Launching
`16 x 16` workers per block physically covers `80 x 64` output positions,
although only `70 x 50` positions belong to C.

```text
physical (row, col) = (69, 49) -> valid logical output
physical (row, col) = (70, 49) -> outside C
physical (row, col) = (64, 50) -> outside C
```

The out-of-range workers cannot simply return before the tiled computation.
Other workers in the same block execute `parallel.await` and
`parallel.release`, and CUDA barriers require every worker in that block to
reach the same collective control flow.

A future predicated distribution should therefore give every worker its
physical index and optionally a logical output index:

```text
task input = (
    worker,
    physical-index : Ix physical-coverage,
    logical-index  : Option<Ix logical-shape>)
```

All workers would execute the cooperative loads and barriers. Only
`Some(logical-index)` would commit a result to C:

```text
for each K tile:
    A-tile[local] =
        if A-index is in bounds then A[A-index] else 0

    B-tile[local] =
        if B-index is in bounds then B[B-index] else 0

    await both tiles
    value += dot(A-tile row, B-tile column)
    release both tiles

match logical-index:
    Some(index) -> write C[index] = value
    None        -> write nothing
```

The zeroes are padding used by the physical computation; they are not logical
matrix elements. Refining a physical index to a logical `Ix` before indexing a
matrix would make the bounds check explicit in the types.

Other distributions need different evidence. A grid-stride distribution can
use fewer workers than logical items and assign several indices to each
worker:

```text
4 workers covering 10 items

worker 0 -> 0, 4, 8
worker 1 -> 1, 5, 9
worker 2 -> 2, 6
worker 3 -> 3, 7
```

Conceptually, a distribution relates a physical worker shape `m` to a logical
shape `n`:

```text
distribution<m, n>

perfect    : distribution<n, n>
predicated : n <= m       -> distribution<m, n>
grid-stride: size(m) > 0  -> distribution<m, n>
```

The current `parallel.materialize` keeps worker and output counts independent,
so its type leaves room for this distinction, but it does not yet accept or
construct a distribution value. CUDA lowering currently implements only the
perfect one-worker/one-output case.

The governing rule for non-perfect execution is:

> A worker may be inactive with respect to logical output, but it must remain
> active with respect to collective control flow.

## Relationship to existing Catena concepts

Catena already represents a logical tensor or multidimensional view as an
indexed closure and materializes it into a flat, dependently sized buffer. The
parallel operations retain that model:

- the task closure still describes how to compute one indexed value;
- the host or worker context says who evaluates it;
- the writable and pending states say when the destination may be observed;
- shaped views remain separate from the underlying linear buffer.

The current parallel primitives specialize this path to `f32`; making the
element type generic does not require changing the execution model.

## Other execution models

The generic parallel types are not limited to GPUs. Other backends could map
the same concepts differently:

- A CPU backend could map a group to a worker pool and a worker to one pool
  member.
- A SIMD backend could map a group to a vector and a worker to one lane.
- HIP, Metal, and Vulkan could define their native grid, workgroup, and
  subgroup levels.
- A sequential backend could use a one-worker topology to test semantics
  independently of parallel code generation.

Each backend defines its topology constructors and how host allocation,
materialization, storage, and synchronization are lowered. It need not support
every level or capability. For example, a program requiring group-local
storage should be rejected for a context whose backend cannot provide it.

## Correctness of GPU kernels

The current types encode part of the required ordering, but a complete
correctness argument also needs properties of the work distribution and the
task. The intended theorem is:

```text
distribution is finite
distribution covers every logical index
distribution produces each logical index uniquely
task is total
task effects are deterministic and race-free
task barriers are convergent
--------------------------------------------------
materialize + await is total and deterministic
```

### Distribution correctness

A valid distribution must prove that every logical output is produced exactly
once. Standard distributions could construct these proofs automatically;
custom distributions would need to provide equivalent evidence.

Under a perfect distribution, the current CUDA lowering uses the worker's
flat index as the output index. Therefore the launch must contain exactly the
output buffer's number of workers. This is true for the perfectly tiled
matmul: `(m / tile) * (n / tile)` blocks times `tile * tile` workers equals
`n * m` outputs.

### Race freedom

A conservative rule divides shared-memory execution into phases separated by
block barriers. For each shared location within one phase, one of these must
hold:

- all workers only read it;
- exactly one worker writes it and no conflicting access occurs; or
- every conflicting access uses an atomic operation with the correct scope.

The tiled matmul follows this rule:

```text
cooperative unique writes
    -> await barrier
shared reads
    -> release barrier
next cooperative writes
```

`pending` and `await` enforce only producer-to-consumer ordering:

```text
writes -> await -> reads
```

They do not ensure that every read completes before the next writes reuse the
same slot:

```text
reads -> release -> next writes
```

That second edge is currently present in the tiled-matmul program and emitted
as a CUDA/HIP block barrier, but—as explained above—the `reduce` shape prevents
the shared-buffer states from being carried through the loop. The type system
therefore cannot yet reject a program that omits `parallel.release`.

## Current boundaries

The implementation deliberately remains small:

- only the GPU grid and block levels have backend lowering;
- `gpu.shared` and the allocation operations currently support `f32` buffers;
- tiled matmul requires perfect tiling and has no boundary predicates;
- hardware launch limits are checked by CUDA/HIP rather than represented in
  the Hex types;
- `parallel.release` does not yet encode which readers must finish;
- general work-distribution policies are not yet represented.

The generic `level`, `workers`, `groups`, `index`, and `storage` metavariables
leave room for other topologies and shapes without adding them prematurely.
