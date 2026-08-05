# Context-Based Parallelism

This document sketches a parallel execution model for Catena. CUDA is the
first target, but CUDA concepts such as grids, blocks, warps, and threads are
not language primitives. They are one possible interpretation of a more
general _execution context_.

The central idea is:

> A context describes a hierarchical execution topology and determines how a
> logical index space is distributed over the workers in that topology.

Programs express parallel work in terms of scheduling discipline, work allocation, and synchronization. A backend decides how those concepts map to GPU launches, SIMD lanes, CPU workers, or another execution model.

This is an initial design. The notation and types below are schematic rather
than declarations in the current Catena syntax.

## Goals

- Represent CUDA-style launch and cooperative execution without baking CUDA
  names into the core language.
- Keep the logical index space separate from the physical execution topology.
- Make values produced by cooperative work unavailable until the required
  synchronization has occurred.
- Permit nested scopes such as grid > block > warp > thread.
- Leave scheduling, memory placement, and synchronization details available to
  backend-specific lowering.

Non-goals for the first version include dynamic task graphs, arbitrary
cross-device communication, and scheduling optimization techniques.

## The execution model

The abstract execution model describes how a hierarchically organized worker
population executes logical work. In particular, it specifies:

1. how workers are organized in a physical topology;
2. how logical work is assigned to those workers; and
3. how workers cooperate through shared storage and synchronization.

A backend interprets this structure using its own execution mechanism. It may
run workers as GPU threads, CPU executions, SIMD lanes, or sequential
interpreter iterations without changing the logical work performed by the
program.

> **Terminology:** A _worker_ is an abstract execution instance that runs a
> unit of work. In CUDA examples, _thread_ names the leaf element of the CUDA
> hierarchy `grid > block > thread`. When discussing its execution rather than
> its position in that hierarchy, this document calls it a _GPU thread_ or a
> _worker_.

The model uses the following vocabulary:

- **Topology**: the complete hierarchy, such as `grid > block > thread`.
- **Level**: one layer of that hierarchy, such as grid, block, or thread.
- **Workers**: the physical leaf executions in the topology. A sequential
  terminal describes work assigned to a worker rather than more workers.
- **Worker path**: the canonical hierarchical location of a worker, with one
  index for each physical worker level of the topology. Sequential terminals
  do not add a path component.
- **Scope**: the topology level used as the coordinate origin for a worker.
- **Worker shape**: the worker coordinate space visible from a scope.
- **Worker index**: the same worker's coordinate within that shape.
- **Logical shape**: the shape of the problem or result being computed.
- **Physical distribution**: how the available workers are organized in the
  execution topology.
- **Logical distribution**: how the work items of a computation are assigned
  to those workers.
- **Task**: the computation executed by the workers.
- **Work item**: one logical unit of that task, such as one element of its
  result.
- **Configuration**: the topology, distributions, and capabilities
  used to execute tasks on a worker population.
- **Context**: a handle to a worker population, passed to the `schedule`
  callback and to running tasks. A task context also identifies the current
  worker and its assigned work.

Consider an `n × m` logical result, with `n` and `m` divisible by
`tile_size`. A CUDA grid can contain `(n / tile_size, m / tile_size)` blocks,
where every block contains `(tile_size, tile_size)` threads:

| Worker population |                         Direct children |             Worker shape |            Worker count |
| ----------------- | --------------------------------------: | -----------------------: | ----------------------: |
| grid              | `(n / tile_size, m / tile_size)` blocks |                 `(n, m)` |                 `n * m` |
| current block     |        `(tile_size, tile_size)` threads | `(tile_size, tile_size)` | `tile_size * tile_size` |
| current thread    |                                    none |                singleton |                     `1` |

`size` gives the cardinality of a worker shape. For a two-dimensional shape,
`size((n, m)) = n * m`; therefore, the block worker shape
`(tile_size, tile_size)` has size `tile_size * tile_size`, while the grid
worker shape `(n, m)` has size `n * m`.

Parallel execution therefore has a physical distribution and a logical
distribution. The physical distribution describes the workers available to
run the computation. In CUDA, it is determined by the launch geometry: the
grid organizes GPU threads into blocks. The logical distribution describes
the work items to compute, such as the elements of an output array, and how
those work items are assigned to the available workers.

The topology is not the logical shape of the data. The simplest and most
common assignment gives one work item to each worker, so the physical worker
shape and logical shape coincide. They need not coincide: `p` workers may
compute `q` result elements by handling several work items each when `q > p`,
while `p > q` leaves some workers without an output item. Keeping the two
distributions separate allows the same computation to use different launch
sizes or backends.

A _configuration_ describes how a worker population should be created. It
combines the physical topology, the logical distribution, and the requested
capabilities, such as synchronization or shared memory. `schedule` consumes
the configuration and passes a _context_ to its callback. That context is the
handle through which work is submitted and synchronized.

When a task runs, it also receives a context. This task context additionally
describes the current worker and its assigned work. For example, a GPU worker
can obtain a context for its current block when cooperating with the other
workers in that block.

The following operations connect the topology, storage, workers, and logical
work.

`schedule` establishes the physical worker organization. Its body runs once
in the caller and receives a context for that worker population.

`allocate` creates a linear buffer owned by an execution context and returns a
storage context. The owner determines the storage scope, its participating
workers, and its synchronization domain.

`materialize` runs a _task_ to fill a storage context. The task is the unit of
work executed by the workers of its owning execution context. Each invocation
receives a worker context and performs the logical work assigned to that
worker. A worker may be assigned one work item, several work items, or no
output item. `materialize` returns the storage as pending because the work may
still be executing or may not yet be visible to all participating workers.

`sync` completes pending work associated with a context and makes its results
safe to observe. For a grid context it may wait for a kernel. For a block
context it is a collective operation reached by all workers in
the block, for example after they have cooperatively filled shared memory.

Together, `schedule` chooses the worker organization, `materialize` states the
task and its logical result, and `sync` states when that result may be used.

### CUDA example

For CUDA, the abstract concepts above can be interpreted as follows:

- topology = `grid > block > thread`
- worker = a GPU thread executing the task
- task = the kernel
- work item = one output cell, such as `C[i, j]` in matrix multiplication

For a perfectly tiled multiplication of `A : n × k` and `B : k × m`, assume
that `n`, `k`, and `m` are divisible by `tile_size`. Scheduling establishes
`(n / tile_size, m / tile_size)` blocks of `(tile_size, tile_size)` workers.
The output and the two block-local tiles are allocated explicitly as linear
buffers, then `materialize` fills them using their owning execution contexts:

```text
A : Ix (n, k) => f32
B : Ix (k, m) => f32

plan = blocks(n / tile_size, m / tile_size)
     > threads(tile_size, tile_size)
     > sequential(1)

schedule plan, \launch_ctx ->
    C_storage = allocate(launch_ctx, n * m, f32)

    pending_C = materialize(launch_ctx, C_storage,
      \(ctx, output_index: Ix (n * m)) ->
        block = ctx.block
        A_tile = allocate(block, tile_size * tile_size, f32)
        B_tile = allocate(block, tile_size * tile_size, f32)

        # block.shape = (tile_size, tile_size)
        # size(block) = tile_size * tile_size
        #             = size(A_tile) = size(B_tile)
        # The block is two-dimensional; both tile buffers are linear.

        acc = 0

        for k_tile: Ix (k / tile_size)
        invariant
            A_tile : writable block (
                buf cap.own (tile_size * tile_size) f32)
            B_tile : writable block (
                buf cap.own (tile_size * tile_size) f32)
        do
            pending_A = materialize(block, A_tile,
              \tile_index: Ix (tile_size * tile_size) ->
                tile_row = tile_index / tile_size
                tile_col = tile_index % tile_size
                (block_row, _) = block.path.block
                A[(block_row * tile_size + tile_row,
                   k_tile * tile_size + tile_col)])
            pending_B = materialize(block, B_tile,
              \tile_index: Ix (tile_size * tile_size) ->
                tile_row = tile_index / tile_size
                tile_col = tile_index % tile_size
                (_, block_col) = block.path.block
                B[(k_tile * tile_size + tile_row,
                   block_col * tile_size + tile_col)])

            # pending_A, pending_B : pending block (
            #     buf cap.own (tile_size * tile_size) f32)

            [A_ready, B_ready] = sync(
                block, [pending_A, pending_B])

            # A_ready, B_ready : readable block (
            #     buf cap.own (tile_size * tile_size) f32)

            acc = accumulate(
                acc, A_ready, B_ready, block.index)

            [A_tile, B_tile] = sync(
                block, [A_ready, B_ready])

            # A_tile, B_tile : writable block (
            #     buf cap.own (tile_size * tile_size) f32)

        return acc)

    C = sync(launch_ctx, pending_C)
```

Passing `launch_ctx` to `materialize` dispatches the task that fills
`C_storage` across the grid. Passing `block` performs the materializations of
`A_tile` and `B_tile` collectively and locally among the already-running block
workers. All three buffers and their callback indices are linear; a callback
may construct a shaped view when it needs multidimensional coordinates. The
tiles remain allocated across the loop: `sync` makes their cells readable,
while the second `sync` proves that all workers have finished reading and
returns the same buffers as writable for the next iteration.

## Context

A configuration describes an execution before it is scheduled. `schedule`
consumes it and passes a _context_ to its callback. `materialize` also passes
a context to every running task. It is the same context type in both places;
worker-specific information is available while the context is used in a
task:

```text
context c p l = {
    config : configuration p,
    path   : worker-path p,
    shape  : worker-shape p l,
    index  : Ix (worker-shape p l),
}
```

The type parameters have the following roles:

- `c` identifies the execution whose work can be materialized or synchronized;
- `p` is the complete execution topology, including any sequential terminal;
- `l` is the physical worker level currently selected within that topology.

The context does not copy or create workers. `path` is the current worker's
canonical location in the physical part of the topology, with one component
per physical worker level. Scoping changes `l`, then derives a new `shape` and
`index` for the same path. A sequential terminal describes work assigned to
that worker and therefore does not extend its path. The `schedule` callback
uses its context to submit work; `path` and `index` are queried while a task is
running.

When the selected level is clear from the surrounding type, later signatures
may omit `l` and use the shorter spelling `context c p`.

### Scoping a context

Scoping narrows a context from one topology level to a level below it:

```text
scope : (ctx : context c p parent)
     -> (child : descendant-level p parent)
     -> context c p child
```

The `descendant-level p parent` constraint means that the requested level must
exist below the current level in `p`. Dot notation is shorthand for this
operation:

```text
ctx.block  = scope(ctx, block)
ctx.warp   = scope(ctx, warp)
ctx.thread = scope(ctx, thread)
```

Scoping does not move execution to another worker. It changes the worker
space relative to which `shape`, `index`, `materialize`, and `sync` are
interpreted:

```text
scope(ctx, child).path  = ctx.path
scope(ctx, child).shape = worker-shape p child
scope(ctx, child).index = flatten-from(child, p, ctx.path)
```

A block context therefore expresses the current worker in block-local
coordinates, while a warp context expresses that same worker in warp-local
coordinates.

### CUDA topology: `grid > block > thread`

Consider a `4 × 4` grid of blocks, each containing `16 × 16` CUDA threads:

```text
p = blocks(4, 4) > threads(16, 16) > 1

ctx : context kernel p grid

ctx.path.block  : Ix (4, 4)
ctx.path.thread : Ix (16, 16)
ctx.shape       : Shape 2 = (64, 64)
ctx.index       : Ix (64, 64)

block_ctx = ctx.block
# block_ctx : context kernel p block

block_ctx.path  = ctx.path
block_ctx.shape : Shape 2 = (16, 16)
block_ctx.index : Ix (16, 16) = ctx.path.thread

thread_ctx = ctx.thread
# thread_ctx : context kernel p thread

thread_ctx.path  = ctx.path
thread_ctx.shape : Shape 1 = (1)
thread_ctx.index : Ix (1) = (0)
```

At grid level, `index` combines `path.block` (`blockIdx`) and `path.thread`
(`threadIdx`). After scoping to `block`, `index` is just `path.thread`. In the
topology type, `blocks(4, 4)` describes the grid,
`threads(16, 16)` describes each block, and `1` describes the singleton thread
scope. This topology has no `warp` level, so `ctx.warp` is not well-typed.

### CUDA topology: `grid > block > warp > thread`

A backend may expose CUDA warps explicitly. The same 256 workers per block can
then be organized as eight warps of 32 threads:

```text
p = blocks(4, 4) > warps(8) > threads(32) > 1

ctx : context kernel p grid

ctx.path.block  : Ix (4, 4)
ctx.path.warp   : Ix (8)
ctx.path.thread : Ix (32)

block_ctx = ctx.block
# block_ctx : context kernel p block

block_ctx.path  = ctx.path
block_ctx.shape : Shape 2 = (8, 32)
block_ctx.index : Ix (8, 32)

warp_ctx = ctx.warp
# warp_ctx : context kernel p warp

warp_ctx.path  = ctx.path
warp_ctx.shape : Shape 1 = (32)
warp_ctx.index : Ix (32) = ctx.path.thread
```

Here `block_ctx.index` is `(path.warp, path.thread)`. Scoping to the current
warp removes the warp component and leaves the thread coordinate.
In this topology type, `blocks(4, 4)` describes the grid, `warps(8)` describes
each block, `threads(32)` describes each warp, and `1` describes the thread
scope. The explicit-warp topology changes the physical coordinate types, but
not the number of GPU workers in a block. A logical distribution may still
map these workers to a differently shaped work-item space.

## Work Item Distribution

`materialize(ctx, n, task)` distributes the logical work items `Ix n` over
the physical workers represented by `ctx`. Its source is always a context. In
this section, assume a perfect tiling: every worker executes exactly one work
item, and every work item is executed by exactly one worker.

The context path and the task index have different meanings. `ctx.path`
identifies the worker on which the task runs, whereas the index passed to the
task identifies the work item assigned by `materialize`. Under perfect
one-to-one distribution, the work-item index can be derived from the scoped
worker index, so a callback may omit it when it is not otherwise useful. It is
not redundant in general: with a grid-stride distribution the same worker path
executes several work-item indices, and with a predicated distribution a
physical state index may have no corresponding logical work item.

Under this assumption, the worker shape carried by the context must match the
work-item shape passed to `materialize`. We can express that directly in a
schematic dependent type:

```text
materialize : context (shape n)
            × (n : Shape k)
            × task (Ix n) t
           -> pending (buf (size n) t)
```

The repeated `n` is the compatibility check. A context with shape `(16, 16)`
can materialize `Ix (16, 16)` work items. It cannot
materialize `Ix (32, 16)` work items under perfect tiling.

### CUDA launch

A CUDA grid with `4 × 4` blocks and `16 × 16` GPU threads per block contains
a `(64, 64)` distribution of workers. A matrix-multiplication kernel over a
`64 × 64` result has one work item for every cell `C[i, j]`:

```text
schedule blocks(4, 4) > threads(16, 16) > 1, \launch_ctx ->
    # launch_ctx : context (shape (64, 64))

    pending_C = materialize(launch_ctx, (64, 64),
      \(ctx, cell) ->
        # ctx    : context (shape (64, 64))
        # cell   : Ix (64, 64)
        compute_C(cell))
```

The context has shape `(64, 64)`, and `materialize` requires work shaped
`(64, 64)`. The dimensions unify, so the distribution is valid. With
the same context, `materialize(launch_ctx, (70, 50), ...)` would be rejected
under perfect tiling because the worker and work-item shapes differ.

### CUDA cooperative loading

Inside the kernel, the block context represents only the
`16 × 16` workers in the current CUDA block. A cooperative load of a
`16 × 16` tile therefore has compatible worker and work-item shapes:

```text
block = ctx.block
# block : context (shape (16, 16))

pending_tile = materialize(block, (16, 16),
  \tile_index ->
    # tile_index : Ix (16, 16)
    load_tile(tile_index))
```

The relevant context is `block`, not the context for the complete grid. Its
type says that `materialize` has a `(16, 16)` worker space, while the
second argument requires `Ix (16, 16)` tile work items. Those dimensions
unify, so each block worker loads one tile element. A `(32, 16)` tile would
not be compatible with this block context under perfect tiling.

> **Open problem:** The current semantics make every `materialize` allocate a
> fresh buffer. Repeated cooperative loads would therefore appear to allocate
> new shared memory on every iteration. CUDA implementations instead allocate
> block-local storage once and refill it. The model still needs a way to
> express or safely lower this reuse, including the synchronization required
> before overwriting values that other workers may still be reading.

Possibile solution: primitive `allocate` for `block` context.

```
tile = allocate(block, tile_size * tile_size, element_type)

for k_tile:
    pending = materialize(tile, (tile_size, tile_size), load_tile)
    ready = sync(block, pending)
```

To make it uniform, materialize takes **always** a `storage-context` that could be a temporary buffer, a shared memory or anything else.

```
allocate
    : execution-context c p l
    × (length : u64)
    × Type
   -> storage-context c p l length t

materialize
    : storage-context c p l length t
    × (n : Shape k)
    × (execution-context c p l × Ix n => t)
   -> pending c (buf-view n t)

require length = size n
```

```
load_A = \(ctx, (row, col): Ix tile_shape) ->
    A[source_index(ctx, row, col)]
```

has meaning

```
for each index i assigned to a worker:
    value = load_A(worker_context, i)
    A_tile[i] = value <=== assign value to shared memory
```

```
materialize(launch_ctx, shape, task)
```

is syntactic sugar for:

```
storage = allocate(launch_ctx, size shape, result_type)
materialize(storage, shape, task)
```

#### Shared-buffer state and loop-carried synchronization

We want the type system to require collective synchronization before shared storage is overwritten while peer workers may still be reading it. For example, in the tiled matmul example, the second sync prevents faster workers to overwrite shared memory while others are still reading, in the native implementation the type system doesn't force the second sync that can be omitted introducing a race condition.

The allocation remains alive across the iterations that use it, while its
handle changes state as each generation of its contents is produced and
consumed:

```text
             materialize              sync
writable ─────────────────→ pending ─────────→ readable
    ↑                                             │
    └──────────── collective release ─────────────┘
```

The handles are affine: they cannot be duplicated, but a readable handle may
be discarded when its storage will not be reused. The second synchronization
is therefore required on a loop backedge, but not after the final use of the
buffer.

The current `reduce` takes an independent producer `Ix n => A`, so it cannot
pass a buffer handle from one iteration to the next. A state-carrying fold
could instead have a type schematically like:

```text
fold : S
     ● (S * Ix n => S)
     ● (n : u64)
    -> S

S = accumulator
  * writable A_tile
  * writable B_tile
```

The fold body consumes `S` at the start of an iteration. Taking the backedge
requires it to return another `S`. Because `S` contains writable tile handles,
the body can construct that result only by collectively releasing the readable
handles produced during the iteration. A final form of the combinator should
distinguish the backedge result from the final result, allowing the last
iteration to discard its readable handles instead of performing an unnecessary
release barrier.

### Note: non-perfect tiling

For non-perfect tiling, the worker shape and work-item shape must be kept
separate. Let `m` be the shape of the workers represented by the context and
`n` the shape of the logical work:

```text
materialize : context (workers m)
            × (n : Shape k)
            × distribution m n
            × task ...
           -> pending (buf (size n) t)
```

The `distribution m n` term describes why workers shaped `m` can cover work
items shaped `n`. It is usually determined by the context and omitted at the
call site.

For a predicated distribution, the shapes have the same rank and the logical
shape fits component-wise within the worker shape:

```text
predicated : (n <= m) -> distribution m n

context (shape (8))       -> Ix (6)       # 6 <= 8
context (shape (16, 16))  -> Ix (13, 15)  # 13 <= 16 and 15 <= 16
```

For a grid-stride distribution, `n` may instead be larger than `m`. Each
worker executes several work items until the complete logical shape is
covered:

```text
grid-stride : (size m > 0) -> distribution m n

context (shape (4)) -> Ix (10)
worker 0 -> (0), (4), (8)
worker 1 -> (1), (5), (9)
worker 2 -> (2), (6)
worker 3 -> (3), (7)
```

### Sequential work below a worker

A topology may end in `sequential(q)` instead of `1`:

```text
blocks(bx, by) > threads(tx, ty) > sequential(q)
```

`q` is the number of iterations, so their indices have type `Ix (q)` and
range from `(0)` through `(q - 1)`. The terminal does not create `q` more
physical workers. It says that each GPU thread can execute `q` work items
sequentially. Consequently, it does not contribute to physical worker shapes
or paths:

```text
grid_ctx.shape   = (bx * tx, by * ty)
block_ctx.shape  = (tx, ty)
thread_ctx.shape = (1)

thread_ctx.path = (block, thread)
```

Instead, `sequential(q)` supplies a distribution from the singleton thread
worker space to the iteration space:

```text
sequential : (q : u64) -> distribution (1) (q)
```

This makes the following materialization compatible even though its work-item
shape is not the physical shape of `thread_ctx`:

```text
thread_ctx : context c p thread
where p = ... > thread > sequential(q)

materialize(thread_ctx, (q),
  \(iteration_ctx, iteration : Ix (q)) -> task(iteration))
```

Every invocation has the same physical worker path as `thread_ctx`; the task
index distinguishes the `q` work items assigned to that worker. A CUDA backend
lowers the materialization to a loop. The iteration count is exact rather than
an upper bound. This is particularly important when the task contains
block-wide synchronization: every thread must traverse the same iteration
indices in the same order.

## Locating workers in a topology

Consider the topology used by a perfectly tiled output:

```text
blocks(n / tile_size, m / tile_size)
> threads(tile_size, tile_size)
> 1
```

Inside a task, the worker has one canonical path:

```text
ctx.path.block  : Ix (n / tile_size, m / tile_size)
ctx.path.thread : Ix (tile_size, tile_size)
```

Let `ctx.path.block = (block_row, block_col)` and
`ctx.path.thread = (thread_row, thread_col)`. At grid scope, the worker shape
and index are:

```text
ctx.shape : Shape 2 = (n, m)
ctx.index : Ix (n, m)

ctx.index = (block_row * tile_size + thread_row,
             block_col * tile_size + thread_col)
```

Scoping to the current block keeps the path but changes the coordinate system:

```text
block_ctx = ctx.block

block_ctx.path  = ctx.path
block_ctx.shape : Shape 2 = (tile_size, tile_size)
block_ctx.index : Ix (tile_size, tile_size) = ctx.path.thread
```

Thus `ctx.path.block` locates the block in the grid, while `block_ctx.index`
locates the same worker within that block. The topology implementation owns
the multiplication used to form `ctx.index`; task code uses the typed path and
scoped indices directly.

### Perfectly tiled matrix multiplication

Suppose `A` is an `n × k` matrix and `B` is a `k × m` matrix. Their product
`C` is an `n × m` matrix. Use a `tile_size × tile_size` tile and assume that
`n`, `k`, and `m` are all divisible by `tile_size`. The output schedule is:

```text
output work items: (n, m)
physical blocks:  (n / tile_size, m / tile_size)
threads/block:    (tile_size, tile_size)
```

A schematic implementation is:

```text
n, k, m, tile_size : u64
require tile_size > 0
require n % tile_size == 0 && k % tile_size == 0 && m % tile_size == 0

A : Ix (n, k) => f32
B : Ix (k, m) => f32

a_reindex : (Ix (n / tile_size, k / tile_size)
             × Ix (tile_size, tile_size))
         => Ix (n, k)
a_reindex = \((tile_row, k_tile), (r, kk)) ->
    (tile_row * tile_size + r,
     k_tile * tile_size + kk)

b_reindex : (Ix (k / tile_size, m / tile_size)
             × Ix (tile_size, tile_size))
         => Ix (k, m)
b_reindex = \((k_tile, tile_col), (kk, cc)) ->
    (k_tile * tile_size + kk,
     tile_col * tile_size + cc)

# f >> g = \index -> g(f(index))
A_by_tile = a_reindex >> A
B_by_tile = b_reindex >> B

A_by_tile : (Ix (n / tile_size, k / tile_size)
             × Ix (tile_size, tile_size)) => f32
B_by_tile : (Ix (k / tile_size, m / tile_size)
             × Ix (tile_size, tile_size)) => f32

plan = parallel-plan(
    physical = blocks(n / tile_size, m / tile_size)
             > threads(tile_size, tile_size)
             > 1,
    distribution = tiled(tile_size, tile_size),
)

C = schedule plan, \launch_ctx ->
    # launch_ctx : context (shape (n, m))

    pending_C = materialize(launch_ctx, (n, m),
      \(ctx, index) ->
        # ctx   : context (shape (n, m))
        # index : Ix (n, m)
        # Under perfect tiling, index and ctx.index have equal coordinates,
        # but index is logical while ctx.index is physical.

        block_ctx = ctx.block
        # block_ctx.path.block : Ix (n / tile_size, m / tile_size)
        # block_ctx            : context (shape (tile_size, tile_size))
        # block_ctx.index      : Ix (tile_size, tile_size)

        # The path selects the A/B tiles in the grid.
        (tile_row, tile_col) = block_ctx.path.block

        # The block-scoped index selects this worker's output within a tile.
        (local_row, local_col) = block_ctx.index
        acc = 0

        for k_tile_index : Ix (k / tile_size):
            (k_tile) = k_tile_index

            A_tile_view : Ix (tile_size, tile_size) => f32
            A_tile_view = \a_tile_index ->
                # Input to a_reindex >> A:
                #   output block row, K tile, and local tile index.
                A_by_tile[((tile_row, k_tile), a_tile_index)]

            B_tile_view : Ix (tile_size, tile_size) => f32
            B_tile_view = \b_tile_index ->
                # Input to b_reindex >> B:
                #   K tile, output block column, and local tile index.
                B_by_tile[((k_tile, tile_col), b_tile_index)]

            pending_A = materialize(
                block_ctx, (tile_size, tile_size), A_tile_view)
            pending_B = materialize(
                block_ctx, (tile_size, tile_size), B_tile_view)

            [A_tile, B_tile] = sync(
                block_ctx, [pending_A, pending_B])

            acc += dot(A_tile[local_row, :], B_tile[:, local_col])
            sync(block_ctx, [])

        return acc)

    return sync(launch_ctx, pending_C)

C : buf (size (n, m)) f32
```

The types show the two compatibility checks: context shape `(n, m)` matches
the output task index `Ix (n, m)`, while context shape
`(tile_size, tile_size)` matches the tile-task indices. `a_reindex` and
`b_reindex` contain the only data-coordinate arithmetic. Composing them with
`A` and `B` produces views that the kernel indexes with the block component of
the worker path and a local tile index.

### Matrix multiplication with a tile level

The reduction tiles can also be represented in the execution topology instead
of by an explicit `for` loop. Extend the hierarchy with a tile level below
each CUDA thread:

```text
blocks(n / tile_size, m / tile_size)
> threads(tile_size, tile_size)
> sequential(k / tile_size)
```

The levels have different lowering strategies:

| Level  | Population or work                     | CUDA lowering              |
| ------ | -------------------------------------- | -------------------------- |
| grid   | `blocks(n / tile_size, m / tile_size)` | kernel grid                |
| block  | `threads(tile_size, tile_size)`        | CUDA thread block          |
| thread | one GPU thread                         | GPU thread                 |
| tile   | `Ix (k / tile_size)` work items        | sequential loop per thread |

The first three levels describe the CUDA launch hierarchy. The
`sequential(k / tile_size)` terminal represents the tile level, but does not
add GPU threads or change the kernel launch shape. Materialization through the
grid context still targets the CUDA threads. Materialization through a thread
context uses the sequential distribution, which the CUDA code generator
lowers to the K-tile loop.

Using the same `A_by_tile` and `B_by_tile` reindexings defined above, the
kernel can replace the K-tile loop with another `materialize`:

```text
tile_plan = parallel-plan(
    physical = blocks(n / tile_size, m / tile_size)
             > threads(tile_size, tile_size)
             > sequential(k / tile_size),
    distribution = tiled(tile_size, tile_size),
)

C = schedule tile_plan, \launch_ctx ->
    pending_C = materialize(launch_ctx, (n, m),
      \ctx ->
        block_ctx = ctx.block
        thread_ctx = ctx.thread

        # The worker path selects the output tile.
        # block_ctx.path.block : Ix (n / tile_size, m / tile_size)
        (tile_row, tile_col) = block_ctx.path.block

        # The block-scoped index selects this thread's element in the tile.
        # block_ctx.index : Ix (tile_size, tile_size)
        (local_row, local_col) = block_ctx.index

        # The thread context contains one physical GPU worker.
        # thread_ctx       : context c p thread
        # thread_ctx.shape : Shape 1 = (1)
        # p ends in thread > sequential(k / tile_size)
        pending_partials = materialize(thread_ctx, (k / tile_size),
          \(tile_ctx, k_tile_index) ->
            # tile_ctx.path  = thread_ctx.path
            # tile_ctx.shape : Shape 1 = (1)
            # tile_ctx.index : Ix (1) = (0)
            # k_tile_index   : Ix (k / tile_size)
            (k_tile) = k_tile_index

            A_tile_view : Ix (tile_size, tile_size) => f32
            A_tile_view = \a_tile_index ->
                # a_reindex >> A uses the block row, tile, and local index.
                A_by_tile[((tile_row, k_tile), a_tile_index)]

            B_tile_view : Ix (tile_size, tile_size) => f32
            B_tile_view = \b_tile_index ->
                # b_reindex >> B uses the tile, block column, and local index.
                B_by_tile[((k_tile, tile_col), b_tile_index)]

            pending_A = materialize(
                block_ctx, (tile_size, tile_size), A_tile_view)
            pending_B = materialize(
                block_ctx, (tile_size, tile_size), B_tile_view)

            [A_tile, B_tile] = sync(
                block_ctx, [pending_A, pending_B])

            partial = dot(
                A_tile[local_row, :], B_tile[:, local_col])
            sync(block_ctx, [])
            return partial)

        partials = sync(thread_ctx, pending_partials)
        return reduce(add, 0, partials))

    return sync(launch_ctx, pending_C)
```

Semantically, the inner `materialize` assigns every
`Ix (k / tile_size)` work item to the same GPU worker and produces one partial
result for each. On CUDA, `sequential(k / tile_size)` tells code generation to
implement it as the original per-thread loop. It may fuse the temporary
`partials` buffer and reduction into a scalar accumulator. All threads in a
block traverse the tile indices in the same order, so the block-level
materializations and synchronizations remain collective.

## The three primitives

The model begins with three primitives: `schedule`, `materialize`, and `sync`.

### `schedule`

```text
schedule : configuration p
         × (context c p grid => r)
        -> r

schedule config, \ctx ->
    ... submit work through ctx ...
```

`schedule(config, body)` creates a population of workers and executes `body`
once in the caller. The body receives a grid context; it is not replicated
over the workers. Work reaches those workers through `materialize`.

### `materialize`

```text
materialize : context c p l
            × (n : Shape k)
            × (context c p l × Ix n => t)
           -> pending c (buf (size n) t)
```

`materialize(ctx, n, task)` distributes the logical shape `n` over the workers
represented by `ctx` and runs `task` on them. The task receives a context and
an assigned index in the distribution's state space. The signature above is
the common case in which the state shape is the logical shape `n`; the
generalized semantics are
described below. Each task index has type `Ix n`.

The source may be the context received from `schedule` or a nested context
obtained inside a running task. In both cases, the submitted task receives a
context for the worker that executes it. Targeting a nested context does not
spawn another population; for example, materializing through `ctx.group`
cooperatively fills a group-local buffer using the workers in the current
group.

> **Context-dependent execution:** The code generator or runtime interprets
> `materialize` according to its first parameter. When that parameter is the
> launch context, `materialize(launch_ctx, ...)` dispatches the task using the
> launch parameters so that the workers execute it. When it is a block context,
> `materialize(block, ...)` is performed locally by workers that are already
> running the surrounding task: each worker performs its assigned part **without**
> dispatching new code to its peers.

A pending buffer may be moved and grouped with other pending values, but it
cannot be indexed or otherwise observed until it is passed to `sync`.

### `sync`

```text
sync : context c p l × list (pending c t) -> list t
```

`sync(ctx, values)` waits until the work represented by `values` is complete
and visible at `ctx`, then removes the `pending` wrapper. Synchronizing several pending values together permits one completion point to make all of them ready.

For one value, we use the convenient spelling:

```text
ready = sync(ctx, pending_value)
```

An empty collection performs synchronization without making a new pending
value ready. This is used to protect completed reads before shared storage is
reused:

```text
sync(ctx, [])
```

On the grid context received from `schedule`, `sync` waits for submitted work
such as a GPU kernel. On a group context inside a task, it is collective:
every worker in the group must reach compatible sync points, and the operation
provides the required memory visibility. `sync` is legal only when the context
has the corresponding synchronization capability.

## Materialize distribution strategies

The context's distribution strategy defines how `materialize` relates its
physical workers to the logical indices `Ix n`. Each strategy defines a state
shape `m`, whose indices `Ix m` drive task execution.
The state shape `m : Shape j` and logical shape `n : Shape k` may have
different ranks. Conceptually:

```text
for n : Shape k, m : Shape j:

materialize : context c p
            × (n : Shape k)
            × (context c p
               × Ix m
               × Option (Ix n)
               => produced n t)
           -> pending c (buf (size n) t)

where m = state-shape(distribution(c), n)
      logical-index : Ix m -> Option (Ix n)

produced n t = produce(Ix n, t) | produce-nothing
```

`materialize` allocates a `buf (size n) t`, runs the task according to the
selected strategy, and passes the task a product containing a context, an
`Ix m`, and the result of the strategy's `logical-index` projection. The
`Ix m` type says which state shape the index ranges over; it is
never an untyped candidate coordinate. The optional `Ix n` says whether that
state is also a logical output index. The task returns either
`produce(logical, value)` for that logical index or `produce-nothing`.
Across the whole context, every logical index `Ix n` must be produced
exactly once. `materialize` linearizes that index according to the layout of
`n` when storing the value in the flat `buf (size n) t`. When `m = n`, the
projection is the identity and the shorter
`(context c p × Ix n => t)` task form may be used.

### Perfect tiling

A perfect tiling divides the logical space into tiles whose shape matches the
physical group shape. The number of groups matches the number of logical
tiles, so every worker corresponds to exactly one logical index. For
example:

```text
physical     : groups(4, 4) > threads(16, 16) > 1
output_space : Shape 2 = (64, 64)

group (gy, gx), thread (ty, tx)
    -> (gy * 16 + ty, gx * 16 + tx) : Ix output_space
```

Here `4 × 4 × 16 × 16 = 64 × 64`, and all computed coordinates are within the
logical bounds. The semantics of `materialize` are:

```text
for each worker:
    logical = tiled-index(ctx.path.group, ctx.path.thread)
    result[linearize(n, logical)] = task(ctx, logical)
```

The state shape and logical shape are both `(64, 64)`, so `m = n`. The task
runs once per worker, its argument already has type `Ix (64, 64)`, and no
bounds predicate is required.

### Grid-stride distribution

A grid-stride distribution linearizes the logical space. Worker `r`
starts at logical offset `r` and repeatedly advances by `P`, the total number
of workers, while the offset remains within the logical space. With four
workers and `output_space : Shape 1 = (10)`:

```text
worker 0 -> [0, 4, 8] : sequence (Ix (10))
worker 1 -> [1, 5, 9] : sequence (Ix (10))
worker 2 -> [2, 6]    : sequence (Ix (10))
worker 3 -> [3, 7]    : sequence (Ix (10))
```

The semantics of `materialize` are:

```text
for each worker with global rank r:
    linear = r
    while linear < size output_space:
        logical = unlinearize(output_space, linear)  # Ix n
        result[linear] = task(ctx, logical)
        linear += P
```

A worker may therefore run the task several times. The loop condition is
the proof required to convert the flat offset into `Ix n`; together, the
workers cover every logical index exactly once. `unlinearize` works for any
rank using the shape `n`, so the same loop can produce `Ix (rows, cols)` or a
higher-dimensional index. As in perfect tiling, the state shape exposed to the
task is the logical shape; in the one-dimensional example,
`m = n = (10)`.

### Predicated distribution

A predicated distribution rounds the physical coverage up to a convenient
tile or group size. It dispatches every worker once, then tests whether that
worker's state-space index also lies within the logical
space. For predication, `m` and `n` have the same rank `k`, although their
axis sizes may differ. For eight workers, `m = (8)` while
`output_space : Shape 1 = (6)`:

```text
worker 0 -> 0 : Ix (8) -> Some(0 : Ix (6))
worker 1 -> 1 : Ix (8) -> Some(1 : Ix (6))
...
worker 5 -> 5 : Ix (8) -> Some(5 : Ix (6))
worker 6 -> 6 : Ix (8) -> None
worker 7 -> 7 : Ix (8) -> None
```

The semantics of `materialize` are:

```text
for each worker:
    state_index = physical-index(ctx)  # Ix m
    within = for every axis a in 0 ..< k:
                 state_index[a] < n[a]
    logical = if within
              then Some(refine(output_space, state_index))
              else None
    produced = task(ctx, state_index, logical)
    commit produced
```

The `within` condition is required. An `Ix m` from the padded state shape is
not an `Ix n` until this check proves that all `k` coordinates are within
their corresponding logical bounds.
Indices that do not refine still run the task so that all workers can
participate in collective control flow. The task must not return
before a group barrier merely because its state index is outside `Ix n`; it
produces nothing only after the collective work is complete. When `logical`
is `Some(index)`, `produce` must use that same index.

## Synchronization examples

### Sharing one barrier

```text
group = ctx.group
pending_keys = materialize(group, key_space, load_key)
pending_vals = materialize(group, value_space, load_value)
[keys, vals] = sync(group, [pending_keys, pending_vals])
```

The program states that both buffers must be ready at the same point, allowing
the backend to emit a single barrier.

## Logical and physical indices

A logical index identifies an element of the problem. A physical index
identifies a worker in the execution. `schedule` creates the physical
topology, and `materialize` connects a logical space to that topology. The two
indices are not the same kind of value.

For a `70 × 50` output matrix, the logical index space is:

```text
Ix (70, 50) = {(row, col) | 0 <= row < 70, 0 <= col < 50}
```

These indices have meaning independently of any backend: `(12, 7)` means the
element in row 12 and column 7. They remain the same whether the computation
runs on one CPU worker, 256 GPU threads, or several waves on another GPU.

A physical index instead describes where code is currently executing. A CUDA
interpretation might expose:

```text
ctx.path.block       # blockIdx
ctx.path.warp        # warp within the block, when exposed
ctx.path.thread      # threadIdx or lane within the immediate parent
```

The context's distribution maps the two spaces:

```text
distribution : physical worker
            -> sequence (Ix m × Option (Ix n))
```

Depending on the strategy, a worker may receive exactly one
logical index, a sequence of logical indices, or a state-space index whose
optional logical index is `None`. The
[materialize distribution strategies](#materialize-distribution-strategies)
section defines how `materialize` invokes the task and commits results in
each case. Changing the physical topology may change that assignment, but it
does not change the logical result.

In a simple CUDA kernel these concepts are easily confused:

```text
physical = join(block_coordinate, thread_coordinate)
logical_index = physical
```

Their numeric values happen to be equal because this schedule uses a
one-to-one mapping. They remain different types: another strategy may assign
several logical indices to one worker or predicate a physical
state-space index out of the logical space.

The architecture keeps the logical and physical configuration distinct:

```text
logical shape       # which result elements exist
physical topology   # which workers execute them
```

`schedule` receives the physical topology and distribution policy:

```text
plan = parallel-plan(
    physical = launch_topology,
    distribution = mapping_policy,
)
```

Every materialization supplies the logical space it distributes:

```text
materialize(ctx, n, task)
```

### Imperfectly tiled matrix multiplication

Now let:

```text
A: 70 × 37
B: 37 × 50
C: 70 × 50
tile: 16 × 16
```

The logical result needs `ceil(70 / 16) × ceil(50 / 16) = 5 × 4` tiles. A
natural launch uses:

```text
physical groups:  (5, 4)
threads/group:   (16, 16)
```

The physical coverage is therefore `80 × 64`, but the logical result is only
`70 × 50`. For example:

```text
group (4, 3), thread (0, 0)   -> (64, 48) : Ix (80, 64), valid for C
group (4, 3), thread (5, 1)   -> (69, 49) : Ix (80, 64), valid for C
group (4, 3), thread (6, 1)   -> (70, 49) : Ix (80, 64), invalid for C
group (4, 3), thread (0, 2)   -> (64, 50) : Ix (80, 64), invalid for C
group (4, 3), thread (15, 15) -> (79, 63) : Ix (80, 64), invalid for C
```

Each value is a valid index in the padded `Ix (80, 64)` state space. An index
that does not refine to `Ix (70, 50)` is not a logical index of `C` and must
never be used to index `C`.

This case reveals an important synchronization requirement. Workers whose
state indices are outside the logical output cannot simply skip the kernel
body: the other workers in their group execute `sync(group, ...)`, so all
workers in the group must still participate. A
tiled-and-predicated distribution
therefore gives every worker a state-space index and includes
the corresponding optional logical index directly in the task's input product:

```text
index   : Ix physical_coverage
logical : Option (Ix result_shape)
```

All state-space indices execute collective operations. Only indices with
`Some(logical)` produce output.
The imperfectly tiled computation can then be written schematically as:

```text
plan = parallel-plan(
    physical = groups(5, 4) > threads(16, 16) > 1,
    distribution = tiled-and-predicated(16, 16),
)

C = schedule plan, \launch_ctx ->
    # launch_ctx : context (
    #     physical = groups(5, 4) > threads(16, 16) > 1)
    #
    # The physical topology covers Ix (80, 64), while the logical result is
    # only Ix (70, 50). materialize constructs the relation between them.
    pending_C = materialize(launch_ctx, (70, 50),
      \(ctx, index, logical) ->
        # ctx     : context launch_ctx
        # index   : Ix (80, 64)
        # logical : Option (Ix (70, 50))
        #
        # index cannot index C. To write C, we must obtain the Ix (70, 50)
        # proof carried by Some(logical).
        (row, col) = index
        tile_row = row / 16
        tile_col = col / 16
        local_row = row % 16
        local_col = col % 16

        # local_row and local_col are each < 16 by the definition of `% 16`.
        # This proves:
        #     (local_row, local_col) : Ix (16, 16)
        # so the coordinate can index the group-local tiles after sync.
        acc = 0

        for k_tile in 0 ..< ceil(37 / 16):
            group = ctx.group

            pending_A = materialize(group, (16, 16), \(r, k) ->
                # (r, k) : Ix (16, 16)
                a_row = tile_row * 16 + r
                a_col = k_tile * 16 + k

                # The launch, loop, and tile bounds prove:
                #     (a_row, a_col) : Ix (80, 48)
                # This is a physical/padded index, not an Ix (70, 37), so it
                # cannot index A directly.
                match refine((70, 37), (a_row, a_col)):
                    # a_index : Ix (70, 37)
                    # The Some branch carries proofs a_row < 70 and
                    # a_col < 37.
                    Some(a_index) -> A[a_index]
                    None          -> 0)

            pending_B = materialize(group, (16, 16), \(k, c) ->
                # (k, c) : Ix (16, 16)
                b_row = k_tile * 16 + k
                b_col = tile_col * 16 + c

                # The launch, loop, and tile bounds prove:
                #     (b_row, b_col) : Ix (48, 64)
                # B requires Ix (37, 50), so the padded index must be refined
                # before it can index B.
                match refine((37, 50), (b_row, b_col)):
                    # b_index : Ix (37, 50)
                    # The Some branch carries proofs b_row < 37 and
                    # b_col < 50.
                    Some(b_index) -> B[b_index]
                    None          -> 0)

            # pending_A : pending group (buf 256 f32), viewed as Ix (16, 16)
            # pending_B : pending group (buf 256 f32), viewed as Ix (16, 16)

            [A_tile, B_tile] = sync(group, [pending_A, pending_B])

            # A_tile, B_tile : ready group-local (buf 256 f32)
            # The sync proves that cooperative writes are complete and
            # visible to every worker in `group`.

            # Every worker reaches the sync above. State-space
            # indices outside C may compute a value, but it is not committed.
            acc += dot(A_tile[local_row, :], B_tile[:, local_col])

            # Protect reads before reusing A_tile and B_tile.
            sync(group, [])

        # Producing C requires an Ix (70, 50), not the padded state-space
        # index. Exhaustive matching forces the inactive case.
        # This could be part of materialize implementation
        # on the contrary of the other guards that must be in user program
        return match logical:
            # c_index : Ix (70, 50)
            Some(c_index) -> produce(c_index, acc)
            None          -> produce-nothing)

    # pending_C : pending launch_ctx (buf 3500 f32)
    return sync(launch_ctx, pending_C)
```

The zero values on out-of-bounds tile loads are padding in the physical
execution. They do not add elements to the logical matrices. The exhaustive
match on `logical` makes `materialize` commit exactly one value for each valid
logical index and nothing for state-space indices outside the result.

Materialization passes the state and logical indices as separate components of
the task's input product, so imperfect coverage does not remove physical
workers from collective control flow:

```text
materialize : context c p l
            × (n : Shape k)
            × (context c p l
               × Ix m
               × Option (Ix n)
               => produced n t)
           -> pending c (buf (size n) t)
```

For perfect coverage, every state index has a logical index. For imperfect
coverage, state indices without a logical index execute the task but do not
commit a value to the result. The governing semantic rule is:

> Workers may be inactive with respect to logical output, but
> they must remain active with respect to collective control flow.

This distinction is also needed when there are more logical tiles than
physical groups. A group may process several logical tiles in successive
rounds; the group component of its physical path stays constant while the
logical tile index changes. As long as the distribution covers every logical
index exactly once and keeps group collectives convergent, the result is
unchanged.

## CUDA interpretation

One CUDA lowering could use the following correspondence:

| Abstract concept            | CUDA interpretation                                    |
| --------------------------- | ------------------------------------------------------ |
| configuration               | grid and block execution configuration                 |
| schedule body               | caller-side orchestration                              |
| schedule-scope materialize  | kernel dispatch plus global-memory stores              |
| group                       | thread block                                           |
| warp/subgroup               | warp                                                   |
| worker / leaf topology node | GPU thread                                             |
| context path                | `blockIdx`, `threadIdx`, and derived warp/lane indices |
| scoped context index        | grid-, block-, warp-, or thread-relative coordinate    |
| group synchronization       | `__syncthreads()`                                      |
| warp synchronization        | `__syncwarp()` or warp-synchronous operations          |
| group-local materialization | shared-memory allocation plus cooperative stores       |
| schedule-scope sync         | stream/event synchronization or an ordering dependency |

This is a lowering, not the definition of the primitives. In particular, a
backend is free to assign indices with a grid-stride loop, map a logical warp
to a different subgroup width, or serialize the whole context for a reference
interpreter.

## Other execution models

The same structure can describe other targets:

- A CPU backend can map a group to a worker pool and a thread to one worker.
- A SIMD backend can map a subgroup to a vector and an agent to a lane.
- HIP, Metal, and Vulkan compute backends can provide their native workgroup
  and subgroup topologies.
- A sequential backend can use a one-worker context, which is useful for
  testing semantics independently of parallel code generation.

Backends need not support every topology or capability. A program requiring a
subgroup shuffle, global barrier, or group-local storage should fail capability
checking when the selected context cannot provide it.

## Relationship to existing Catena concepts

Catena already represents a logical tensor or multidimensional view as an
indexed closure and materializes it into a flat, dependently sized buffer. This proposal retains that model. It adds an explicit execution context to describe _who_ evaluates the indexed closure and a `pending` phase to describe _when_ the result is safe to observe.

The rank-polymorphic source-level primitive is written consistently as:

```text
materialize : (n : Shape k) × (Ix n => t) -> buf (size n) t
```

## Correctness of GPU kernels

This is out of scope for now, but the following "theorem" should prove that a
GPU kernel created from `materialize + sync` is total and deterministic:

```
distribution is finite
distribution covers every logical index
distribution produces each logical index uniquely
task is total
task effects are deterministic and race-free
task syncs are convergent
--------------------------------------------------
materialize + sync is total and deterministic
```

### Distributions

Standard distributions provide allocation propositions automatically. Custom distributions must provide equivalent proofs.

```text
materialize : context (workers m)
            × (n : Shape k)
            × distribution m n
            × task ...
           -> pending (buf (size n) t)
```

Constructor guards

```
perfect : n -> distribution n n

predicated :
      n
    × m
    × |- n <= m
    -> distribution m n

grid_stride :
      n
    × m
    × |- size(m) > 0
    -> distribution m n
```

### Race free

#### Shared memory

A conservative sufficient condition is to divide shared-memory execution into synchronization phases. For every shared-memory location within one phase:

- Either all threads only read it.
- Or exactly one thread owns it (only writer), and no other thread reads or writes it.
- Or all conflicting accesses use a correctly scoped atomic operation.

Some useful properties

- If a distribution is valid, every work item is assigned to exactly one worker. `materialize` "constructs" the proof from the distribution. Roughly, a task can have this signature:

```
task :
    (worker   : context c p l)
    × (work-item-index   : Ix n)
    × (|- owns(worker.path, work-item-index))
    -> t
```

### `pending` and `sync` solve only one side

`pending` and `sync` establish producer-to-consumer ordering:

```text
writes to shared data -> sync -> reads of shared data
```

They do not ensure that all reads finish before the storage is reused:

```text
reads of shared data -> synchronization -> next writes to shared data
```

An empty `sync(ctx, [])` can provide this barrier, but the current types do not require it. Either code generation must insert barriers when reusing storage, or shared storage must track a linear `writable -> pending -> readable -> writable` state.
