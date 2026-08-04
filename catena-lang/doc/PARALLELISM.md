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
- **Workers**: the leaf executions in the topology.
- **Worker shape**: the physical arrangement of those workers.
- **Logical shape**: the shape of the problem or result being computed.
- **Physical distribution**: how the available workers are organized in the
  execution topology.
- **Logical distribution**: how the work items of a computation are assigned
  to those workers.
- **Task**: the computation executed by the workers.
- **Work item**: one logical unit of that task, such as one element of its
  result.
- **Context configuration**: the topology, distributions, and capabilities
  used to execute tasks on a worker population.
- **Active context**: the worker-side information available while a task is
  running, including the current worker and its assigned work.

Consider a CUDA grid containing `4 × 4` blocks, where every block contains
`16 × 16` threads:

| Worker population |    Direct children | Worker shape | Worker count |
| ----------------- | -----------------: | -----------: | -----------: |
| grid              |    `(4, 4)` blocks |   `(64, 64)` |       `4096` |
| current block     | `(16, 16)` threads |   `(16, 16)` |        `256` |
| current thread    |               none |    singleton |          `1` |

`size` gives the cardinality of a worker shape. For a two-dimensional shape,
`size((n, m)) = n * m`; therefore, the block worker shape `(16, 16)` has size
256, while the grid worker shape `(64, 64)` has size 4096.

Parallel execution therefore has a physical distribution and a logical
distribution. The physical distribution describes the workers available to
run the computation. In CUDA, it is determined by the launch geometry: the
grid organizes GPU threads into blocks. The logical distribution describes
the work items to compute, such as the elements of an output array, and how
those work items are assigned to the available workers.

The topology is not the logical shape of the data. The simplest and most
common assignment gives one work item to each worker, so the physical worker
shape and logical shape coincide. They need not coincide: 256 workers may
compute 10,000 result elements by handling several work items each, while a
launch with more workers than result elements leaves some workers without an
output item. Keeping the two distributions separate allows the same
computation to use different launch sizes or backends.

A _context configuration_ describes how a worker population will execute
work. It combines the physical topology, the logical distribution, and the
capabilities available to the workers, such as synchronization or shared
memory. It is created outside the running task, so it has no current worker or
assigned work.

When a task runs, each worker receives an _active context_ derived from the
configuration. The active context describes that particular execution: which
worker is running, which work has been assigned to it, and which capabilities
it may use. For example, a GPU worker can obtain an active context for its
current block when cooperating with the other workers in that block.

Three primitives connect the topology, workers, and logical work.

`schedule` establishes the physical worker organization. Its body runs once
in the caller and receives a context configuration for that worker population.
The configuration does not perform the parallel work itself; it is the handle
through which work is later submitted and synchronized.

`materialize` submits a _task_ through a context configuration. The task is
the unit of work executed by workers. Each task invocation receives the active
context and performs the logical work assigned to its worker. A worker may be
assigned one work item, several work items, or no output item. `materialize`
collects the results into a buffer but returns it as pending, because the work
may still be executing. A task may also obtain an active block context and
start cooperative work, such as filling shared memory.

`sync` completes pending work associated with a context and makes its results
safe to observe. For a grid configuration it may wait for a kernel. For an
active block context it is a collective operation reached by all workers in
the block, for example after they have cooperatively filled shared memory.
Together, `schedule` chooses the worker organization, `materialize` states the
task and its logical result, and `sync` states when that result may be used.

### CUDA example

For CUDA, the abstract concepts above can be interpreted as follows:

- topology = `grid > block > thread`
- worker = a GPU thread executing the task
- task = the kernel
- work item = one output cell, such as `C[i, j]` in matrix multiplication

With `4 × 4` blocks and `16 × 16` threads per block, scheduling establishes a
grid containing a `(64, 64)` worker shape. Materializing through the grid
configuration launches the task as a kernel across all 4096 GPU threads:

```text
schedule blocks(4, 4) > threads(16, 16), \launch_ctx ->
    pending_output = materialize(launch_ctx, (64, 64),
      \(active, work_item) ->
        block_ctx = current_block(active)
        pending_tile = materialize(block_ctx, (16, 16),
          \tile_item ->
            load_tile_element(tile_item))

        tile = sync(block_ctx, pending_tile)
        compute_output(tile, work_item))

    output = sync(launch_ctx, pending_output)
```

Inside the kernel task, obtaining the active block context selects the 256 GPU
threads in the current block. The nested materialization uses those workers to
fill the `(16, 16)` shared-memory tile cooperatively before the task computes
its output.

Thus the same abstract operation can use different CUDA levels: grid-level
`materialize` becomes kernel-wide work, while block-level `materialize`
becomes cooperative work by the GPU threads of one block.

## Contexts

The schematic type signatures use two context parameters:

```text
p : physical topology
c : execution-context identity
```

For example, `p` may be `groups(4, 4) > threads(16, 16)`. The parameter `c`
identifies the context whose work is being submitted or synchronized; this is
why `context-configuration c p`, `active-context c p`, and `pending c` agree
on the same `c`. Neither parameter describes the logical index shape.

The context configuration has no current thread, group, or lane position
because it exists outside task execution.

`materialize` submits an indexed task through the context configuration. That
task runs on the workers and receives an _active context_. In addition to the
shared topology and capabilities, the active context gives each worker access
to:

1. **Position**: its coordinates in the physical hierarchy.
2. **Index assignment**: the indices in the distribution's state space that
   are assigned to it by the materialization.

Active contexts form a hierarchy. A kernel may narrow its context to one of
its physical scopes:

```text
active.group
active.warp
active.thread
```

## Index-space notation

An index-space shape may have any rank:

> **Notation note:** Catena does not currently have a `Shape` type. `Shape k`
> is introduced only as shorthand for expressing the rank and dimensions of
> the index types in this document. It is not a proposal or requirement for
> how shapes, index spaces, or `materialize` must be represented in a future
> implementation.

```text
Shape k = Vec k u64
n       : Shape k

Ix n = {coord : Vec k u64 |
        for every axis a, coord[a] < n[a]}

size n = product(a in 0 ..< k, n[a]) : u64
```

`Ix (10)` and `Ix (64, 64)` are index types. Example values are:

```text
(1)     : Ix (10)
(3, 49) : Ix (64, 64)
```

Likewise, `Ix (4, 8, 16)` is a three-dimensional index type. In the signatures
below, `n` is a shape of the appropriate rank rather than a single size. `Ix n`
is the Catena type of coordinates proven to be within that shape; it is not
wrapped in a second `index` type. `size n` is the number of elements:

```text
size (n, m) = n * m
size (n0, ..., nk-1) = n0 * ... * nk-1

size (10) = 10
size (64, 64) = 64 * 64 = 4096
```

This size determines the length of the flat buffer produced by `materialize`.

## The three primitives

The model begins with three primitives: `schedule`, `materialize`, and `sync`.

### `schedule`

```text
schedule : dimensions p × (context-configuration c p => r) -> r

schedule dims, \ctx ->
    ... submit work through ctx ...
```

`schedule(dims, body)` creates a scoped, logical population of workers and
executes `body` once in the caller. The body receives the
context-configuration handle `ctx`; it is not replicated over the parallel
workers. Work reaches those workers through `materialize`.

### `materialize`

At schedule scope:

```text
materialize : context-configuration c p
            × (n : Shape k)
            × (active-context c p × Ix n => t)
           -> pending c (buf (size n) t)
```

`materialize(ctx, n, task)` distributes the logical shape `n` over the workers
represented by `ctx` and runs `task` on them. The task receives an active
context and an assigned index in the
distribution's state space. The signature above is the common case in which
the state shape is the logical shape `n`; the generalized semantics are
described below. Each task index has type `Ix n`.

Inside a running task, `materialize` can target an active subcontext:

```text
materialize : active-context c p
            × (n : Shape k)
            × (Ix n => t)
           -> pending c (buf (size n) t)
```

This form distributes nested work over workers already executing in that
scope. It does not spawn another population. For example, materializing
through `active.group` cooperatively fills a group-local buffer using the
workers in the current group.

A pending buffer may be moved and grouped with other pending values, but it
cannot be indexed or otherwise observed until it is passed to `sync`.

### `sync`

```text
sync : context c p × list (pending c t) -> list t
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

On a context configuration, `sync` waits for submitted work such as a
GPU kernel. On an active group context, it is collective: every worker in the
scope must reach compatible sync points, and the operation provides the
required memory visibility. `sync` is legal only when the context has the
corresponding synchronization capability.

## Materialize distribution strategies

The context's distribution strategy defines how `materialize` relates the
physical workers in the context to the logical indices `Ix n`. Each
strategy defines a state shape `m`, whose indices `Ix m` drive task execution.
The state shape `m : Shape j` and logical shape `n : Shape k` may have
different ranks. Conceptually:

```text
for n : Shape k, m : Shape j:

materialize : context c p
            × (n : Shape k)
            × (active-context c p
               × Ix m
               × Option (Ix n)
               => produced n t)
           -> pending c (buf (size n) t)

where m = state-shape(distribution(c), n)
      logical-index : Ix m -> Option (Ix n)

produced n t = produce(Ix n, t) | produce-nothing
```

`materialize` allocates a `buf (size n) t`, runs the task according to the
selected strategy, and passes the task a product containing the active
context, an `Ix m`, and the result of the strategy's `logical-index`
projection. The `Ix m` type says which state shape the index ranges over; it is
never an untyped candidate coordinate. The optional `Ix n` says whether that
state is also a logical output index. The task returns either
`produce(logical, value)` for that logical index or `produce-nothing`.
Across the whole context, every logical index `Ix n` must be produced
exactly once. `materialize` linearizes that index according to the layout of
`n` when storing the value in the flat `buf (size n) t`. When `m = n`, the
projection is the identity and the shorter
`(active-context c p × Ix n => t)` task form may be used.

### Perfect tiling

A perfect tiling divides the logical space into tiles whose shape matches the
physical group shape. The number of groups matches the number of logical
tiles, so every worker corresponds to exactly one logical index. For
example:

```text
physical     : groups(4, 4) > threads(16, 16)
output_space : Shape 2 = (64, 64)

group (gy, gx), thread (ty, tx)
    -> (gy * 16 + ty, gx * 16 + tx) : Ix output_space
```

Here `4 × 4 × 16 × 16 = 64 × 64`, and all computed coordinates are within the
logical bounds. The semantics of `materialize` are:

```text
for each worker:
    logical = tiled-index(group.rank, thread.rank)
    result[linearize(n, logical)] = task(active, logical)
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
        result[linear] = task(active, logical)
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
    state_index = physical-index(active)  # Ix m
    within = for every axis a in 0 ..< k:
                 state_index[a] < n[a]
    logical = if within
              then Some(refine(output_space, state_index))
              else None
    produced = task(active, state_index, logical)
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

## Basic examples

### CUDA-like launch

```text
schedule dims, \launch_ctx ->
    pending_buf = materialize(
        launch_ctx,
        output_space,
        \(active, index) -> kernel(active, index))

    buf = sync(launch_ctx, pending_buf)
    return buf
```

The schedule body runs once in the caller. `launch_ctx` represents the worker
at every CUDA thread position in every group created by the launch.
`materialize` dispatches `kernel` to that complete worker population and
supplies each worker with its active context, assigned state-space indices,
and their optional logical indices.

The physical size of `launch_ctx` and the logical size of `output_space` must
be compatible with the context's distribution policy. For example, an exact
two-dimensional distribution may have these schematic types:

```text
p           = groups(4, 4) > threads(16, 16)
launch_ctx  : context-configuration c p
output_space: Shape 2 = (64, 64)
active      : active-context c p
index       : Ix output_space
```

There are `4 × 4 × 16 × 16 = 4096` CUDA thread positions, hence 4096 workers,
and `64 × 64 = 4096` output elements. The distribution relates their shapes,
and `materialize` gives the task an `Ix (64, 64)`, matching the shape used
to allocate the result.

The distribution strategies above describe other compatible relationships
between the physical launch and `output_space`. With a predicated CUDA
distribution, the task receives the full product because some physical
workers have no logical index. The CUDA lowering must explicitly test the
state-space index against the logical bounds before constructing the optional
task index:

```text
state_shape = (
    gridDim.y * blockDim.y,
    gridDim.x * blockDim.x)

state_index = (
    blockIdx.y * blockDim.y + threadIdx.y,
    blockIdx.x * blockDim.x + threadIdx.x) : Ix state_shape

logical = if state_index.y < output_space.rows
          && state_index.x < output_space.cols
          then Some(refine(output_space, state_index))
          else None

task(active, state_index, logical)
```

Only the `Some` branch carries an index compatible with `output_space` and may
commit an output element. Workers in the `None` branch still execute any group
collectives in the kernel.

`sync(launch_ctx, pending_buf)` waits in the caller until the materialization
is complete.

### Cooperative loading within a group

```text
kernel = \active, index ->
    group = active.group
    pending_tile = materialize(group, tile_space, \tile_index ->
        f(tile_index))
    tile = sync(group, pending_tile)
    return use(tile, index)
```

The kernel is already running on the scheduled workers. The nested
`materialize` distributes `tile_space` over `group`, which represents all the
workers in the current group and no workers from any other group; it does not
spawn another population. The number of workers in `group` and the size of
`tile_space` must likewise be compatible with the distribution policy. For
the launch above, one exact cooperative mapping has these types:

```text
group       : active-context g (threads(16, 16))
tile_space  : Shape 2 = (16, 16)
tile_index  : Ix tile_space
pending_tile: pending g (buf (size tile_space) t)
```

The group contains 256 CUDA thread positions (`16 × 16`), hence 256 workers,
and the tile contains 256 elements (`16 × 16`). The `tile_index` type matches
`tile_space`, so it can index the resulting tile. An index from the outer
`Ix (64, 64)` output space is not a tile index and cannot be substituted
without an explicit conversion or refinement.

A one-to-one cooperative load requires one worker per tile element; other
policies may make each worker load several elements or make excess workers
inactive for the load, while retaining them for collective synchronization.
No worker can observe `tile` until the group synchronization discharges
`pending_tile`. The backend may place this buffer in CUDA shared memory when
its size and lifetime permit it.

### Sharing one barrier

```text
group = active.group
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
active.group.rank       # blockIdx
active.thread.rank      # threadIdx
active.warp.rank
active.lane.rank
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
physical = block_rank * group_size + thread_rank
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

### Perfectly tiled matrix multiplication

Suppose `A`, `B`, and `C` are `64 × 64` matrices and the tile size is
`16 × 16`. The logical output contains `4 × 4` tiles. We launch `4 × 4`
physical groups, each containing `16 × 16` CUDA thread positions:

```text
logical C:       (64, 64)
logical tiles:   (4, 4)

physical groups:  (4, 4)
threads/group:   (16, 16)
```

The tiled distribution is:

```text
logical_tile = physical_group_rank

logical_row = logical_tile.row * 16 + physical_thread_rank.row
logical_col = logical_tile.col * 16 + physical_thread_rank.col
```

This is a perfect tiling: every worker at a CUDA thread position maps to one
valid logical output index, and every logical output index has exactly one
worker producing it. The equal coordinates are a property of this schedule,
not an identification of logical tiles with physical groups.

A schematic implementation using the primitives is:

```text
plan = parallel-plan(
    physical = groups(4, 4) > threads(16, 16),
    distribution = tiled(16, 16),
)

C = schedule plan, \launch_ctx ->
    pending_C = materialize(launch_ctx, (64, 64),
      \(active, index) ->
        # index : Ix (64, 64)
        (row, col) = index
        tile_row = row / 16
        tile_col = col / 16
        local_row = row % 16
        local_col = col % 16
        acc = 0

        for k_tile in 0 ..< 4:
            group = active.group

            pending_A = materialize(group, (16, 16), \(r, k) ->
                A[tile_row * 16 + r, k_tile * 16 + k])

            pending_B = materialize(group, (16, 16), \(k, c) ->
                B[k_tile * 16 + k, tile_col * 16 + c])

            [A_tile, B_tile] = sync(group, [pending_A, pending_B])

            acc += dot(A_tile[local_row, :], B_tile[:, local_col])

            # All workers must finish reading the current tiles before the
            # next iteration overwrites their group-local storage.
            sync(group, [])

        return acc)

    return sync(launch_ctx, pending_C)
```

This pseudocode omits representation details. Its important mappings are:

- The outer `materialize` distributes logical `(row, col)` output indices over
  the workers represented by `launch_ctx`.
- Every group materializes the logical `16 × 16` spaces of its `A` and `B`
  tiles using its workers.
- `sync(group, ...)` is tied to the physical group, because those are the
  workers that wrote and will read the tile buffers.
- The second `sync(group, [])` in each K iteration protects the completed reads
  before the next iteration reuses the same group-local storage.
- Indices into `A`, `B`, and `C` remain logical data coordinates throughout.

The final `sync(launch_ctx, pending_C)` runs in the caller and waits for the
outer materialization to complete. It is not an in-kernel grid barrier.

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
    physical = groups(5, 4) > threads(16, 16),
    distribution = tiled-and-predicated(16, 16),
)

C = schedule plan, \launch_ctx ->
    # launch_ctx : context-configuration (
    #     physical = groups(5, 4) > threads(16, 16))
    #
    # The physical topology covers Ix (80, 64), while the logical result is
    # only Ix (70, 50). materialize constructs the relation between them.
    pending_C = materialize(launch_ctx, (70, 50),
      \(active, index, logical) ->
        # active  : active-context launch_ctx
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
            group = active.group

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
materialize : context-configuration c p
            × (n : Shape k)
            × (active-context c p
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
rounds; its physical group rank stays constant while the logical tile index
changes. As long as the distribution covers every logical index exactly once
and keeps group collectives convergent, the result is unchanged.

## CUDA interpretation

One CUDA lowering could use the following correspondence:

| Abstract concept            | CUDA interpretation                                    |
| --------------------------- | ------------------------------------------------------ |
| context configuration       | grid and block execution configuration                 |
| schedule body               | caller-side orchestration                              |
| schedule-scope materialize  | kernel dispatch plus global-memory stores              |
| group                       | thread block                                           |
| warp/subgroup               | warp                                                   |
| worker / leaf topology node | GPU thread                                             |
| context position            | `blockIdx`, `threadIdx`, and derived lane indices      |
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
