# Context-Based Parallelism

This document sketches a parallel execution model for Catena. CUDA is the
first target, but CUDA concepts such as grids, blocks, warps, and threads are
not language primitives. They are one possible interpretation of a more
general _execution context_.

The central idea is:

> A context describes a hierarchical execution topology and determines how a
> logical index space is distributed over the participants in that topology.

Programs express parallel work in terms of contexts, indexed producers, and
synchronization. A backend decides how those concepts map to GPU launches,
SIMD lanes, CPU workers, or another execution model.

This is an initial design. The notation and types below are schematic rather
than declarations in the current Catena syntax.

## Goals

- Represent CUDA-style launch and cooperative execution without baking CUDA
  names into the core language.
- Keep the logical index space separate from the physical execution topology.
- Make values produced by cooperative work unavailable until the required
  synchronization has occurred.
- Permit nested scopes such as group > warp > thread.
- Leave scheduling, memory placement, and synchronization details available to
  backend-specific lowering.

Non-goals for the first version include dynamic task graphs, arbitrary
cross-device communication, and a single cost model that works for every
backend.

## Contexts

`schedule` creates a context representing a scoped population of parallel
processes. A process here is an abstract execution agent: it may lower to a GPU thread, CPU worker, SIMD lane, or sequential interpreter iteration.

The schedule body runs once in the caller and receives a _schedule context_.
This is a handle used to submit work to the population. It contains:

1. **Topology**: the hierarchy and extent of the execution, for example
   `grid > group > warp > thread`.
2. **Distribution policy**: how logical index spaces are assigned to the
   population.
3. **Capabilities**: operations supported by the topology, such as group
   barriers, shared storage, subgroup operations, or asynchronous copies.

The schedule context has no current thread, group, or lane position because
the caller is not one of the parallel processes.

`materialize` submits an indexed producer to the schedule context. That
producer runs on the parallel processes and receives an _active context_. In
addition to the shared topology and capabilities, the active context gives
each process access to:

1. **Position**: its coordinates in the physical hierarchy.
2. **Index assignment**: the logical indices or slots assigned to it by the
   materialization.

The topology is not the logical shape of the data. A context with 256 physical
threads may cover a logical space of 10,000 elements by assigning several
indices to each thread. Conversely, some participants may receive no index.
This separation allows a program to retain the same meaning when launch sizes
or backends change.

Active contexts form a hierarchy. A kernel may narrow its context to one of
its physical scopes:

```text
active.group
active.warp
active.thread
```

The schedule context and all active contexts produced from it share a
generative identity. Pending work, group-local storage, and synchronization
are tagged with that identity so an unrelated context cannot complete them.

## The three primitives

The model begins with three primitives: `schedule`, `materialize`, and `sync`.

### `schedule`

```text
schedule : dimensions p × (schedule-context c p => r) -> r

schedule dims, \ctx ->
    ... submit work through ctx ...
```

`schedule(dims, body)` spawns a scoped, logical population of parallel
processes and executes `body` once in the caller. The body receives the
schedule-context handle `ctx`; it is not replicated over the parallel
processes. Work reaches those processes through `materialize`.

The context cannot escape the schedule body. Every pending operation submitted
through it must be synchronized before the schedule ends. A backend realizes
the population using its execution model, such as GPU threads, a worker pool,
vector lanes, or a sequential loop.

### `materialize`

At schedule scope:

```text
materialize : schedule-context c p
            × space s
            × (active-context c p × slot s => t)
           -> pending c (buf s t)
```

`materialize(ctx, space, producer)` distributes the logical space over the
parallel processes represented by `ctx` and runs `producer` on them. The
producer receives an active context and an assigned logical slot. Every logical index is produced exactly once.

When every physical slot is active, the producer may bind the slot's logical
index directly as a shorthand.

Inside a running producer, `materialize` can target an active subcontext:

```text
materialize : active-context c p
            × space s
            × (slot s => t)
           -> pending c (buf s t)
```

This form distributes nested work over processes already executing in that
scope. It does not spawn another population. For example, materializing through `active.group` cooperatively fills a group-local buffer using the threads in the current group.

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

On a caller-side schedule context, `sync` waits for submitted work such as a
GPU kernel. On an active group context, it is collective: every process in the
scope must reach compatible sync points, and the operation provides the
required memory visibility. `sync` is legal only when the context has the
corresponding synchronization capability.

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

The schedule body runs once in the caller. `materialize` dispatches `kernel` to the parallel population and supplies each process with its active context and assigned logical indices. `sync(launch_ctx, pending_buf)` waits in the caller until the materialization is complete.

### Cooperative loading within a group

```text
kernel = \active, index ->
    group = active.group
    pending_tile = materialize(group, tile_space, \tile_index ->
        f(tile_index))
    tile = sync(group, pending_tile)
    return use(tile, index)
```

The kernel is already running on the scheduled processes. The nested
`materialize` distributes `tile_space` over the threads in the current group;
it does not spawn more processes. No thread can observe `tile` until the group
synchronization discharges `pending_tile`. The backend may place this buffer in CUDA shared memory when its size and lifetime permit it.

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
identifies a participant in the execution. `schedule` creates the physical
topology, and `materialize` connects a logical space to that topology. The two
indices are not the same kind of value.

For a `70 × 50` output matrix, the logical index space is:

```text
Ix (70, 50) = {(row, col) | 0 <= row < 70, 0 <= col < 50}
```

These indices have meaning independently of any backend: `(12, 7)` means the
element in row 12 and column 7. They remain the same whether the computation
runs on one CPU thread, 256 CUDA threads, or several waves on another GPU.

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
distribution : physical participant -> sequence of logical indices
```

For ten logical elements and four physical threads, a grid-stride
distribution could be:

```text
thread 0 -> [0, 4, 8]
thread 1 -> [1, 5, 9]
thread 2 -> [2, 6]
thread 3 -> [3, 7]
```

Conceptually, a caller-level `materialize` executes:

```text
materialize(launch_ctx, logical_space, f):
    allocate result for logical_space
    dispatch f to launch_ctx's parallel processes

    on each process with active context:
        for logical_slot in active.assigned_slots:
            result[logical_slot.index] = f(active, logical_slot)

    return pending(launch_ctx, result)
```

The physical thread rank selects the sequence, while `logical_slot.index`
selects a result element. Changing the number of physical threads changes the
distribution, but not the logical result.

In a simple CUDA kernel these concepts are easily confused:

```text
physical = block_rank * group_size + thread_rank
logical_index = physical
```

Their numeric values happen to be equal because this schedule uses a one-to-one mapping. The grid-stride example makes the difference apparent: one physical index is responsible for several logical indices.

The architecture keeps the logical and physical configuration distinct:

```text
logical shape       # which result elements exist
physical topology   # which participants execute them
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
materialize(ctx, logical_space, producer)
```

### Perfectly tiled matrix multiplication

Suppose `A`, `B`, and `C` are `64 × 64` matrices and the tile size is
`16 × 16`. The logical output contains `4 × 4` tiles. We launch `4 × 4` physical groups, each containing `16 × 16` physical threads:

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

This is a perfect tiling: every physical thread maps to exactly one valid
logical output index, and every logical output index has exactly one physical
producer. The equal coordinates are a property of this particular schedule,
not an identification of logical tiles with physical groups.

A schematic implementation using the primitives is:

```text
plan = parallel-plan(
    physical = groups(4, 4) > threads(16, 16),
    distribution = tiled(16, 16),
)

C = schedule plan, \launch_ctx ->
    pending_C = materialize(launch_ctx, Ix (64, 64),
      \(active, slot) ->
        (row, col) = slot.logical
        tile_row = row / 16
        tile_col = col / 16
        local_row = row % 16
        local_col = col % 16
        acc = 0

        for k_tile in 0 ..< 4:
            group = active.group

            pending_A = materialize(group, Ix (16, 16), \(r, k) ->
                A[tile_row * 16 + r, k_tile * 16 + k])

            pending_B = materialize(group, Ix (16, 16), \(k, c) ->
                B[k_tile * 16 + k, tile_col * 16 + c])

            [A_tile, B_tile] = sync(group, [pending_A, pending_B])

            acc += dot(A_tile[local_row, :], B_tile[:, local_col])

            # All threads must finish reading the current tiles before the
            # next iteration overwrites their group-local storage.
            sync(group, [])

        return acc)

    return sync(launch_ctx, pending_C)
```

This pseudocode omits representation details. Its important mappings are:

- The outer `materialize` distributes logical `(row, col)` output indices over
  the processes represented by `launch_ctx`.
- Every group materializes the logical `16 × 16` spaces of its `A` and `B`
  tiles using its physical threads.
- `sync(group, ...)` is tied to the physical group, because those are the
  participants that wrote and will read the tile buffers.
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
group (4, 3), thread (0, 0)   -> candidate logical index (64, 48), valid
group (4, 3), thread (5, 1)   -> candidate logical index (69, 49), valid
group (4, 3), thread (6, 1)   -> candidate logical index (70, 49), invalid
group (4, 3), thread (0, 2)   -> candidate logical index (64, 50), invalid
group (4, 3), thread (15, 15) -> candidate logical index (79, 63), invalid
```

An invalid candidate is not a logical index of `C`; it is only a coordinate
obtained from the physical tiling. It must never be used to index `C`.

This case reveals an important synchronization requirement. Threads mapped to
invalid output candidates cannot simply skip the kernel body: the other
threads in their group execute `sync(group, ...)`, so all physical threads in
the group must still participate. A tiled distribution therefore needs a
_slot_ for every physical participant:

```text
slot.candidate : integer coordinates
slot.logical   : optional (Ix result_shape)
slot.active    : whether slot.logical exists
```

All slots execute collective operations. Only active slots produce output.
The imperfectly tiled computation can then be written schematically as:

```text
plan = parallel-plan(
    physical = groups(5, 4) > threads(16, 16),
    distribution = tiled-and-predicated(16, 16),
)

C = schedule plan, \launch_ctx ->
    # launch_ctx : schedule-context (
    #     physical = groups(5, 4) > threads(16, 16))
    #
    # The physical topology covers Ix (80, 64), while the logical result is
    # only Ix (70, 50). materialize constructs the relation between them.
    pending_C = materialize(launch_ctx, Ix (70, 50),
      \(active, slot) ->
        # active : active-context launch_ctx
        # slot   : slot (logical = Ix (70, 50), physical = Ix (80, 64))
        #
        # slot.candidate : Ix (80, 64)
        # slot.logical   : Option (Ix (70, 50))
        #
        # slot.candidate cannot index C. To write C, we must obtain the
        # Ix (70, 50) proof carried by Some(slot.logical).
        (row, col) = slot.candidate
        tile_row = row / 16
        tile_col = col / 16
        local_row = row % 16
        local_col = col % 16

        # local_row and local_col are each < 16 by the definition of `% 16`.
        # This proves (local_row, local_col) : Ix (16, 16), so they can index
        # the group-local tiles after synchronization.
        acc = 0

        for k_tile in 0 ..< ceil(37 / 16):
            group = active.group

            pending_A = materialize(group, Ix (16, 16), \(r, k) ->
                # (r, k) : Ix (16, 16)
                a_row = tile_row * 16 + r
                a_col = k_tile * 16 + k

                # The launch, loop, and tile bounds prove:
                #     (a_row, a_col) : Ix (80, 48)
                # This is a physical/padded index, not an Ix (70, 37), so it
                # cannot index A directly.
                match refine(Ix (70, 37), (a_row, a_col)):
                    # a_index : Ix (70, 37)
                    # The Some branch carries proofs a_row < 70 and
                    # a_col < 37.
                    Some(a_index) -> A[a_index]
                    None          -> 0)

            pending_B = materialize(group, Ix (16, 16), \(k, c) ->
                # (k, c) : Ix (16, 16)
                b_row = k_tile * 16 + k
                b_col = tile_col * 16 + c

                # The launch, loop, and tile bounds prove:
                #     (b_row, b_col) : Ix (48, 64)
                # B requires Ix (37, 50), so the padded index must be refined
                # before it can index B.
                match refine(Ix (37, 50), (b_row, b_col)):
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
            # visible to every physical thread in `group`.

            # Every physical thread reaches the sync above. Inactive output
            # slots may compute a value, but it will not be committed to C.
            acc += dot(A_tile[local_row, :], B_tile[:, local_col])

            # Protect reads before reusing A_tile and B_tile.
            sync(group, [])

        # Producing C requires a logical Ix (70, 50), not the physical
        # slot.candidate. Exhaustive matching forces the inactive case.
        # This could be part of materialize implementation
        # on the contrary of the other guards that must be in user program
        return match slot.logical:
            # c_index : Ix (70, 50)
            Some(c_index) -> produce(c_index, acc)
            None          -> produce-nothing)

    # pending_C : pending launch_ctx (buf 3500 f32)
    return sync(launch_ctx, pending_C)
```

The zero values on out-of-bounds tile loads are padding in the physical
execution. They do not add elements to the logical matrices. The exhaustive
match on `slot.logical` makes `materialize` commit exactly one value for each
valid logical index and nothing for inactive physical slots.

Materialization uses a slot so imperfect coverage does not remove physical
participants from collective control flow:

```text
slot s = candidate-coordinate × optional (ix s)

materialize : schedule-context c p
           × space s
           × (active-context c p × slot s => produced t)
          -> pending c (buf s t)
```

For perfect coverage, every slot contains a logical index. For imperfect
coverage, inactive slots execute the producer but do not commit a value to the
result. The governing semantic rule is:

> Physical participants may be inactive with respect to logical output, but
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
| schedule context            | scoped grid and block configuration                    |
| schedule body               | caller-side orchestration                              |
| schedule-scope materialize  | kernel dispatch plus global-memory stores              |
| group                       | thread block                                           |
| warp/subgroup               | warp                                                   |
| thread/agent                | CUDA thread                                            |
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
- A sequential backend can use a one-participant context, which is useful for
  testing semantics independently of parallel code generation.

Backends need not support every topology or capability. A program requiring a
subgroup shuffle, global barrier, or group-local storage should fail capability
checking when the selected context cannot provide it.

## Relationship to existing Catena concepts

Catena already represents a logical tensor or multidimensional view as an
indexed closure and materializes it into a flat, dependently sized buffer. This proposal retains that model. It adds an explicit execution context to describe _who_ evaluates the indexed closure and a `pending` phase to describe _when_ the result is safe to observe.

The existing source-level primitive:

```text
materialize : (n : u64) × (ix n => t) -> buf n t
```
