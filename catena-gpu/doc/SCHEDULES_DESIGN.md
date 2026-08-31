# Conflicting writes and ownership schedules

When concurrent threads write global memory, we must ensure that two threads
cannot write the same buffer cell.

Catena GPU associates every writable buffer with an ownership schedule. A
schedule determines whether the current thread owns a particular cell. The
implemented schedules are race-free by construction: each cell can resolve to
at most one owning thread.

The current policies are:

- `gpu.scheduling.own-each`: 1D cell `i` is owned by global thread `i`;
- `gpu.scheduling.2d.row-major.own-each`: matrix cell `(row, column)` is owned
  by grid thread `(column, row)`.

A launch may contain zero, one, or several `cap.own` buffers. Each writable
buffer carries its own schedule in the generic kernel arguments. Read-only
`cap.ref` buffers need no schedule because concurrent reads do not conflict.

Conceptually, a schedule defines a relation:

```text
assigned(schedule, thread, cell)
```

Race freedom requires that one cell is not assigned to two distinct threads:

```text
assigned(schedule, thread-a, cell)
and assigned(schedule, thread-b, cell)
implies thread-a = thread-b
```

The current implementation does not expose generic `assigned` or `injective`
proof primitives. Instead, it provides two trusted schedule constructors whose
relations have this property, and `gpu.scheduling.can-own` decides membership
in the selected relation at runtime.

## Type-level identities

The schedule and ownership proposition are declared as:

```rust
gpu.scheduling
    : BufferName
    -> BufferSize
    -> GridTopology
    -> ScheduleName
    -> Type

gpu.schedule.owns
    : BufferName
    -> ScheduleName
    -> ThreadName
    -> CellName
    -> Prop
```

Consequently, an ownership proof is specific to:

- one allocation;
- one schedule;
- the currently executing thread;
- one bounded buffer cell.

A proof for one buffer, schedule, thread, or cell cannot be used for another.

## The 1D ownership schedule

The 1D policy assigns cell `i` to global thread `i`:

```text
cell 0 -> thread 0
cell 1 -> thread 1
cell 2 -> thread 2
...
```

Its interface is:

```rust
gpu.scheduling.own-each
    : buffer-name                                      // allocation identity
    ● grid-shape : shape.1d(grid-count)
    ● block-shape : shape.1d(block-count)
    ● global-shape : shape.1d(global-count)
    ● |- buffer-size <= global-size                   // every cell has a thread
    -> schedule-name
    × gpu.scheduling(
        buffer-name,
        buffer-size,
        grid(grid-shape, block-shape, global-shape),
        schedule-name)
```

Most inputs are static or proofs and are erased. Code generation creates the
runtime schedule value:

```c
schedule = { CATENA_SCHEDULING_OWN_EACH, 0 };
```

Excess threads are allowed. They enter the kernel but cannot construct a
bounded output cell when their global index is outside the buffer.

## The 2D row-major ownership schedule

For a matrix with `rows × columns` cells, the policy uses:

```text
cell(row, column) = row * columns + column
owner(row, column) = thread(column, row)
```

Its interface is:

```rust
gpu.scheduling.2d.row-major.own-each
    : buffer-name
    ● buffer-size
    ● matrix-columns : u64
    ● grid-shape : shape.2d(grid-columns, grid-rows)
    ● block-shape : shape.2d(block-columns, block-rows)
    ● global-shape : shape.2d(global-columns, global-rows)
    -> schedule-name
    × gpu.scheduling(
        buffer-name,
        buffer-size,
        grid(grid-shape, block-shape, global-shape),
        schedule-name)
```

At runtime, only the policy kind and logical matrix width are stored:

```c
schedule = {
    CATENA_SCHEDULING_2D_ROW_MAJOR_OWN,
    matrix_columns
};
```

The grid may be smaller than the matrix. That is race-free but functionally
incomplete because some cells will not be processed.

## Querying ownership

The kernel chooses a bounded buffer cell and asks whether the current thread
owns it:

```rust
gpu.scheduling.can-own
    : schedule : gpu.scheduling(                         // runtime policy
        buffer-name, buffer-size, grid, schedule-name)
    ● thread : Named(thread-name, gpu.thread(grid))      // executing thread
    ● cell : ix(shape.1d(buffer-size))                   // bounded candidate
    -> schedule                                          // returned unchanged
    × thread                                            // returned unchanged
    × cell                                              // returned, now named
    × decision : bool                                   // runtime membership
    × (|- decision = true ->
         gpu.schedule.owns(
             buffer-name,
             schedule-name,
             thread-name,
             cell-name))
    × (|- gpu.schedule.owns(
             buffer-name,
             schedule-name,
             thread-name,
             cell-name)
         -> decision != false)
```

The Boolean is a runtime result. The implications and ownership proposition are
proof terms and are erased.

Code generation resolves the selected schedule against the actual executing
thread and cell:

```c
decision =
    catena_scheduling_resolve(schedule, thread, cell)
    == CATENA_CELL_OWNED;
```

For `own-each`, the resolver checks:

```c
thread.global_index.first == cell.first
```

For the 2D row-major policy, it checks:

```c
thread.global_index.first  == cell.first % matrix_columns
&&
thread.global_index.second == cell.first / matrix_columns
```

If `matrix_columns` is zero, the resolver returns inaccessible rather than
performing division or remainder.

The kernel obtains the concrete ownership proof only on the true path. In the
current definitions this is commonly written using `assert-then` after the
kernel has already selected an in-bounds cell consistent with the schedule.
If the kernel selects an in-bounds cell that its thread does not own,
`can-own` is false and that assertion traps. Predicated threads avoid the
assertion by taking the outer false branch.

## Writing global memory

The write primitive requires a bounded index and an ownership proof for the
exact thread, buffer, schedule, and cell:

```rust
gpu.global.write
    : thread : Named(thread-name, gpu.thread(grid))
    ● buffer : Named(
        buffer-name,
        gpu.global(cap.own, buffer-size, element-type))  // consumed resource
    ● cell : Named(                                     // bounded destination
        cell-name,
        ix(shape.1d(buffer-size)))
    ● value : element-type                              // value to store
    ● |- gpu.schedule.owns(
        buffer-name,
        schedule-name,
        thread-name,
        cell-name)                                      // exact permission
    -> ()                                               // current Hex interface
```

The proof is erased. The generated operation receives four runtime inputs and
writes directly through the linear global pointer:

```c
buffer[cell.first] = value;
```

The current Hex primitive has no output. It mutates the allocation contents;
it does not change the stable allocation name used by the schedule and proof.
The enclosing launch keeps the original buffer handle in its kernel arguments
and returns those arguments after synchronization.

## Reading borrowed global memory

A borrowed buffer is read-only, so a bounded index is sufficient:

```rust
gpu.global.read
    : buffer : Named(
        buffer-name,
        gpu.global(cap.ref, buffer-size, element-type))  // read-only handle
    ● cell : ix(shape.1d(buffer-size))                   // bounded source
    -> buffer                                           // returned unchanged
    × value : element-type                              // loaded value
```

No schedule or per-thread read proof is needed. The capability system prevents
kernel code from passing a `cap.ref` buffer to `gpu.global.write`.

## Generic kernel launch

`gpu.launch` does not prescribe the contents of the kernel arguments. Their
product can contain multiple owned buffers and schedules, borrowed buffers,
lengths, matrix dimensions, and scalar values:

```rust
gpu.launch
    : grid : gpu.grid(grid-shape, block-shape, global-shape)
    ● kernel-arguments
    ● kernel : (
        kernel-arguments
        × gpu.thread(grid)
        -> kernel-result)
    -> kernel-arguments
```

For example, a kernel with two writable buffers can receive:

```text
(
    output-a,
    output-a-schedule,
    output-b,
    output-b-schedule,
    read-only-input,
    dimensions
)
```

The schedules remain associated with their buffers through the static buffer
and schedule names in their types.

Code generation launches one backend thread for every physical grid position.
The wrapper creates the complete thread value and calls the Hex kernel
unconditionally:

```c
global_index = {
    blockIdx.x * blockDim.x + threadIdx.x,
    blockIdx.y * blockDim.y + threadIdx.y,
    blockIdx.z * blockDim.z + threadIdx.z
};

thread = {
    global_index,
    in_block_index,
    block_index
};

kernel(kernel_arguments..., thread);
```

After the launch, the host synchronizes and returns the kernel arguments
unchanged.

## Predicated threads

Predication happens inside the Hex kernel, not in the launch wrapper.

For a 1D output, the kernel obtains its global index and checks it against the
buffer length:

```text
grid-index = gpu.thread.in-grid.index(thread)
candidate = ix.1d.value(grid-index)
decision = candidate < buffer-size
```

It then uses `bool.ifc`:

```text
if decision {
    cell = u64.to-ix(candidate, buffer-size, candidate-is-in-bounds)
    can-own, owns-if-allowed =
        gpu.scheduling.can-own(schedule, thread, cell)
    owns = assert-then(can-own, owns-if-allowed)
    gpu.global.write(thread, buffer, cell, value, owns)
} else {
    # This physical thread has no output cell.
    # Return the schedule and branch state unchanged.
}
```

For a 2D matrix, the kernel separately checks:

```text
thread-column < matrix-columns
thread-row < matrix-rows
```

Only the true branch constructs the row-major bounded cell and performs the
ownership query. Threads outside either logical dimension enter the kernel and
take the false branch.

For example, a two-row, three-column matrix may be launched with four physical
threads per row:

```text
(0,0) (1,0) (2,0) (3,0)
(0,1) (1,1) (2,1) (3,1)
```

Threads `(3,0)` and `(3,1)` are predicated out inside the kernel. Thread
`(0,1)` selects linear cell:

```text
1 * 3 + 0 = 3
```

The 2D schedule agrees that cell `3` belongs to thread `(0,1)`. It does not use
the padded physical width `4` when resolving ownership.

## Why the schedules are race-free

For `own-each`, each linear cell has one possible owner with the same global
thread index.

For the 2D policy, division and remainder recover one `(column, row)` pair from
each linear cell using the fixed logical column count. Therefore one cell also
has at most one possible owner.

Because `gpu.global.write` requires the corresponding ownership proof, two
different concurrent threads cannot both write the same scheduled cell.
Separate owned buffers have separate allocation and schedule names, so their
proofs cannot be exchanged.

## Safety is not functional correctness

The schedule and permission types establish access safety. They do not prove
that the kernel implements the intended algorithm.

A race-free kernel may still:

- use a grid that does not cover all output cells;
- predicate every thread out because it computed the wrong bound;
- select an in-bounds but unintended cell;
- skip a permitted write;
- write the wrong value;
- read the wrong in-bounds cells from a borrowed buffer.

The runtime `can-own` decision protects memory access. It does not establish
functional correctness or totality of the kernel.
