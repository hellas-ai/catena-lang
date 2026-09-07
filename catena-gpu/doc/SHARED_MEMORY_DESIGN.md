# Shared memory and barrier safety

Shared memory is local to one GPU block. A kernel describes its synchronization
protocol as a trace of barriers. Each barrier has trusted pre- and
postconditions, so a protocol may define any phases and permission transitions
needed by the kernel.

Tiled matrix multiplication uses a simple two-phase cycle:

1. every thread writes the cells it owns;
2. after a barrier, every thread may read the shared allocation.

A second barrier returns the allocation to the write phase. This cycle is one
use of the generic mechanism, not a restriction imposed by shared memory. The
types track the declared transitions and require every thread to execute the
same barrier trace.

## Shared layouts and slots

The host describes one shared allocation as a typed list of named slots:

```rust
gpu.shared.layout(Slots)

gpu.shared.slot(
    slot-name,
    element-count,
    element-type,
    remaining-slots)
```

For example, tiled matrix multiplication uses two `u64` slots:

```text
layout =
    slot(tile-a, tile-elements, u64,
    slot(tile-b, tile-elements, u64,
    empty))
```

At runtime the layout is only the total number of shared-memory bytes. The
slot names, element types, and list structure are static.

The host layout has no allocation identity:

```rust
Layout(slots)
```

Inside the kernel, the shared allocation identity declared by the barrier
protocol is attached to that layout:

```rust
gpu.shared.layout.with-name
    : Layout(slots)
    ● shared-name
    -> shared-name : Layout(slots)
```

`shared-name` identifies the block-local allocation used by permissions. It is
not a slot name and does not affect the runtime layout or its byte size.

Inside the kernel, `gpu.shared.take` takes the first slot from the layout:

```rust
gpu.shared.take
    : shared-name : Layout(
        slot(slot-name, element-count, element-type, rest))
    ● element-count : u64
    ● thread-name : Thread(grid)
    ● block-name : Block(grid)
    -> shared-name : Layout(rest)
    ● element-count
    ● thread-name : Thread(grid)
    ● block-name : Block(grid)
    ● shared-name : Shared(
        block-name,
        slot-name,
        element-count,
        element-type)
```

The slot name prevents a handle for `tile-a` from being used as `tile-b`.
The allocation and block names prevent handles from different allocations or
blocks from being mixed.

## Launch and kernel barriers

`gpu.launch` gives every physical thread the same shared layout and the same
initial barrier trace:

```rust
gpu.launch
    : shared-layout : Layout(slots)
    ● grid : Grid(grid-shape, block-shape, global-shape)
    ● kernel-arguments
    ● kernel : (
        (shared-layout ● kernel-arguments)
        ● (thread-name : Thread(grid)
           ● |- barrier.cursor(grid, barrier-trace))
        -> kernel-result
        ● |- barrier.cursor(grid, barrier.trace.end))
    -> kernel-arguments
```

The launch wrapper calls the kernel for every thread, including threads that
do not correspond to a logical output cell. Such threads may skip the final
output write, but they must still consume the complete barrier trace.

The kernel can return only after its cursor reaches `barrier.trace.end`.
Therefore a kernel body that skips, adds, or reorders a barrier does not match
the kernel type.

## Shared-memory permissions

Writing requires ownership of one exact cell:

```rust
gpu.shared.owns(
    phase,
    thread-name,
    cell-name,
    shared-name)
```

Reading requires permission for the shared allocation:

```rust
gpu.shared.reads(
    phase,
    thread-name,
    shared-name)
```

The access operations require these proofs:

```rust
gpu.shared.write
    : thread-name : Thread(grid)
    ● block-name : Block(grid)
    ● shared-name : Shared(
        block-name, slot-name, element-count, element-type)
    ● cell-name : Cell(element-count)
    ● value : element-type
    ● |- gpu.shared.owns(
        phase, thread-name, cell-name, shared-name)
    -> thread-name ● block-name ● shared-name ● cell-name ● ownership

gpu.shared.read
    : thread-name : Thread(grid)
    ● block-name : Block(grid)
    ● shared-name : Shared(
        block-name, slot-name, element-count, element-type)
    ● cell-name : Cell(element-count)
    ● |- gpu.shared.reads(
        phase, thread-name, shared-name)
    -> thread-name ● block-name ● shared-name ● cell-name
    ● read-permission ● value
```

The permission refers to the allocation rather than an individual slot. One
ownership proof permits the thread to write its cell in several slots before
the barrier. One read proof permits reads from several slots after the
barrier. Slot handles still determine which typed region is accessed.

The host constructs a trusted block-local schedule. In the kernel,
`gpu.schedule.shared.can-own` checks that the current thread owns the selected
cell and conditionally provides the initial ownership proof:

```rust
gpu.schedule.shared.can-own
    : shared-schedule
    ● thread-name : Thread(grid)
    ● cell-name : Cell(element-count)
    -> shared-schedule
    ● thread-name
    ● cell-name
    ● decision : bool
    ● (|- decision = true ->
         gpu.shared.owns(phase, thread-name, cell-name, shared-name))
```

The current `shared.own-each` schedule assigns each block-local linear cell to
at most one thread. Consequently, two threads cannot obtain ownership of the
same cell.

## Barrier steps and synchronization

A barrier step gives a name to one trusted proof transition:

```rust
gpu.barrier.step(
    barrier-name,
    pre-proof,
    post-proof)
```

The pre- and postconditions may be any proof types required by a kernel. For
tiled matrix multiplication, the two steps happen to transition between shared
ownership and shared read permission:

```text
write-to-read:
    |- owns(write-phase, thread, cell, shared)
    -> |- reads(read-phase, thread, shared)

read-to-write:
    |- reads(read-phase, thread, shared)
    -> |- owns(write-phase, thread, cell, shared)
```

The kernel trace repeats these steps once per inner-product iteration:

```text
repeat(inner-size,
    sync(write-to-read,
    sync(read-to-write,
    end)),
end)
```

`gpu.sync` consumes exactly the first event in the cursor and applies that
event's pre/post transition:

```rust
gpu.sync
    : block
    ● |- barrier.cursor(
        grid,
        sync(step(barrier-name, pre-proof, post-proof), rest))
    ● pre-proof
    -> block
    ● |- barrier.cursor(grid, rest)
    ● post-proof
```

The generated operation is a block barrier:

```c
__syncthreads();
```

The precondition must exactly match the proof supplied by the kernel. The
postcondition is exactly the proof needed by the following phase. These
conditions are part of the barrier definition and are trusted; `gpu.sync`
does not derive or inspect them.

## Why this is safe

The model establishes two properties:

- **Race-free writes:** a trusted schedule gives each shared cell at most one
  owning thread, and `gpu.shared.write` requires that exact ownership proof.
- **Barrier safety:** every thread starts with the same trace and must consume
  all of it. A synchronization consumes one named event with its exact
  precondition and produces its declared postcondition.

This does not prove that a kernel computes the intended result. Barrier and
permission types prevent conflicting writes and divergent barrier protocols;
they do not prove matrix dimensions, selected indices, or arithmetic.
