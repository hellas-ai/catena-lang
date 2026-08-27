# Scheduling and memory capabilities

Catena GPU uses capabilities and ownership schedules to prevent data races. This establishes access safety, not that a kernel computes the intended result.

## Safety is not functional correctness

The type system controls which threads may access each global-memory cell. It does not prove that a kernel selects the right cells, writes the right values,
or processes every cell.

For example, a kernel can safely:

- skip some cells;
- compute the wrong value for a cell it owns;
- compute a cell that it does not own and take the false permission branch;
- launch excess threads that do no work.

These programs may be functionally wrong, but they do not perform an unauthorized memory access. If a kernel asserts that a failed permission query must be true, it may also fail at runtime. **Scheduling guarantees race freedom, not totality.**

## Borrowed buffers: `cap.ref`

A `cap.ref` global buffer is read-only. Any number of threads may read the same cell concurrently because read-read concurrency is not a data race. Code using
the reference cannot call `gpu.global.write`, so no per-thread read permission or read schedule is required.

`gpu.global.read` therefore needs only the borrowed buffer and a bounded index:

```text
buffer : gpu.global(cap.ref, buffer-size, element-type)
cell   : ix(shape.1d(buffer-size))

gpu.global.read(buffer, cell)
```

The capability discipline assumes that the allocation is not also exposed for concurrent mutation while it is borrowed. A `cap.ref` passed to a kernel is not
returned as an owned result; ownership remains with the caller.

## Owned buffers: `cap.own`

A `cap.own` global buffer may be written, so a bounded index alone is not enough. The kernel must also prove that the current thread owns that cell under
the schedule chosen for the buffer.

Informally, an ownership schedule maps each writable buffer cell to at most one thread. This rules out concurrent writers by construction. Other threads cannot obtain an ownership proof for that cell, and reads of mutable buffers are not available without ownership.

Each owned buffer has its own schedule. Schedules stay in the generic kernel arguments beside their buffers rather than being a distinguished argument of `gpu.launch`. This supports kernels with zero, one, or several owned buffers, possibly using different policies:

```text
(output-a, output-a-schedule)
(output-b, output-b-schedule)
(read-only-input)
```

The schedule type carries the identities of its buffer, grid, and scheduling instance. Consequently, a schedule or ownership proof for one buffer cannot be
used to write another buffer.

## Why scheduling is opaque

The kernel computes its destination cell inside its body. The corresponding ownership proposition depends on that computed cell:

```text
|- gpu.schedule.owns(buffer, schedule, thread, cell)
```

Passing this proof directly as a kernel argument would require the cell to be known before the kernel runs. Alternatively, the kernel would need a dependent
proof function that can produce ownership evidence for a cell selected later. Metacat does not currently express that function cleanly.

The schedule is therefore an opaque runtime value with a typed elimination operation. After computing a bounded cell, the kernel queries the schedule:

```text
decision, owns-if-true =
    gpu.scheduling.can-own(schedule, thread, cell)
```

The true branch turns `owns-if-true` into the concrete ownership proof. The proof is passed directly to `gpu.global.write` and is erased during code generation:

```text
owns = assert(decision, owns-if-true)
gpu.global.write(thread, buffer, cell, value, owns)
```

The schedule itself remains at runtime because `can-own` must resolve the policy selected for that particular buffer. Its construction is restricted to
known race-free policies.

## Current policies

`gpu.scheduling.own-each` assigns 1D buffer cell `i` to global thread `i`. It requires `buffer-size <= global-size`; excess threads own no cell.

`gpu.scheduling.2d.row-major.own-each` applies the same idea to a row-major 2D buffer and a 2D grid.

Both policies are race-free by construction. Kernel code still checks bounds before constructing an `ix` and queries `can-own` before every write.
