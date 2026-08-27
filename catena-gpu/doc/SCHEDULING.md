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

`gpu.scheduling.2d.row-major.own-each` is needed for predicated 2D launches.

Suppose the matrix has 2 rows and 3 columns:

```text
matrix coordinates and linear cells

(0,0)=0  (1,0)=1  (2,0)=2
(0,1)=3  (1,1)=4  (2,1)=5
```

We launch a grid with 4 threads per row, perhaps because of the block size:

```text
thread coordinates and flattened thread IDs

(0,0)=0  (1,0)=1  (2,0)=2  (3,0)=3
(0,1)=4  (1,1)=5  (2,1)=6  (3,1)=7
```

Threads with `x = 3` are outside the matrix, so the kernel predicates them out.

The kernel wants thread `(0,1)` to process matrix cell `(0,1)`, which has linear buffer index:

```text
y * matrix_width + x
= 1 * 3 + 0
= 3
```

But the 1D `own-each` schedule uses the flattened thread ID. For thread
`(0,1)`, that is:

```text
y * grid_width + x
= 1 * 4 + 0
= 4
```

Therefore, the 1D schedule says that thread `(0,1)` owns cell `4`, while the kernel wants it to write cell `3`. Its `can-own` query for cell `3` fails.

According to the 1D schedule, cell `3` belongs to thread `(3,0)`. But `(3,0)` is an excess thread and is predicated out, so cell `3` is never written.

The 2D schedule avoids flattening the thread with the padded grid width. It directly establishes that thread `(x,y)` owns matrix cell `(x,y)`. Thus thread `(0,1)` owns cell `1 * 3 + 0 = 3`, and threads with `x = 3` own no matrix cell.

For perfect tiling, the matrix width and grid width are both `3`, so the 1D and 2D formulas happen to agree. The 2D policy becomes necessary when predication
adds padding within each row.

### Race free by construction

Both policies are race-free by construction. Kernel code still checks bounds before constructing an `ix` and queries `can-own` before every write.
