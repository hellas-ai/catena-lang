# Launch, scheduling, and permissions

A global memory is a buffer with a number of cells. A thread can own or read from some cells in global memory, but we want to avoid race conditions. This means that when we launch the kernel we need to specify how permissions are distributed over threads in a grid.

This distribution is called a "scheduling". For each global memory in input for a kernel function, we specify what a thread can do for each cell. Informally, we can see a schedule as a function from buffer indices to thread indices.

A scheduling is an opaque runtime value. Schedules are ordinary kernel arguments and can only be queried through `gpu.scheduling.can-own` and `gpu.scheduling.can-read`. Both operations use one backend resolver. The proofs produced by these queries are erased.

`gpu.launch` is generic: it receives a grid, one product of kernel arguments, and a specialized kernel. The kernel declares the product it expects. The current example builds one destination triple and one source triple with `*.intro`, then combines them:

```text
(destination-length, destination-buffer, destination-schedule)
(source-length, source-buffer, source-schedule)
```

The types connect each triple. If `b` names a global buffer of length `n`, its schedule is indexed by the same `b` and `n`. A schedule or bounded index for one buffer therefore cannot be used with another buffer.

A race-free condition is: for each cell, either only one thread owns it or some threads can read it.

## What sort of proofs do we want? Safety is not functional correctness

The types guarantee **race-free and safe access**, not that the kernel computes the intended mapping or result. A bad in-bounds candidate can fail the ownership assertion, and a thread can write the wrong value to a cell it legitimately owns. These remain race-free, but the program may fail or produce the wrong result.

Race free and safe access is the guarnatee we need in order to prove that the kernel function is deterministic and total.

## Own-each scheduling

For convenience, `gpu.scheduling.own-each` constructs a schedule that gives each buffer cell one owner. Common scheduling disciplines as primitives avoid repeating tedious race-freedom proofs.

Specifically, `gpu.scheduling.own-each` assigns buffer cell `i` to global thread `i`. The owner may read and write that cell. No other thread receives permission for it, so there cannot be two writers or a writer and a distinct reader.

The constructor requires `buffer-size <= global-size`. This ensures that every buffer cell has a corresponding global thread. If the grid is larger, excess threads own no cell. Own-each scheduling is therefore race-free by construction; the internal race-free argument is part of this fixed primitive rather than a proof rebuilt by each program. The schedule and every permission carry the buffer identity, preventing a grant for one buffer from being used with another.

`gpu.scheduling.read-all` is the complementary read-only policy: every thread may read every cell, and no thread may own or write one. It is race-free because concurrent reads do not conflict. A kernel can therefore use an own-each schedule for its destination buffer and a read-all schedule for a separate source buffer.

## Access inside a kernel

The kernel computes a candidate cell and calls `gpu.scheduling.can-own` or `gpu.scheduling.can-read` with the actual `gpu.thread` handle and a bounded `ix[shape.1d(buffer-size)]`.

In the kernel, `gpu.thread.in-grid.index` obtains the current coordinate and `ix.to-u64` computes its row-major linear offset. `u64.lt` then checks the offset against the buffer size. The true branch uses `u64.to-ix` to construct a bounded cell before calling `can-own`; the false branch skips without asking for permission.
The entry receives `grid-x` and `block-x`, so the same program covers exact tiling when `grid-x * block-x == buffer-size` and predication when the product is larger. If it is smaller, schedule construction fails its fit assertion.
