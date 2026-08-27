# Launch, scheduling, and permissions

A global memory is a buffer with a number of cells. Owned buffers need a scheduling that determines which thread may write each cell. Borrowed `cap.ref` buffers are read-only and need no scheduling.

Informally, an ownership schedule is a function from buffer indices to thread indices.

A scheduling is currently an opaque runtime value passed with the owned buffer and queried through `gpu.scheduling.can-own`. The ownership proof produced by the query is erased.

`gpu.launch` is generic: it receives a grid, one product of kernel arguments, and a specialized kernel. The current examples pass an owned destination with its schedule and borrowed inputs without schedules:

```text
(destination-length, destination-buffer, destination-schedule)
(source-length, source-buffer)
```

If `b` names an owned global buffer of length `n`, its schedule is indexed by the same `b` and `n`. A schedule or bounded index for one buffer therefore cannot be used with another buffer.

Names in these types are static identities, not runtime objects. For example,
`destination-buffer-name` names `destination-buffer`, while
`destination-schedule-name` names `destination-scheduling`. The same convention
is used for borrowed buffers, threads, and blocks.

An allocation is either owned, with at most one scheduled writer per cell, or borrowed through read-only references. The capability types prevent a borrowed input from being written.

## What sort of proofs do we want? Safety is not functional correctness

The types guarantee **race-free and safe access**, not that the kernel computes the intended mapping or result. A bad in-bounds candidate can fail the ownership assertion, and a thread can write the wrong value to a cell it legitimately owns. These remain race-free, but the program may fail or produce the wrong result.

## Own-each scheduling

For convenience, `gpu.scheduling.own-each` constructs a schedule that gives each buffer cell one owner. Common scheduling disciplines as primitives avoid repeating tedious race-freedom proofs.

Specifically, `gpu.scheduling.own-each` assigns buffer cell `i` to global thread `i` for a 1D launch. Only the owner may write that cell, so there cannot be two writers. Reads are available only through separate `cap.ref` buffers.

The constructor requires `buffer-size <= global-size`. This ensures that every buffer cell has a corresponding global thread. If the grid is larger, excess threads own no cell. Own-each scheduling is therefore race-free by construction; the internal race-free argument is part of this fixed primitive rather than a proof rebuilt by each program. The schedule and every permission carry the buffer identity, preventing a grant for one buffer from being used with another.

The corresponding matrix policy is `gpu.scheduling.2d.row-major.own-each`. Read-only inputs use `cap.ref` instead of a read schedule.

## Access inside a kernel

The kernel computes a candidate destination cell and calls `gpu.scheduling.can-own` with the actual `gpu.thread` handle and a bounded `ix[shape.1d(buffer-size)]`.

In the kernel, `gpu.thread.in-grid.index` obtains the current coordinate and `ix.to-u64` computes its row-major linear offset. `u64.lt` then checks the offset against the buffer size. The true branch uses `u64.to-ix` to construct a bounded cell before calling `can-own`; the false branch skips without asking for permission.
The entry receives `grid-x` and `block-x`, so the same program covers exact tiling when `grid-x * block-x == buffer-size` and predication when the product is larger. If it is smaller, schedule construction fails its fit assertion.

The example uses the finite `fold` primitive to visit every bounded source
index. Its direct named body carries only the borrowed source buffer and
running sum. A bounded index may read the `cap.ref` buffer directly, without a
thread, schedule, decision, or read grant.
