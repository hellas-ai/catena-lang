# Launch, scheduling, and permissions

A grid has a rank-specific block-grid shape, block shape, derived global-thread
shape, and total thread count. Thread coordinates are ordinary `ix[shape]`
values. A schedule represents an owner using the thread's canonical in-grid
index. The grid in the surrounding types prevents indices from unrelated grids
from being mixed.

## Why scheduling is race-free

`gpu.scheduling.linear` assigns buffer cell `i` to global thread `i`. The owner
may read and write that cell. No other thread receives permission for it, so
there cannot be two writers or a writer and a distinct reader.

The constructor requires `buffer-size <= global-size`. This ensures that every
buffer cell has a corresponding global thread. If the grid is larger, excess
threads own no cell. Linear scheduling is therefore race-free by construction;
the internal race-free argument is part of this fixed primitive rather than a
proof rebuilt by each program. The schedule and every permission carry the
buffer identity, preventing a grant for one buffer from being used with
another.

## Access inside a kernel

The kernel computes a candidate cell and calls `gpu.scheduling.can-own` or
`gpu.scheduling.can-read` with the actual `gpu.thread` handle and a bounded
`ix[shape.1d(buffer-size)]`. A raw offset cannot be used for a permission query.
The decision operation obtains the thread's in-grid index internally; an
arbitrary coordinate cannot be supplied to impersonate another thread. Its
result keeps the Boolean, its true and false equalities, and the permission
implications explicit. For an in-bounds cell, the examples assert ownership,
grant it, and access memory.

The parameterized linear example maps each bounded buffer index to the same
bounded linear in-grid index. Before constructing the schedule, `u64.lte`
checks that the buffer fits in the grid; ordinary `assert` and `ax-mp` turn its
true result into the proof required by `gpu.scheduling.linear`.
In the kernel,
`gpu.thread.in-grid.index` obtains the current coordinate and `ix.to-u64`
computes its row-major linear offset. `u64.lt` then checks the offset against
the buffer size. The true branch uses `u64.to-ix` to construct a bounded cell
before calling `can-own`; the false branch skips without asking for permission.
The entry receives `grid-x` and `block-x`, so the same program covers exact
tiling when `grid-x * block-x == buffer-size` and predication when the product is
larger. If it is smaller, schedule construction fails its fit assertion.

## Safety is not functional correctness

The types guarantee race-free access, not that the kernel computes the intended
mapping or result. A bad in-bounds candidate can fail the ownership assertion,
and a thread can write the wrong value to a cell it legitimately owns. These
remain race-free, but the program may fail or produce the wrong result.
