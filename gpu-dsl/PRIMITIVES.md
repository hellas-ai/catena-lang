# GPU kernel primitives

This is a small, language-independent interface for GPU kernels. The notation
follows Catena: `●` is a product, `->` is a function, `=>` is a closure, `|-`
is for proofs, and `val(t)` is a runtime value. The last section extends the
basic global-memory interface with block-scoped shared memory and barriers.

## Supporting types

Here, I use shapes to represent index spaces. We might want them or not. Here,
they are just useful for understanding what values are intended to be.

Finite logical spaces and their indices are:

```text
space ::= d1(n) | d2(rows, cols) | d3(depth, rows, cols)

ix(d1(n))                 = ix n
ix(d2(rows, cols))        = ix rows ● ix cols
ix(d3(depth, rows, cols)) = ix depth ● ix rows ● ix cols
```

An `ix n` is an integer in `[0, n)`, so indexing is safe by construction.
`vec n t` is a vector containing exactly `n` values of type `t`.

Launch geometry is CUDA/HIP-shaped:

```text
dim3 : {
  x : u64,
  y : u64,
  z : u64
}

launch-config : {
  grid  : dim3,
  block : dim3
}

coord[d : dim3] : {
  x : ix d.x,
  y : ix d.y,
  z : ix d.z
}

thread[config : launch-config] : {
  block-index  : coord(config.grid),
  thread-index : coord(config.block)
}
```

A layout maps logical indices to cells of a flat buffer:

```text
layout[s : space, n : u64] : {
  offset : val(ix s) -> val(ix n)
}
```

## Schedule

A schedule assigns every logical output to one physical thread:

```text
schedule[config : launch-config, s : space] : {
  owner : val(ix s) -> val(thread config)
}
```

Since `owner` is a total function, every output has exactly one owner. A thread may own zero, one, or several outputs.

The schedule is only a mapping; it does not run the kernel. `gpu.launch` uses it to collect, for every physical thread, the ordered vector of indices owned by that thread.

For example, given a positive `block-size`, a linear schedule uses this
one-dimensional configuration:

```text
grid.x  = max(1, ceil-div(size, block-size))
block.x = block-size
grid.y = grid.z = block.y = block.z = 1
```

It assigns logical index `i` to:

```text
owner(i) = thread {
  block-index  = (i / block-size, 0, 0),
  thread-index = (i % block-size, 0, 0)
}
```

With this schedule, an active thread owns one logical index. Threads introduced by rounding the grid size up own no indices and receive an empty vector.

## Perfect and predicated tiling

For a two-dimensional output `d2(rows, cols)`, let `tile-rows` and
`tile-cols` be positive. CUDA's X axis follows columns and its Y axis follows
rows:

```text
block = (tile-cols, tile-rows, 1)
```

Perfect tiling applies when the tile dimensions divide the output dimensions:

```text
perfect-config :
  (rows : u64) ● (cols : u64) ●
  (tile-rows : u64) ● (tile-cols : u64)
  -> launch-config

perfect-config = \rows cols tile-rows tile-cols ->
  {
    grid  = (cols / tile-cols, rows / tile-rows, 1),
    block = (tile-cols, tile-rows, 1)
  }
```

The physical grid then covers the logical output exactly. Every physical
thread owns one output.

Predicated tiling rounds the grid up instead:

```text
predicated-config :
  (rows : u64) ● (cols : u64) ●
  (tile-rows : u64) ● (tile-cols : u64)
  -> launch-config

predicated-config = \rows cols tile-rows tile-cols ->
  {
    grid = (
      max(1, ceil-div(cols, tile-cols)),
      max(1, ceil-div(rows, tile-rows)),
      1
    ),
    block = (tile-cols, tile-rows, 1)
  }
```

Both configurations use the same schedule. For a logical output `(row, col)`:

```text
owner(row, col) = thread {
  block-index  = (col / tile-cols, row / tile-rows, 0),
  thread-index = (col % tile-cols, row % tile-rows, 0)
}
```

In a predicated launch, threads in a partial edge tile whose physical
coordinates are outside `(rows, cols)` own no logical outputs and receive an empty vector. They still run the kernel. This matters for a
shared-memory kernel: even a thread with no output must participate in every block-wide load and barrier, using guarded loads and padding out-of-bounds values when necessary.

Physical coordinates are ordinary integers, not safe buffer indices. After a bounds check, the kernel refines an in-bounds coordinate to `ix size` before reading a buffer. If the check fails, no index is constructed and the predicated load returns a padding value such as zero.

The construction applies independently along each axis, so it extends to 1D, 3D, and rectangular tiles. For tiled matrix multiplication, the inner
dimension uses the same choice: exact division for perfect tiling, or
`ceil-div(inner, tile-inner)` iterations with out-of-bounds input loads
replaced by zero for predicated tiling.

## Kernel

```text
kernel[config : launch-config, s : space, t : type] :
  forall count.
  val(thread config) ● val(vec count (ix s))
  => val(vec count t)
```

A kernel is a closure from the current physical thread and all logical outputs owned by that thread to one value for each output. A thread may receive an empty vector (i.e. inactive threads). The kernel may capture read-only input buffers.

For example, global input reads use the ordinary Catena buffer operation:

```text
ref.get[n, t] : ref n t ● val(ix n) -> val(t)
```

There is no output-store primitive in a kernel body. The launch primitive owns that operation.

## Launch

```text
gpu.launch[s, n, t] :
  (config : launch-config) ●
  buf n t ●
  layout(s, n) ●
  schedule(config, s) ●
  kernel(config, s, t)
  -> buf n t
```

`gpu.launch` collects the logical indices assigned to each physical thread and runs the kernel once per thread. It stores each returned value at the layout offset of the corresponding logical index. The updated owned output buffer is returned. Execution order between different threads is unspecified.

A kernel may also contain the shared-buffer declarations and ownership plan described below. In that case, `gpu.launch` creates those buffers once per block and supplies the initial ownership grants to each thread.

Here, we specify the layout for the output buffer. Alternatively, we can follow the approach used in previous implementations, where `launch` takes a linear buffer and a helper computes the layout before calling `launch`.

## Defining `materialize`

`materialize` evaluates an indexed producer and stores its values in a new
linear buffer:

```text
materialize[n, t] :
  (n : u64) ● (val(ix n) => val(t))
  -> buf n t
```

It uses ordinary buffer allocation and vector mapping:

```text
buf.alloc[n, t] : (n : u64) -> buf n t

vec.map[n, a, b] :
  (val(a) => val(b)) ● val(vec n a)
  -> val(vec n b)
```

We can define `materialize` entirely in terms of these operations:

```text
materialize = \n producer ->
  let config   = linear-config(n)
  let output   = buf.alloc[n, t](n)
  let layout   = linear-layout(n)
  let schedule = linear-schedule(config, n)

  let kernel = \_thread indices ->
    vec.map(producer, indices)

  gpu.launch(config, output, layout, schedule, kernel)
```

Thus `materialize` needs no additional GPU primitive beyond `gpu.launch`;
allocation and `vec.map` come from the surrounding language.

## Shared memory and barriers

Shared buffers belong to one block and are declared by the kernel. `gpu.launch` allocates a separate instance of each declared buffer for every block. A handle therefore includes the block identity, not just a source-level name:

```text
shared-ref[block, n, t]
```

Shared-memory indices have type `ix n`, so an out-of-bounds access cannot be expressed. Race freedom and initialization require additional proof tokens.

### Ownership plan

A shared-memory kernel declares a fixed assignment from logical shared cells to threads in the block:

```text
ownership-plan[config, cells] : {
  owner : val(cells) -> val(coord(config.block))
}
```

For example, a square matrix tile assigns logical cell `(row, col)` to block thread `(col, row, 0)`. A layout maps those logical cells to physical shared buffer indices and must be injective.

`gpu.launch` realizes this plan and passes the resulting initial ownership to
the kernel body. Thus a shared-memory kernel has one additional input:

```text
shared-kernel[config, s, t] :
  forall count.
  (thread : val(thread config)) ●
  (indices : val(vec count (ix s))) ●
  initial-owns(plan, thread)
  => val(vec count t)
```

For every shared buffer and logical cell assigned to this thread,
`initial-owns(plan, thread)` contains:

```text
|- owns([], thread, buffer, layout(cell))
```

`gpu.launch` may include this token only after checking that the layout is injective and `plan.owner(cell) = thread.thread-index`. There is no separate grant operation in the kernel: it receives the checked tokens before entering the loop. The ownership plan is fixed for the whole kernel; a barrier cannot replace it with another plan.

### Reads and writes

An exclusive `owns` token permits a write to one cell:

```text
shared.write :
  (buffer : shared-ref[block, n, t]) ●
  (index : val(ix n)) ●
  val(t) ●
  (|- owns(trace, thread, buffer, index))
  -> |- wrote(trace, thread, buffer, index)
```

A read needs two independent facts: the cell contains a value established by an earlier synchronized write, and the current phase permits shared reads:

```text
shared.read :
  (buffer : shared-ref[block, n, t]) ●
  (index : val(ix n)) ●
  (|- defined(trace, buffer, index)) ●
  (|- read-share(trace, thread, buffer))
  -> val(t)
```

`read-share` covers the buffer and may be reused for several reads during the same read-only phase. `defined` concerns one cell and prevents reading shared memory before some thread has initialized it.

### Block synchronization

A barrier advances the trace. Since every permission mentions its trace, all old `owns` and `read-share` tokens become unusable. The barrier then grants exactly one of these modes:

```text
grant ::= reads(buffer, ...)
        | owns(buffer, layout, cell, owner-proof, ...)
```

The modes cannot be mixed. An `owns` grant is accepted only when the fixed ownership plan assigns the cell to the current thread and the layout is injective. All threads in the block execute the same barrier and permission transition.

The barrier also makes the current thread's fact available for every thread in the block:

```text
shared.sync :
  (block : val(block)) ●
  (barrier : barrier-id) ●
  (current : |- fact(trace, thread)) ●
  grant
  -> (|- forall other in block. fact(trace, other))
     ● permissions(trace + barrier, thread, grant)
```

**Note.** I think we need some sort of admissable condition here, but not sure what it is. We cannot "lift" every fact. See Kuiper paper where they have a similar mechanism.

Consequently, a synchronized write establishes that its cell is defined after the barrier:

```text
(|- wrote(trace, writer, buffer, index))
-> |- defined(trace + barrier, buffer, index)
```

For tiled matrix multiplication, one loop iteration has the following shape:

```text
own-a ● own-b = initial ownership supplied by launch

for each tile:
  wrote-a = shared.write(tile-a, thread-cell, value-a, own-a)
  wrote-b = shared.write(tile-b, thread-cell, value-b, own-b)

  all-writes ● (read-a ● read-b) =
    shared.sync(block, tile-loaded, wrote-a ● wrote-b,
                reads(tile-a, tile-b))

  for each k in tile:
    defined-a = all-writes(writer-of-a(k)).a
    defined-b = all-writes(writer-of-b(k)).b
    a = shared.read(tile-a, (thread-row, k), defined-a, read-a)
    b = shared.read(tile-b, (k, thread-col), defined-b, read-b)

  _ ● (own-a ● own-b) =
    shared.sync(block, tile-consumed, unit,
                owns(tile-a, tile-b, plan))
```

The first barrier publishes initialization and changes the block from exclusive writing to shared reading. The second barrier waits until all reads are finished, resets the read permissions, and grants exclusive ownership for the next tile. Thus a shared cell is never writable while another thread has permission to read it.
