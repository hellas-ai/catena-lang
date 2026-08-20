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

The schedule is only a mapping; it does not run the kernel. An ownership access
plan uses it to give every physical thread the complete list of cells that
thread owns.

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

With this schedule, an active thread owns one logical index. Threads introduced
by rounding the grid size up own no cells.

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

A kernel accepts arbitrary inputs together with permissions tied to the actual
resources inside those inputs:

```text
kernel[config : launch-config, inputs : type] :
  val(inputs) ●
  val(thread config) ●
  permissions(inputs, thread)
  => unit
```

The kernel also declares a closed access plan. A read grant identifies a
global buffer that the thread may read. An ownership grant gives the thread
the complete list of cells assigned to it by an owner function:

```text
global-access ::= reads(buffer, region)
                | owns(buffer, layout, owner)
                | global-access ● global-access
```

Global operations require the resulting permissions:

```text
global.read :
  buffer ● val(ix n) ● (|- read-share(thread, buffer, region))
  ● (|- index in region)
  -> val(t)

global.write :
  buffer ● val(ix n) ● val(t) ● (|- owns(thread, buffer, index))
  -> unit
```

An output buffer is therefore an ordinary field of `inputs`; it is not a
special kernel result.

## Launch

```text
gpu.launch[inputs] :
  (config : launch-config) ●
  val(inputs) ●
  kernel(config, inputs)
  -> launch
```

`gpu.launch` interprets the kernel's access plan and runs the kernel once per
physical thread. Read and ownership permissions refer to the exact buffers in
`inputs`. One thread may own zero, one, or several cells. Execution order
between different threads is unspecified.

A kernel may also contain the shared-buffer declarations and ownership plan described below. In that case, `gpu.launch` creates those buffers once per block and supplies the initial ownership grants to each thread.

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

We can define `materialize` using an output buffer in the kernel inputs and an
ownership plan over its cells:

```text
materialize = \n producer ->
  let config   = linear-config(n)
  let output   = buf.alloc[n, t](n)
  let layout   = linear-layout(n)
  let inputs = { output }

  let kernel =
    global-access owns(inputs.output, layout, linear-owner(config, n))

    \inputs _thread owned-cells ->
      for index in owned-cells:
        global.write(inputs.output, index, producer(index), owned-cells[index])

  gpu.launch(config, inputs, kernel)
  output
```

Thus `materialize` needs no output-specific launch primitive. Allocation is
provided by the surrounding language, and the kernel writes only the cells
granted by its global access plan.

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
shared-kernel[config, inputs] :
  (inputs : val(inputs)) ●
  (thread : val(thread config)) ●
  global-permissions(inputs, thread) ●
  initial-owns(plan, thread)
  => unit
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

The matmul tile loop carries its proof invariant separately from its partial
sum. The invariant type and its initial value can be written directly on the
loop. `initial-own-a` and `initial-own-b` are supplied by `gpu.launch`:

```text
for each tile
  maintaining invariant :
    (|- owns(trace, thread, tile-a, thread-cell)) ●
    (|- owns(trace, thread, tile-b, thread-cell))
  initially initial-own-a ● initial-own-b:

  own-a ● own-b = invariant

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

  _ ● next-invariant =
    shared.sync(block, tile-consumed, unit,
                owns(tile-a, tile-b, plan))

  continue with next-invariant
```

`maintaining` means that the loop may start only with the two ownership tokens
given by `initially`, and that its body must re-establish the same two `owns`
facts at `trace + tile-loaded + tile-consumed` before continuing. Temporary
`wrote`, `defined`, and `read-share` facts are used inside one iteration but
are not part of `next-invariant`.

The first barrier publishes initialization and changes the block from exclusive writing to shared reading. The second barrier waits until all reads are finished, resets the read permissions, and grants exclusive ownership for the next tile. Thus a shared cell is never writable while another thread has permission to read it.
