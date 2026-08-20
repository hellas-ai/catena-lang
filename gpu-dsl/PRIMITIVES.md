# GPU kernel primitives

This note describes a small, language-independent interface for GPU kernels.
It focuses on memory safety and race freedom rather than execution details or
Lean encodings.

The examples use Catena-like notation:

```text
●    product of values or resources
->   function
=>   closure
|- p proof of proposition p
```

Permission proofs are linear resources: they cannot be copied or silently
retained after an operation consumes them. Other proofs, such as bounds or
initialization facts, may be reusable.

## Indices and buffers

An `ix n` is an integer in `[0, n)`. Buffer operations use bounded indices, so
out-of-bounds accesses cannot be expressed.

```text
global.buffer[n, t]
```

A global buffer identifies a global-memory resource. A cell is identified by
its buffer and offset.

A layout maps logical indices to physical offsets:

```text
layout[logical, n] : {
  offset    : logical -> ix n
  injective : |- offset(i) = offset(j) -> i = j
}
```

## Threads

A launch configuration contains a grid size and a block size. A thread carries
both its block coordinate and its local coordinate:

```text
block[config] : {
  index : coord(config.grid)
}

thread[config] : {
  block-index  : coord(config.grid)
  thread-index : coord(config.block)
}

block : (thread : thread[config]) -> block[config]
```

## Global memory

A kernel takes arbitrary inputs. It separately declares the parts of the
global buffers in those inputs that each thread may read or write:

```text
access {
  read  A in read-a(thread)
  read  B in read-b(thread)
  write C in write-c(thread)
}
```

The write region directly describes the output cells owned by a thread. A
thread may own zero, one, or several cells, so predicated launches can contain
extra threads with an empty write region.

Regions belong to the access declaration only. Launch expands them into
permissions for individual cells:

```text
index in read-a(thread)  -> (|- read(thread, A, index))
index in read-b(thread)  -> (|- read(thread, B, index))
index in write-c(thread) -> (|- own(thread, C, index))
```

This region notation is shorthand. A kernel may state the same contract
directly with dependent proof parameters, as in the matmul example below.

The access declaration is valid when, for every physical global-memory cell:

```text
two writers to the same cell are the same thread
a written cell has no readers
```

Reads may overlap, and accesses to different buffers or different offsets do
not conflict. Regions can therefore be as precise as one cell per thread; they
do not need to cover a whole buffer.

Global operations use those cell permissions directly:

```text
global.read[n, t] :
  (buffer : global.buffer[n, t]) ● (index : ix n)
  ● (|- read(thread, buffer, index))
  -> value ● (|- read(thread, buffer, index))

global.write[n, t] :
  (buffer : global.buffer[n, t]) ● (index : ix n) ● value
  ● (|- own(thread, buffer, index))
  -> (|- own(thread, buffer, index))
```

Reading preserves the cell's read permission. Writing preserves its exclusive
ownership, so the owning thread may update the cell more than once.

## Kernels and launch

A kernel is a dependent function. Resource arguments are ordinary values, and
proof arguments may refer to the exact resources introduced earlier in the
function type. For example:

```text
kernel-body[config, n, t, index] :
  (inputs : inputs)
  -> (thread : thread[config])
  -> (tile : shared.buffer[block thread, n, t])
  -> (permission : |- own(thread, tile, index(thread)))
  => unit
```

The `tile` binder behaves like any other value binder. Its type tells launch
to create one buffer per block. The later `permission` binder refers to that
particular buffer value.

The proof parameters form the kernel's access contract. Launch checks that
the requirements for all threads are race-free, supplies global buffers from
the kernel inputs, creates block-local shared buffers, constructs the permitted
proofs, and applies the function once per physical thread:

```text
gpu.launch : config ● inputs ● kernel-function(inputs) -> launch
```

## Shared memory

A shared buffer belongs to one block. Launch creates a separate instance for
every block:

```text
shared.buffer[block, n, t]
```

Shared operations receive the block as a runtime argument. The dependent
buffer type ensures that the buffer belongs to that same block.

The kernel function expresses initial ownership with dependent proof
parameters. For example:

```text
kernel-body[config, n, t, index] :
  (thread : thread[config])
  -> (tile : shared.buffer[block thread, n, t])
  -> (initial-own : |- own(thread, tile, index(thread)))
  => unit
```

Taken across all threads, these requirements are the ownership plan. Launch
constructs the proofs only after checking that two different threads are not
given ownership of the same cell. A kernel can use several proof parameters
when a thread initially owns several cells.

### Shared writes

Exclusive ownership permits one thread to write a cell:

```text
shared.write[config, n, t] :
  (block : block[config])
  ● (buffer : shared.buffer[block, n, t])
  ● (index : ix n) ● value
  ● (|- own(thread, buffer, index))
  -> (|- wrote(thread, buffer, index))
```

The write consumes ownership and produces evidence that this thread wrote the
cell. No other thread can read or write it while the exclusive permission is
held.

### Shared reads

A shared read needs both permission to read the cell and evidence that it was
initialized:

```text
shared.read[config, n, t] :
  (block : block[config])
  ● (buffer : shared.buffer[block, n, t])
  ● (index : ix n)
  ● (|- read(thread, buffer, index))
  ● (|- defined(buffer, index))
  -> value ● (|- read(thread, buffer, index))
```

The cell's read permission is reusable during the current read phase.
`defined` is a pure fact derived from writes published at a barrier.

## Block synchronization

There is one block synchronization primitive:

```text
shared.sync :
  current-shared-context
  ● block
  ● barrier
  ● contribution
  ● grants
  -> published-facts ● new-shared-context(grants)
```

`current-shared-context` means all shared permissions currently held by the
thread, not a caller-selected subset.

It has three effects:

1. Every thread in `block` waits at the same barrier.
2. The barrier publishes its fixed contribution from every thread.
3. It discards the entire old shared-permission context and replaces it with
   `grants`.

The third rule is important. A thread cannot preserve an old permission merely
by omitting it from the arguments to `shared.sync`. After the barrier, only
the new shared permissions returned by the barrier are usable. Global-memory
permissions are unaffected.

The new grants are declared at the barrier site:

```text
grants ::= reads(buffers)
         | owns(required cells)
```

A barrier grants exclusive ownership only according to the kernel's checked
ownership requirements. Read and exclusive-write modes are not mixed for the
same cell. A `reads` grant produces a cell-level read permission for every
shared cell covered by that barrier.

The contribution is also fixed by the barrier site; it is not an arbitrary
proposition chosen by the caller. For example, a tile-loaded barrier may
publish shared writes:

```text
tile-loaded contribution(thread) =
  (|- wrote(thread, tile-a, cell-a(thread)))
  ● (|- wrote(thread, tile-b, cell-b(thread)))
```

After every thread arrives, each thread receives the corresponding facts for
all threads, together with read permissions for both tiles. From those
published writes it can prove that every planned tile cell is defined.

A later tile-consumed barrier need not publish a fact. Its synchronization
ensures that all threads have finished the read phase; replacing the shared
context removes all read permissions, and the barrier grants ownership for
the next write phase.

An implementation can enforce whole-context replacement with a hidden fresh
phase attached to shared permissions, or with an indexed effect. The phase is
not part of kernel source syntax. In either case, all threads in a block must
execute the same barriers in the same order.

## Tiled matrix multiplication

Here is a complete example for square tiles and dimensions divisible by
`tile-size`. There is one thread for every output cell.

```text
A : global.buffer[rows * inner, f32]
B : global.buffer[inner * cols, f32]
C : global.buffer[rows * cols, f32]

config = {
  grid  = (cols / tile-size, rows / tile-size, 1)
  block = (tile-size, tile-size, 1)
}

row(thread) = (block thread).index.y * tile-size + thread.thread-index.y
col(thread) = (block thread).index.x * tile-size + thread.thread-index.x

row-major(width, row, col) = row * width + col

a-index(thread, q) =
  row-major(
    inner,
    row(thread),
    q * tile-size + thread.thread-index.x)

b-index(thread, q) =
  row-major(
    cols,
    q * tile-size + thread.thread-index.y,
    col(thread))

c-index(thread) =
  row-major(cols, row(thread), col(thread))
```

The complete kernel is a function whose proof parameters depend on its buffer
and thread parameters:

```text
tile-layout(row, col) = row * tile-size + col

tile-cell(thread) =
  tile-layout(thread.thread-index.y, thread.thread-index.x)

matmul[rows, inner, cols, tile-size, config] :
  (A : global.buffer[rows * inner, f32])
  -> (B : global.buffer[inner * cols, f32])
  -> (C : global.buffer[rows * cols, f32])
  -> (thread : thread[config])
  -> (tile-a :
        shared.buffer[block thread, tile-size * tile-size, f32])
  -> (tile-b :
        shared.buffer[block thread, tile-size * tile-size, f32])

  -> (read-a :
        (q : ix(inner / tile-size))
        -> (|- read(thread, A, a-index(thread, q))))

  -> (read-b :
        (q : ix(inner / tile-size))
        -> (|- read(thread, B, b-index(thread, q))))

  -> (own-c : |- own(thread, C, c-index(thread)))

  -> (initial-own-a :
        |- own(thread, tile-a, tile-cell(thread)))

  -> (initial-own-b :
        |- own(thread, tile-b, tile-cell(thread)))

  => unit

matmul[rows, inner, cols, tile-size, config] =
  \A B C thread tile-a tile-b
   read-a read-b own-c initial-own-a initial-own-b =>

  barrier tile-loaded:
    publish every thread's writes to tile-a and tile-b
    grant each thread read proofs for every cell in tile-a and tile-b

  barrier tile-consumed:
    publish nothing
    grant each thread:
      (|- own(thread, tile-a, tile-cell(thread)))
      ● (|- own(thread, tile-b, tile-cell(thread)))

  sum = 0

  for each q in ix(inner / tile-size)
    maintaining invariant :
      (|- own(thread, tile-a, tile-cell(thread)))
      ● (|- own(thread, tile-b, tile-cell(thread)))
    initially initial-own-a ● initial-own-b:

    own-a ● own-b = invariant

    a-value =
      global.read(A, a-index(thread, q)) using read-a(q)

    b-value =
      global.read(B, b-index(thread, q)) using read-b(q)

    wrote-a =
      shared.write(block thread, tile-a, tile-cell(thread), a-value)
        using own-a

    wrote-b =
      shared.write(block thread, tile-b, tile-cell(thread), b-value)
        using own-b

    all-writes ● reads-a ● reads-b =
      shared.sync(block thread, tile-loaded)
        publishing wrote-a ● wrote-b

    for each k in ix(tile-size):
      index-a = tile-layout(thread.thread-index.y, k)
      index-b = tile-layout(k, thread.thread-index.x)

      defined-a = defined-from(all-writes, tile-a, index-a)
      defined-b = defined-from(all-writes, tile-b, index-b)

      a = shared.read(block thread, tile-a, index-a)
            using reads-a[index-a] ● defined-a

      b = shared.read(block thread, tile-b, index-b)
            using reads-b[index-b] ● defined-b

      sum = sum + a * b

    next-invariant = shared.sync(block thread, tile-consumed)
    continue with next-invariant

  global.write(C, c-index(thread), sum) using own-c

gpu.launch(
  config,
  A ● B ● C,
  matmul[rows, inner, cols, tile-size, config])
```

`rows`, `inner`, `cols`, `tile-size`, and `config` are metavariables in square
brackets; they are not runtime arguments. `A`, `tile-a`, and the other buffers
are value arguments, like an argument `x : u64`. Later proof arguments refer
to those exact buffer values. The block is obtained from `block thread`, so it
does not need another runtime binder. Launch supplies `A`, `B`, and `C`, creates
`tile-a` and `tile-b` once per block, and constructs the proof arguments after
checking the requirements across all threads. The accumulator `sum` remains
an ordinary value, separate from the loop's proof invariant. Here, `using`
passes a proof and threads it forward when the primitive preserves it.

The first barrier proves that reads are initialized and changes the block from
exclusive writing to shared reading. The second waits for all reads, removes
every read permission, and restores exclusive ownership. Consequently, a
shared cell is never written while another thread can read or write it.
