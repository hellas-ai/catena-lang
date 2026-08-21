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
global buffers in those inputs that each thread may read or write. This
declaration denotes a global access plan:

```text
global-plan = access {
  read  A in read-a(thread)
  read  B in read-b(thread)
  write C in write-c(thread)
}

mode ::= none | read | write

global-plan(thread, buffer, index) : mode
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

Race freedom is a proposition about the complete plan:

```text
race-free(plan) =
  for every distinct threads t and u,
  for every buffer and index,

    plan(t, buffer, index) = write
    -> plan(u, buffer, index) = none
```

Thus a cell written by one thread cannot be read or written by another thread.
Reads may overlap, and accesses to different buffers or different offsets do
not conflict. Regions can therefore be as precise as one cell per thread; they
do not need to cover a whole buffer. The checked access declaration produces a
certificate `|- race-free(plan)`.

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

Together, the plan certificate and the operation types prove global race
freedom: launch grants only permissions described by a race-free plan, and the
kernel cannot perform a global access without the corresponding permission.

## Kernels and launch

A kernel is a dependent function. Resource arguments are ordinary values, and
proof arguments may refer to the exact resources introduced earlier in the
function type. For example:

```text
kernel-body[config, n, t, index] :
  (inputs : inputs)
  -> (thread : thread[config])
  -> (tile : shared.buffer[block thread, n, t])
  -> (permission :
        |- own(write-phase, thread, tile, index(thread)))
  => unit
```

The `tile` binder behaves like any other value binder. Its type tells launch
to create one buffer per block. The later `permission` binder refers to that
particular buffer value.

The proof parameters form the kernel's access contract. Launch checks that
the requirements for all threads are race-free, supplies global buffers from
the kernel inputs, creates block-local shared buffers, constructs the permitted
proofs, and applies the function once per physical thread. Launch also creates
the kernel's expected barrier plan. The plan is linear: the kernel must consume
it completely and return the empty plan.

```text
kernel-function[access-plan][expected-plan : trace] :
  inputs ● permissions(access-plan) ● expected-plan -> result ● []

gpu.launch[access-plan][expected-plan : trace] :
  config
  ● inputs
  ● (|- race-free(access-plan))
  ● (inputs ● permissions(access-plan) ● expected-plan -> result ● [])
  -> launch
```

For each thread, launch derives `permissions(access-plan)` from the actual
input buffers and applies the kernel function to them. It also supplies every
thread in a block with a distinct linear barrier plan having the same
`expected-plan` type. A barrier plan cannot be copied, discarded, or
fabricated, so returning `[]` proves that every thread executed the expected
barriers in order.

## Shared memory

A shared buffer belongs to one block. Launch creates a separate instance for
every block:

```text
shared.buffer[block, n, t]

phase ::= write-phase | read-phase
```

Shared operations receive the block as a runtime argument. The dependent
buffer type ensures that the buffer belongs to that same block. Phases are
syntax constructors, not values or metavariables. The type checker tracks the
current phase: ordinary shared operations preserve it, while `shared.sync`
changes it according to the barrier declaration.

The kernel function expresses initial ownership with dependent proof
parameters. For example:

```text
kernel-body[config, n, t, index] :
  (thread : thread[config])
  -> (tile : shared.buffer[block thread, n, t])
  -> (initial-own :
        |- own(write-phase, thread, tile, index(thread)))
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
  phase
  ● (block : block[config])
  ● (buffer : shared.buffer[block, n, t])
  ● (index : ix n) ● value
  ● (|- own(phase, thread, buffer, index))
  -> (|- wrote(phase, thread, buffer, index))
```

The write consumes ownership and produces evidence that this thread wrote the
cell. No other thread can read or write it while the exclusive permission is
held.

### Shared reads

A shared read needs both permission to read the cell and evidence that it was
initialized:

```text
shared.read[config, n, t] :
  phase
  ● (block : block[config])
  ● (buffer : shared.buffer[block, n, t])
  ● (index : ix n)
  ● (|- read(phase, thread, buffer, index))
  ● (|- defined(buffer, index))
  -> value
     ● (|- read(phase, thread, buffer, index))
```

The cell's read permission is reusable during the current read phase.
`defined` is a pure fact derived from writes published at a barrier.

## Block synchronization

There is one block synchronization primitive:

```text
shared.sync[config, rest : trace] :
  phase
  ● (block : block[config])
  ● (barrier :: rest)
  ● barrier[phase -> next-phase]
  ● contribution(phase)
  ● grants
  -> published-facts
     ● grants(next-phase)
     ● rest
```

It has four effects:

1. Every thread in `block` waits at the same barrier.
2. The barrier publishes its fixed contribution from every thread.
3. It changes the current phase and returns `grants` indexed by the barrier's
   declared next-phase constructor.
4. It removes its barrier from the front of the remaining barrier plan.

Synchronization also consumes the entire old shared-permission context. A
thread cannot retain a proof merely by omitting it from the call. After the
barrier, only the returned grants are available. The phase argument of a shared
operation must equal the ambient current phase; it cannot be freely selected
by the caller. Global-memory permissions are unaffected.

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
tile-loaded contribution(phase, thread) =
  (|- wrote(phase, thread, tile-a, cell-a(thread)))
  ● (|- wrote(phase, thread, tile-b, cell-b(thread)))
```

After every thread arrives, each thread receives the corresponding facts for
all threads, together with read permissions for both tiles. From those
published writes it can prove that every planned tile cell is defined.

A later tile-consumed barrier need not publish a fact. Its synchronization
ensures that all threads have finished the read phase; changing the ambient
phase invalidates all read permissions, and the barrier grants ownership tied
to the next phase. All threads in a block must execute the same barriers in
the same order.

## Loops

A loop may carry both an invariant and a variant. The invariant has the same
type before and after every iteration. The variant describes a type that
decreases monotonically according to one iteration's exact transition:

```text
for each i in count
  maintaining invariant :
    invariant
  initially initial-invariant

  decreasing variant [rest] :
    next(rest) => rest
  initially iterate(count, next, final):

  body
```

The body is checked for an arbitrary `rest`, and the generic loop has the
following rule:

```text
body :
  invariant ● next(rest)
  -> invariant ● rest

for[count](body) :
  initial-invariant ● iterate(count, next, final)
  -> invariant ● final
```

This is a type-level counterpart of the loop variant used by verification
languages. It decreases in an order where `rest < next(rest)`. The declared
transition is exact, not merely the monotonicity inequality, so the loop can
compute its result type. The order and transition are generic; they do not
refer to GPU execution or barriers.

## Barrier traces

A barrier trace records synchronization structure in the kernel type:

```text
trace ::= []
        | barrier :: trace
```

Kernel invocation receives its expected trace as a remaining-work plan.
Ordinary operations preserve it. A call to `shared.sync` transforms
`barrier :: rest` into `rest`. The kernel must return `[]`.

Barrier plans instantiate the generic variant mechanism. The structured type
`iterate(count, barriers)` describes `count` repetitions of a barrier sequence;
its final empty tail is implicit. For tiled matrix multiplication the plan is:

```text
iterate(count, tile-loaded :: tile-consumed)
```

The plan decreases under the suffix order. The generic loop checks its body
once for an arbitrary `rest`: it must transform
`tile-loaded :: tile-consumed :: rest` into `rest`. It can therefore consume
the structured `iterate` plan without expanding the trace. Both branches of a
conditional must return the same remaining plan. Missing, reordered, extra, or
conditionally executed barriers therefore prevent the kernel from returning
`[]`.

An implementation may check this by threading an internal linear value:

```text
gpu.state[block, phase, trace]
```

Shared reads and writes preserve this value, while `shared.sync` changes its
phase and removes the expected barrier. The generic loop checks the declared
variant and eliminates one `iterate` layer per logical iteration. The value
cannot be copied, discarded, or fabricated, and may be erased after type
checking. It is implementation machinery rather than a runtime value.

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

The exact global access plan is:

```text
matmul-access = access [thread] {
  read A at a-index(thread, q)
    for q in ix(inner / tile-size)

  read B at b-index(thread, q)
    for q in ix(inner / tile-size)

  write C at c-index(thread)
}

c-index-injective :
  |- c-index(t) = c-index(u) -> t = u

matmul-global-safe :
  |- race-free(matmul-access)

proof:
  A and B are read-only
  C is distinct from the read-only buffer resources
  c-index-injective gives at most one writer for each C cell
```

`c-index-injective` follows from the launch geometry: distinct launched
threads have distinct `(row, col)` coordinates, and the row-major layout is
injective within the matrix bounds. The proof is about the actual buffer
resources, not merely their source-level names. If inputs may alias, launch
must also establish that `C` does not alias `A` or `B`; otherwise it rejects the
plan.

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
        |- own(write-phase, thread, tile-a, tile-cell(thread)))

  -> (initial-own-b :
        |- own(write-phase, thread, tile-b, tile-cell(thread)))

  -> (barrier-plan :
        iterate(
          inner / tile-size,
          tile-loaded :: tile-consumed))

  => unit ● []

matmul[rows, inner, cols, tile-size, config] =
  \A B C thread tile-a tile-b
   read-a read-b own-c initial-own-a initial-own-b barrier-plan =>

  barrier tile-loaded : write-phase -> read-phase:
    publish every thread's writes to tile-a and tile-b
    grant each thread read proofs for every cell
      in tile-a and tile-b

  barrier tile-consumed : read-phase -> write-phase:
    publish nothing
    grant each thread:
      (|- own(write-phase, thread, tile-a, tile-cell(thread)))
      ● (|- own(write-phase, thread, tile-b, tile-cell(thread)))

  sum = 0

  final-own-a ● final-own-b ● empty-plan =
    for each q in ix(inner / tile-size)
    maintaining invariant :
      (|- own(write-phase, thread, tile-a, tile-cell(thread)))
      ● (|- own(write-phase, thread, tile-b, tile-cell(thread)))
    initially initial-own-a ● initial-own-b

    decreasing variant [rest] :
      tile-loaded :: tile-consumed :: rest => rest
    initially barrier-plan:

    own-a ● own-b = invariant
    current-plan = variant

    a-value =
      global.read(A, a-index(thread, q)) using read-a(q)

    b-value =
      global.read(B, b-index(thread, q)) using read-b(q)

    wrote-a =
      shared.write(
        write-phase, block thread, tile-a, tile-cell(thread), a-value)
      using own-a

    wrote-b =
      shared.write(
        write-phase, block thread, tile-b, tile-cell(thread), b-value)
      using own-b

    all-writes ● reads-a ● reads-b ● after-loaded =
      shared.sync(write-phase, block thread, current-plan, tile-loaded)
        publishing wrote-a ● wrote-b

    for each k in ix(tile-size):
      index-a = tile-layout(thread.thread-index.y, k)
      index-b = tile-layout(k, thread.thread-index.x)

      defined-a = defined-from(all-writes, tile-a, index-a)
      defined-b = defined-from(all-writes, tile-b, index-b)

      a =
        shared.read(read-phase, block thread, tile-a, index-a)
          using reads-a[index-a] ● defined-a

      b =
        shared.read(read-phase, block thread, tile-b, index-b)
          using reads-b[index-b] ● defined-b

      sum = sum + a * b

    next-own-a ● next-own-b ● after-consumed =
      shared.sync(
        read-phase, block thread, after-loaded, tile-consumed)

    continue with next-own-a ● next-own-b ● after-consumed

  global.write(C, c-index(thread), sum) using own-c

  return unit ● empty-plan

gpu.launch(
  config,
  A ● B ● C,
  access matmul-access using matmul-global-safe,
  matmul[rows, inner, cols, tile-size, config])
```

`rows`, `inner`, `cols`, `tile-size`, and `config` are metavariables in square
brackets; they are not runtime arguments. `A`, `tile-a`, and the other buffers
are value arguments, like an argument `x : u64`. Later proof arguments refer
to those exact buffer values. The block is obtained from `block thread`, so it
does not need another runtime binder. Launch supplies `A`, `B`, and `C`, creates
`tile-a` and `tile-b` once per block, and constructs the proof arguments after
checking the requirements across all threads. The initial ownership proofs are
in the `write-phase` constructor; no phase value or phase metavariable is
passed. The accumulator `sum` remains an ordinary value, separate from the
loop's proof invariant. Here, `using` passes a proof and threads it forward
when the primitive preserves it. Launch supplies `barrier-plan`; the kernel
cannot construct or discard it. In the loop body, `rest` is an arbitrary
remaining plan. The iteration starts with
`tile-loaded :: tile-consumed :: rest`, the first synchronization produces
`tile-consumed :: rest`, and the second produces `rest`. The body continues
with that `rest` together with the ownership invariant. Therefore the generic
loop consumes:

```text
iterate(
  inner / tile-size,
  tile-loaded :: tile-consumed)
```

and returns `[]`, as required by the kernel result type. Kernel completion
reclaims the final permissions associated with its global and shared-memory
access contract.

For global memory, `matmul-global-safe` proves that the permissions constructed
by launch are compatible across all threads. The dependent `read-a`, `read-b`,
and `own-c` parameters then restrict every global operation in the kernel to
those checked permissions.

The first barrier proves that reads are initialized and changes the block from
exclusive writing to shared reading. The second waits for all reads, removes
every read permission, and restores exclusive ownership. Consequently, a
shared cell is never written while another thread can read or write it.
