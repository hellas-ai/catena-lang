# GPU kernel primitives

This note describes a small, language-independent interface for GPU kernels.
It focuses on the ingredients needed to make kernels total and deterministic
by construction: bounded indexing and iteration, initialized reads, exclusive
writes, race-free launches, and uniform block synchronization. It does not
specify execution details or Lean encodings.

The examples use Catena-like notation:

```text
●     product of values or resources
->    function pointer
=>    closure, and the body separator in a closure literal
==>   logical implication
|- p  proof of proposition p
```

Square brackets contain type parameters and indices. Parentheses introduce
named value or proof binders and are used for value-level application. A
callable takes one product of binders and has one result
arrow (`->` or `=>`); arrows are not used as informal punctuation between
arguments. Calls use parentheses for values. The `using` clause is reserved
for authorization proofs or resources that the operation returns unchanged;
resources that are consumed or transformed are ordinary positional inputs.

An indexed linear family is written `family[index] { resource }`. Selecting
`permissions[index]` borrows that entry for one operation; if the operation
preserves it, the entry is threaded back into the same family. Selection does
not construct or copy a permission.

Families are deliberately not written as ordinary functions. A type such as
`(index : ix[n]) -> permission(index)` would appear callable more than once at
the same index and would therefore appear to fabricate copies of a linear
permission. A family instead denotes permissions that already exist, with one
entry selected and threaded linearly. A future surface language could use
function-like syntax only if its type also exposed the consumption and return
of the permission collection. Another possible representation is one
region-level permission plus a pure proof that each accessed index belongs to
the region; the family notation is used here because this document models
launch as producing cell-level permissions.

The examples use effect-style result binding. Ordinary results and resources
whose types change are named on the left-hand side. A resource passed with
`using` and returned unchanged is threaded implicitly, so it need not be named
again. Unit inputs and outputs are omitted. These are surface conveniences for
the full product types shown in the primitive declarations.

Definitions use `=`. Record fields use `=` inside `{ ... }`. A spaced colon
separates a declaration from its type; a trailing colon introduces an indented
body. Hyphenated names are used for DSL identifiers; prose metavariables use
lower-case names. Static arguments may be positional, inferred, or named as in
`shared.read[phase = read-phase](...)`.

Permission proofs are linear resources: they cannot be copied or silently
retained after an operation consumes them. Other proofs, such as bounds or
initialization facts, may be reusable.

## Indices and buffers

An `ix[n]` is an integer in `[0, n)`. Buffer operations use bounded indices, so
out-of-bounds accesses cannot be expressed.

```text
global.buffer[n, t]
```

A global buffer identifies a global-memory resource. A cell is identified by
its buffer and offset.

A layout maps logical indices to physical offsets:

```text
layout[logical, n] : {
  offset    : logical -> ix[n]
  injective : |- offset(i) = offset(j) ==> i = j
}
```

## Threads

A launch configuration contains a grid size and a block size. A thread carries
both its block coordinate and its local coordinate:

```text
block[config] : {
  index : coord[config.grid]
}

thread[config] : {
  block-index  : coord[config.grid]
  thread-index : coord[config.block]
}

block-of[config] : (thread : thread[config]) -> block[config]

threads[block] = {
  thread : thread[config] | block-of(thread) = block
}
```

## Global memory

A kernel takes arbitrary inputs. It separately declares the parts of the
global buffers in those inputs that each thread may read or write. This
declaration denotes a global access plan:

```text
global-plan = access (thread) {
  read  A in region-a(thread)
  read  B in region-b(thread)
  write C in region-c(thread)
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
index in region-a(thread) ==> (|- read(thread, A, index))
index in region-b(thread) ==> (|- read(thread, B, index))
index in region-c(thread) ==> (|- own(thread, C, index))
```

This region notation is shorthand. A kernel may state the same contract
directly with dependent proof parameters, as in the matmul example below.
For one thread, `permissions[plan, thread]` packages the resulting cell proofs
as indexed linear families.

Race freedom is a proposition about the complete plan:

```text
race-free(plan) =
  for every distinct threads t and u,
  for every buffer and index,

    plan(t, buffer, index) = write
    ==> plan(u, buffer, index) = none
```

Thus a cell written by one thread cannot be read or written by another thread.
Reads may overlap, and accesses to different buffers or different offsets do
not conflict. Regions can therefore be as precise as one cell per thread; they
do not need to cover a whole buffer. The checked access declaration produces a
certificate `|- race-free(plan)`.

Global operations use those cell permissions directly. In the following
types, the permission passed with `using` is part of the input product:

```text
global.read[config, n, t] :
  (thread : thread[config])
  ● (buffer : global.buffer[n, t]) ● (index : ix[n])
  ● (permission : |- read(thread, buffer, index))
  -> (value : t) ● (|- read(thread, buffer, index))

global.write[config, n, t] :
  (thread : thread[config])
  ● (buffer : global.buffer[n, t]) ● (index : ix[n]) ● (value : t)
  ● (permission : |- own(thread, buffer, index))
  -> (|- own(thread, buffer, index))
```

Reading preserves the cell's read permission. Writing preserves its exclusive
ownership, so the owning thread may update the cell more than once.

Together, the plan certificate and the operation types prove global race
freedom: launch grants only permissions described by a race-free plan, and the
kernel cannot perform a global access without the corresponding permission.

## Kernels and launch

A kernel body is a dependent closure. Resource arguments are ordinary values,
and proof arguments may refer to the exact resources introduced earlier in
the argument product. For example:

```text
dependent-arguments[config, n, t, index] =
  (inputs : inputs)
  ● (thread : thread[config])
  ● (tile : shared.buffer[block-of(thread), n, t])
  ● (permission :
        |- own(write-phase, thread, tile, index(thread)))
```

The `tile` binder behaves like any other value binder. Its type tells launch
to create one buffer per block. The later `permission` binder refers to that
particular buffer value.

The proof parameters form the kernel's access contract. Launch checks that
the requirements for all threads are race-free, supplies global buffers from
the kernel inputs, creates block-local shared buffers, constructs the permitted
proofs, and applies the closure once per physical thread. Launch also creates
the kernel's expected barrier trace. The trace is linear: the kernel must
consume it completely and return the empty trace.

```text
kernel-function[
  config : launch-config,
  access-plan : access,
  expected-trace : trace] =
    (inputs : inputs)
    ● (thread : thread[config])
    ● (shared : shared-arguments[thread])
    ● permissions[access-plan, thread]
    ● initial-shared-context[thread]
    ● expected-trace
    => result ● []

gpu.launch[access-plan : access, expected-trace : trace] :
  (config : launch-config)
  ● (inputs : inputs)
  ● (|- race-free(access-plan))
  ● (kernel :
       kernel-function[config, access-plan, expected-trace])
  -> launch[result]
```

For each physical thread, launch derives
`permissions[access-plan, thread]` from the actual input buffers, creates its
block's `shared-arguments[thread]`, constructs its checked
`initial-shared-context[thread]`, and applies the kernel closure. It also
supplies every thread in a block with a distinct linear trace having the same
`expected-trace` type. A trace cannot be copied, discarded, or
fabricated, so returning `[]` proves that every thread executed the expected
barriers in order.

`shared-arguments[thread]` is the product of the kernel's declared shared
buffers for `block-of(thread)`. Kernel syntax may flatten that product into
named binders such as `tile-a` and `tile-b`. It may likewise flatten `inputs`,
`permissions[access-plan, thread]`, and `initial-shared-context[thread]` into
the `A`/`B`/`C`, `read-a`/`read-b`/`own-c`, and `initial-own-a`/`initial-own-b`
binders used by the complete example.

Kernel arguments supplied by launch form a scoped resource context. On
`return`, all still-live permissions from that context are transferred back to
launch; arbitrary resources cannot be discarded this way. The empty barrier
trace remains explicit in the return value because completion must check that
the kernel consumed its whole synchronization trace.

## Shared memory

A shared buffer belongs to one block. Launch creates a separate instance for
every block:

```text
shared.buffer[block, n, t]

phase ::= write-phase | read-phase
```

Shared reads and writes receive a block rather than a thread. Their buffer is
indexed by that block, while the permission is indexed by the current thread.
The block argument has the refined type
`block[config] where block = block-of(thread)`, so it is the exact block of the
ambient kernel thread. The buffer is then indexed by that same block value.
There is no separate thread argument or implicit thread parameter on either
primitive.

Phases are static syntax constructors, not runtime values. The type checker
tracks the current phase: ordinary shared operations preserve it, while
`shared.sync` changes it according to the instruction's `next-phase` index.

The kernel closure expresses initial ownership with dependent proof
parameters. For example:

```text
initial-ownership[thread, tile, index] =
  (initial-own :
    |- own(write-phase, thread, tile, index(thread)))
```

Taken across all threads, these requirements are the ownership plan. Launch
constructs the proofs only after checking that two different threads are not
given ownership of the same cell. A kernel can use several proof parameters
when a thread initially owns several cells.

### Shared writes

Exclusive ownership permits one thread to write a cell:

```text
shared.write[config, n, t, phase] :
  (block : block[config] where block = block-of(thread))
  ● (buffer : shared.buffer[block, n, t])
  ● (index : ix[n]) ● (value : t)
  ● (permission : |- own(phase, thread, buffer, index))
  -> (|- wrote(phase, thread, buffer, index))
```

The write consumes ownership and produces evidence that this thread wrote the
cell. No other thread can read or write it while the exclusive permission is
held.

### Shared reads

A shared read needs both permission to read the cell and evidence that it was
initialized:

```text
shared.read[config, n, t, phase] :
  (block : block[config] where block = block-of(thread))
  ● (buffer : shared.buffer[block, n, t])
  ● (index : ix[n])
  ● (permission : |- read(phase, thread, buffer, index))
  ● (initialized : |- defined(buffer, index))
  -> (value : t)
     ● (|- read(phase, thread, buffer, index))
```

The cell's read permission is reusable during the current read phase.
`defined` is a pure fact derived from block-wide write facts produced at a
barrier.

`defined-from` is a proof rule, not a memory operation:

```text
defined-from :
  (block-facts : facts)
  ● (buffer : shared.buffer[block, n, t])
  ● (index : ix[n])
  ● (contains : |- contains-write(block-facts, buffer, index))
  -> (|- defined(buffer, index))
```

The final `contains` proof is pure and may be inferred. In the matmul example,
the first sync instruction's `all-writes` family and the tile-coverage proof
provide it.

## Block synchronization

There are no separate barrier definitions. Each `shared.sync` instruction
describes the facts it lifts and the grants it creates. Its label records that
instruction in the common trace.

### Lifting thread facts

The caller does not merely state the desired result family. A `lifting` clause
consumes concrete facts about the ambient thread and generalizes their type by
replacing that thread with an arbitrary block thread `u`:

```text
lifting current-facts as [u]:
  fact-schema[u]
```

The clause must establish that `current-facts` has type
`fact-schema[thread]`. Because every thread executes the same sync instruction,
synchronization receives one value of `fact-schema[u]` from every `u` in the
block. It returns exactly:

```text
lift-block[block, fact-schema] =
  family[u : threads[block]] { fact-schema[u] }

lift-block[block, unit] = unit
```

This is a collective rule, not an ordinary local function that manufactures
other threads' facts. The result family is determined by the schema in the
`lifting` clause. The schema may contain any well-formed proposition about
`u`, not only `wrote`. When the current facts and schema are both `unit`, the
clause and its result may be omitted.

### Grant constructors

The kernel has one fixed shared ownership plan, checked at launch:

```text
ownership-plan[config] : {
  cell  : type
  owner : cell -> coord[config.block]
}
```

Synchronization cannot describe an arbitrary output permission set. Its
`granting` clause must construct one of two grant forms:

```text
barrier-grants[plan, thread] ::=
    reads[read-grants[thread]]
  | owns[own-grants[plan, thread]]
```

A read grant names a shared buffer and produces the family of cell read
permissions for that buffer in the next phase:

```text
read-grant[thread, buffer] =
  family[index : ix[buffer.length]] {
    |- read(next-phase, thread, buffer, index) }
```

Read grants may be combined with other read grants. An ownership grant is more
restricted:

```text
own-grant[plan, thread] :
  (buffer : shared.buffer[block-of(thread), n, t])
  ● (layout : plan.cell -> ix[n])
  ● (|- injective(layout))
  ● (cell : plan.cell)
  ● (|- plan.owner(cell) = thread.thread-index)
  -> own(next-phase, thread, buffer, layout(cell))
```

Ownership grants may be combined only with other ownership grants. Therefore
one sync instruction returns either read grants or ownership grants, never a
mixture. More importantly, `own` cannot be granted merely because the desired
outputs appear race-free: the caller must prove that the fixed launch plan
already designates this thread as the owner of the logical cell, and that the
layout maps logical cells injectively to physical cells. This is the same
evidence used to obtain the thread's initial ownership grant at launch, so a
sync may restore ownership only to its previously designated owner.

The `own` permission itself remains linear and may already have been consumed
by `shared.write`. What is reused at a later sync is the pure ownership
eligibility proof—`plan.owner(cell) = thread.thread-index` together with layout
injectivity—not the consumed permission token. This is the same semantic
distinction made by the Lean prototype.

### Synchronization rule

```text
shared.sync[
  name : barrier,
  config : launch-config,
  phase : phase,
  next-phase : phase,
  rest : trace,
  fact-schema : thread[config] -> facts,
  plan : ownership-plan[config],
  new-grants : barrier-grants[plan, thread]] :
  (block : block[config] where block = block-of(thread))
  ● (current-trace : name :: rest)
  ● (current-grants : grants[phase, thread])
  ● (current-facts : fact-schema[thread])
  -> lift-block[block, fact-schema]
     ● grants-result[new-grants, next-phase]
     ● (remaining-trace : rest)
```

The block argument has the same refinement used by `shared.read` and
`shared.write`. The ambient `thread` is determined by `current-grants` and
`current-facts`, and the refinement enforces `block = block-of(thread)`. The
thread is not another runtime argument.

The `current-grants` argument is that thread's complete shared-permission
context for `phase`; synchronization consumes it in full. `current-facts` are
also consumed and lifted. The checked grant constructors produce the complete
permission context for `next-phase`. `current-trace` is consumed and its head
is removed. None of these inputs uses the preserving `using` syntax. Global
memory permissions are unaffected.

All threads in a block must execute the same labeled sync instructions, with
the same fact family and grant form, in the same order. The common
trace enforces the labels and order. At a call site,
`lifting facts as [u]: schema[u]` supplies both `current-facts` and
`fact-schema`. The `granting [u]` clause must elaborate to either the `reads`
or `owns` constructor of `barrier-grants`; an `owns` clause must supply the
ownership-plan and injectivity proofs above.

### Canonical call forms

The remainder of this note uses only these surface forms:

| Primitive | Canonical invocation | Explicitly bound result |
| --- | --- | --- |
| `global.read` | `global.read(thread, buffer, index) using read-proof` | `value`; the read proof is preserved |
| `global.write` | `global.write(thread, buffer, index, value) using ownership` | none; ownership is preserved |
| `shared.write` | `shared.write[phase = p](block, buffer, index, value, ownership)` | write evidence; ownership is consumed |
| `shared.read` | `shared.read[phase = p](block, buffer, index) using read-proof ● defined-proof` | `value`; the read proof is preserved |
| `shared.sync` | `shared.sync[name = b](block, trace, grants) lifting facts as [u]: schema[u] granting [u]: reads ... | owns ... using proofs` | lifted block facts, new grants, and the remaining trace |
| `gpu.launch` | `gpu.launch[access-plan](config, inputs, kernel) using race-free-proof` | `launch[result]` |

Static parameters not shown in a call are inferred from its arguments and
expected result.

## Loops

A loop may carry both an invariant and a variant. The invariant has the same
type before and after every iteration. The variant describes a type that
decreases monotonically according to one iteration's exact transition:

```text
for each i in ix[count]
  maintaining invariant:
    invariant
  initially initial-invariant

  decreasing variant [rest]:
    from next(rest) to rest
  initially iterate[count, next, final]:

  body
```

The body is checked for an arbitrary `rest`, and the generic loop has the
following rule:

```text
body :
  invariant ● next(rest)
  => invariant ● rest

for[count, next, final] :
  ([rest] invariant ● next(rest) => invariant ● rest)
  ● initial-invariant
  ● iterate[count, next, final]
  -> invariant ● final
```

This is a type-level counterpart of the loop variant used by verification
languages. It decreases in an order where `rest < next(rest)`. The declared
transition is exact, not merely the monotonicity inequality, so the loop can
compute its result type. The order and transition are generic; they do not
refer to GPU execution or barriers.

When an `ix[n]` loop preserves the entire resource context, the finite index
domain supplies the ordinary termination measure and the annotations may be
omitted:

```text
for each i in ix[n]:
  body
```

The inner accumulation loop in the matmul example uses this shorthand. The
annotated outer loop is required because its barrier-trace type changes.

## Barrier traces

A barrier trace records synchronization structure in the kernel type:

```text
trace ::= []
        | barrier :: trace
```

Kernel invocation receives its expected trace as a linear remaining trace.
Ordinary operations preserve it. A call to `shared.sync` transforms
`barrier :: rest` into `rest`. The kernel must return `[]`.

Barrier plans instantiate the generic variant mechanism without overloading
`iterate`. Prefixing a fixed barrier sequence is an ordinary transition:

```text
prefix[barriers](rest) = barriers ++ rest

iterate[
  count,
  prefix[tile-loaded :: tile-consumed],
  []]
```

This describes `count` repetitions of the two-barrier sequence ending in the
empty trace.

The trace decreases under the suffix order. The generic loop checks its body
once for an arbitrary `rest`: it must transform
`tile-loaded :: tile-consumed :: rest` into `rest`. It can therefore consume
the structured `iterate` trace without expanding it. Both branches of a
conditional must return the same remaining trace. Missing, reordered, extra, or
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

## Guarantee boundary

For a well-typed kernel, totality means that every launched thread finishes
under fair execution. All memory indices are bounded, shared reads are proved
initialized, global buffers contain values, and every loop ranges over a finite
`ix[n]` domain with an exact decreasing variant. Every barrier is one that all
threads in its block must reach. The core language must likewise expose only
total scalar primitives and must reject uncontrolled recursion or any other
source of divergence.

Determinism means that fixed inputs and a fixed launch configuration determine
the same observable buffer contents, independently of thread scheduling. Cell
permissions exclude conflicting global and shared accesses, barriers derive
only fixed block facts from fixed per-thread grants, and each thread
evaluates its scalar expressions in a fixed order. Nondeterministic primitives
such as unordered atomics are not part of this interface.

This guarantee is relative to the scalar semantics. In particular, identical
floating-point source expressions need not be bitwise identical across GPU
targets unless rounding, contraction, exceptional values, and other relevant
floating-point behavior are fixed by the language.

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

row(thread) = block-of(thread).index.y * tile-size + thread.thread-index.y
col(thread) = block-of(thread).index.x * tile-size + thread.thread-index.x

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

The exact global access plan follows. Regions are sets of physical indices; a
singleton region is written `{ index }`:

```text
a-region(thread) = {
  a-index(thread, q) | q in ix[inner / tile-size]
}

b-region(thread) = {
  b-index(thread, q) | q in ix[inner / tile-size]
}

matmul-access = access (thread) {
  read  A in a-region(thread)
  read  B in b-region(thread)
  write C in { c-index(thread) }
}

c-index-injective :
  |- c-index(t) = c-index(u) ==> t = u

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

The complete kernel is a closure whose proof parameters depend on its buffer
and thread parameters:

```text
tile-coordinate(thread) =
  (thread.thread-index.y, thread.thread-index.x)

tile-layout((row, col)) = row * tile-size + col

tile-cell(thread) =
  tile-layout(tile-coordinate(thread))

tile-owner((row, col)) = (col, row, 0)

tile-ownership = ownership-plan {
  cell  = ix[tile-size] × ix[tile-size]
  owner = tile-owner
}

tile-layout-injective :
  |- injective(tile-layout)

tile-cell-owned(thread) :
  |- tile-ownership.owner(tile-coordinate(thread))
     = thread.thread-index

matmul[rows, inner, cols, tile-size, config] :
  (A : global.buffer[rows * inner, f32])
  ● (B : global.buffer[inner * cols, f32])
  ● (C : global.buffer[rows * cols, f32])
  ● (thread : thread[config])
  ● (tile-a :
        shared.buffer[block-of(thread), tile-size * tile-size, f32])
  ● (tile-b :
        shared.buffer[block-of(thread), tile-size * tile-size, f32])

  ● (read-a :
        family[q : ix[inner / tile-size]] {
          |- read(thread, A, a-index(thread, q)) })

  ● (read-b :
        family[q : ix[inner / tile-size]] {
          |- read(thread, B, b-index(thread, q)) })

  ● (own-c : |- own(thread, C, c-index(thread)))

  ● (initial-own-a :
        |- own(write-phase, thread, tile-a, tile-cell(thread)))

  ● (initial-own-b :
        |- own(write-phase, thread, tile-b, tile-cell(thread)))

  ● (barrier-trace :
        iterate[
          inner / tile-size,
          prefix[tile-loaded :: tile-consumed],
          []])

  => unit ● []

matmul[rows, inner, cols, tile-size, config] =
  \A B C thread tile-a tile-b
   read-a read-b own-c initial-own-a initial-own-b barrier-trace =>

  sum = 0

  final-own-a ● final-own-b ● empty-trace =
    for each q in ix[inner / tile-size]
    maintaining invariant:
      (|- own(write-phase, thread, tile-a, tile-cell(thread)))
      ● (|- own(write-phase, thread, tile-b, tile-cell(thread)))
    initially initial-own-a ● initial-own-b

    decreasing variant [rest]:
      from tile-loaded :: tile-consumed :: rest to rest
    initially barrier-trace:

    own-a ● own-b = invariant
    current-trace = variant

    a-value =
      global.read(thread, A, a-index(thread, q)) using read-a[q]

    b-value =
      global.read(thread, B, b-index(thread, q)) using read-b[q]

    wrote-a =
      shared.write[phase = write-phase](
        block-of(thread), tile-a, tile-cell(thread), a-value, own-a)

    wrote-b =
      shared.write[phase = write-phase](
        block-of(thread), tile-b, tile-cell(thread), b-value, own-b)

    all-writes ● reads-a ● reads-b ● after-loaded-trace =
      shared.sync[
        name = tile-loaded,
        phase = write-phase,
        next-phase = read-phase](
        block-of(thread),
        current-trace,
        unit)
      lifting wrote-a ● wrote-b as [u]:
        (|- wrote(write-phase, u, tile-a, tile-cell(u)))
        ● (|- wrote(write-phase, u, tile-b, tile-cell(u)))
      granting [u] reads:
        tile-a
        ● tile-b

    for each k in ix[tile-size]:
      index-a = tile-layout((thread.thread-index.y, k))
      index-b = tile-layout((k, thread.thread-index.x))

      defined-a = defined-from(all-writes, tile-a, index-a)
      defined-b = defined-from(all-writes, tile-b, index-b)

      a =
        shared.read[phase = read-phase](
          block-of(thread), tile-a, index-a)
          using reads-a[index-a] ● defined-a

      b =
        shared.read[phase = read-phase](
          block-of(thread), tile-b, index-b)
          using reads-b[index-b] ● defined-b

      sum = sum + a * b

    next-own-a ● next-own-b ● after-consumed-trace =
      shared.sync[
        name = tile-consumed,
        phase = read-phase,
        next-phase = write-phase](
        block-of(thread),
        after-loaded-trace,
        reads-a ● reads-b)
      granting [u] owns:
        tile-a at tile-coordinate(u) via tile-layout
          using tile-layout-injective ● tile-cell-owned(u)
        ● tile-b at tile-coordinate(u) via tile-layout
          using tile-layout-injective ● tile-cell-owned(u)

    continue with next-own-a ● next-own-b ● after-consumed-trace

  global.write(thread, C, c-index(thread), sum) using own-c

  return unit ● empty-trace

gpu.launch[matmul-access](
  config,
  A ● B ● C,
  matmul[rows, inner, cols, tile-size, config])
using matmul-global-safe
```

`rows`, `inner`, `cols`, `tile-size`, and `config` are metavariables in square
brackets; they are not runtime arguments. `A`, `tile-a`, and the other buffers
are value arguments, like an argument `x : u64`. Later proof arguments refer
to those exact buffer values. The block is obtained from `block-of(thread)`, so it
does not need another runtime binder. Launch supplies `A`, `B`, and `C`, creates
`tile-a` and `tile-b` once per block, and constructs the proof arguments after
checking the requirements across all threads. The initial ownership proofs are
in the `write-phase` constructor; no phase value or phase metavariable is
passed. The accumulator `sum` remains an ordinary value, separate from the
loop's proof invariant. Here, `using` passes a proof and threads it forward
when the primitive preserves it. Launch supplies `barrier-trace`; the kernel
cannot construct or discard it. In the loop body, `rest` is an arbitrary
remaining trace. The iteration starts with
`tile-loaded :: tile-consumed :: rest`, the first synchronization produces
`tile-consumed :: rest`, and the second produces `rest`. The body continues
with that `rest` together with the ownership invariant. Therefore the generic
loop consumes:

```text
iterate[
  inner / tile-size,
  prefix[tile-loaded :: tile-consumed],
  []]
```

and returns `[]`, as required by the kernel result type. Kernel completion
reclaims the final permissions associated with its global and shared-memory
access contract.

For global memory, `matmul-global-safe` proves that the permissions constructed
by launch are compatible across all threads. The dependent `read-a`, `read-b`,
and `own-c` parameters then restrict every global operation in the kernel to
those checked permissions.

The first sync instruction proves that reads are initialized and changes the
block from exclusive writing to shared reading. The second waits for all
reads, removes every read permission, and restores exclusive ownership.
Consequently, a shared cell is never written while another thread can read or
write it.
