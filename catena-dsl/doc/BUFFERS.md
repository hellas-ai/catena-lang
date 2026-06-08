# Buffers, Indexes, and Finite Sets

This document concerns the design of *buffers*, *indices*, and *index spaces* in catena.
In short:

- *Memory* is untyped, sized (in bytes) regions of memory (in C, pairs of `void*, size_t`), parameterized by an ownership capability
- Buffers are contiguous sequences of elements of some type, with dependent-typed size
- Indices represent *positions* in a buffer
- Index spaces are the sets representing the *sizes* of buffers and indices
    - they are *iterable*

A quick example: we may represent a 2D matrix `x` of `f32` elements with dimensions `(a, b)` as

    x : Ix (a, b) => f32

Now suppose this is backed by a buffer; the closure must internally represent
the indexing into this buffer explicitly.

For example, let's say our buffer `m` is a single element `c`, and `x` is the
"broadcast" of this buffer to the space `(a, b)`.
Then

    m : buf 1 f32
    m = [0]

    x : Fin (a, b) => f32
    x = \(i, j) -> b[i*a + j]

Here, the *code* x translates logical 2d coordinates into 1d buffer indices.

Thus:

- Buffers can only be indexed with completely flat (1D) indices
- Closures are used to represent different multidimensional logical indexing schemes
- Finite spaces represent the "shapes" of different schemes

**The main memory types in catena** are therefore:

- `cap.own`: the capability for owned memory
- `cap.ref`: the capability for borrowed/read-only memory
- `mem c`: an untyped memory block with capability `c`, conceptually a pair `(void*, size_t)`
- `buf c n t`: a typed buffer with capability `c`, `n` elements, and element type `t`. `n` must be a type-level u64 here.

For readability, we usually use aliases:

- `mem` means `mem cap.own`
- `mem.ref` means `mem cap.ref`
- `buf n t` means `buf cap.own n t`
- `ref n t` means `buf cap.ref n t`

Thus `ref n t` is the read-only/reference form of `buf n t`; it can be copied freely but not written to.
    - Used primarily for passing in model weights
- `Ix s`: an index: a value of a finite set s

This is mathematically straightforward, but an implementation detail rears its
ugly head:

> How do we get dependently-typed buffers into the program?

Consider for example the `head-or-zero` program, which returns the first element of
an array reference, or zero for an empty reference.

    head-or-zero : val(ref n u32) -> val(u32)

This seems fine, but when we try to *lower* `ref.head`, we see a problem:

1. `ref n u32` is not monomorphic: `n` is a free type variable
2. If `ref n u32` lowers to a raw pointer, its size can never be measured (raw pointers don't have associated lengths)

So how do we write programs that deal with buffers of "proven size"?
Some ideas:

1. Runtime representation of `ref n t` is a *pair* of ref and size (simplest?)
    - problem: we need to use templates in codegen - polymorphic over t!
2. A `ref` cannot be passed as an argument, but must be *fetched* by some API (e.g. "load" op - ref by id; ids are passed in?)
    - The API, being 'internal', provides proofs of length
3. We expose a type for the ABI like `ref.typed n t` which *is* a pair of ptr/len, but can be unwrapped to `ref n t`
    - explicit iso `ref.typed n t ↔ ref n t * u64`
4. There is a special predicate `is.length(b, n)` which can be interpreted by runtime as an input constraint to signal a 'boundary contract'
5. Hybrid of 3 and 4: expose a boundary-only ABI type which lowers inline to ptr/len arguments, and also signals the runtime/checker to enforce the length contract before refining to `ref n t`.

Actually, what we'll go with is a kind of hybrid approach:

- Add an opaque `mem c` type, representing a sized handle to some memory with capability `c`
- This is isomorphic to `void* × size_t` (and in fact lowers to it)
    - See [this tweet](https://x.com/ZPostFacto/status/2061537537932636194)
- `mem` is not encoded as a "fat pointer" (explicit pair), but actually lowers
  to two separate values
- To recover a dependently typed buf, we have...
    - `mem c -> buf c n t ● (n : u64)`
    - Notice that this explicitly links a runtime value `n : u64` with the buffer size.
- Then we can test `n` explicitly, e.g.
    - `assert-nz : (n : u64) -> |- n > 0`

In order to make this work, we're missing a couple things:

- A way to recover information about types (`assert`, branches)
- Type ascription (`n : u64`)

# Buffers, Indices, and Finite Spaces

This section covers the basic types, and their intended meanings

## Buffers

Catena DSL has the following buffer types:

    mem c           # untyped memory with capability c
    buf c n t       # buffer with capability c

with common aliases:

    mem             # mem cap.own
    mem.ref         # mem cap.ref
    buf n t         # buf cap.own n t
    ref n t         # buf cap.ref n t

`buf c n t` represents a buffer of n elements of type t.
The GPU backend treats buffers as *device* buffers.
currently there is no host/device distinction at the language level.

Some design notes:

- `buf cap.own n t` cannot be arbitrarily discarded; it must be converted back to `mem cap.own` or explicitly deallocated.
- `buf cap.ref n t` can be freely copied and discarded, but one cannot create a ref (yet)
- Forgetting element type preserves capability: `buf c n t -> mem c`

## Indices

TODO

## Finite Spaces

TODO

# Examples

The following are examples of various programs involving buffers.

## buf.head

One primitive for indices is the `index.zero` function, which takes a proof of
non-zero size, and gives the zero index.

    index.zero : (|- size(s) > 0) -> val(ix s)

(I beg the reader's forgiveness: please assume an appropriate `size` exists
here).
We also need to be able to look up elements in a buf or ref at their index:

    ix : val(ix n) ● val(ref n t) -> val(t)

Using this, we can write the `ref.head` function, returning the first element of a non-zero buffer.

    ref.head : (|- size(n) > 0) ● val(ref n t) -> val(t)
    ref.head = ({index.zero id} ref.ix)

But where should we get the proof `|- size(n) > 0` from?
Naturally, this does *not* hold in general: some `n` are zero sized!

This is where we need to either branch or `assert`:

    assert-then : (b : bool) ● (|- b = true ==> p) -> |- p

this lets us conclude `p` from a runtime value.
When `p` is false, the program stops (partial).

Combining this with a *witnessed test* lets us obtain a `|- size(n) > 0`;
for example:

    u64.nz : (x : u64) -> (b : bool) ● (|- b = true => size(x) > 0) ● (|- b = false => ¬ (size(x) > 0))

Using the 'true' branch proposition with `assert` and `b` allows us to construct `|- size(x) > 0`.

## Multiplying by identity matrix

(TODO: this example is complicated and incomplete!)
