# Buffers, Indexes, and Finite Sets

This document concerns the design of *buffers*, *indices*, and *index spaces* in catena.
In short:

- Buffers are memory regions with dependent-typed size
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
    x = \(i, j) -> b[i*a + b]

Here, the *code* x translates logical 2d coordinates into 1d buffer indices.

Thus:

- Buffers can only be indexed with completely flat (1D) indices
- Closures are used to represent different multidimensional logical indexing schemes
- Finite spaces represent the "shapes" of different schemes

**The main memory types in catena** are therefore:

- `buf n t`: an owned buffer of `n` elements of type `t`. `n` must be a type-level u64 here.
- `ref n t`: same as `buf`, but merely a read-only reference; can be copied freely but not written to.
    - Used primarily for passing in model weights
- `Ix s`: an index: a value of a finite set s

This is mathematically straightforward, but an implementation detail rears its
ugly head:

> How do we get dependently-typed buffers into the program?

Consider for example the `ref.head` program, which returns the first element of
an array reference, or zero for an empty reference.

    ref.head-or-zero : val(ref n u32) -> val(u32)

This seems fine, but when we try to *lower* `ref.head`, we see a problem:

1. `ref n u32` is not monomorphic: `n` is a free type variable
2. If `ref n u32` lowers to a pointer, its size can never be measured (pointers don't have associated lengths)

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

- Add an opaque `mem` type, representing a sized handle to some *owned* memory
    - Also a `mem` version
- This is isomorphic to `void* × size_t` (and in fact lowers to it)
    - See [this tweet](https://x.com/ZPostFacto/status/2061537537932636194)
- There is no runtime "fat pair", it actually ends up as two arguments
- To recover a dependently typed buf, we have...
    - `mem -> buf n t ● (n : u64)`
- Then we can test `n` explicitly, e.g.
    - `nz : (n : u64) -> |- n > 0`

NOTE: for this to work, we need "type ascription"; i.e., named variables at the
type level!
This allows us to name runtime values at type level!

**NOTE: BELOW HERE IS WIP NOTES**

# Detail

Let's look at each part of the design in detail (WIP)

## Buffers

Catena DSL has the following buffer types:

    buf n t         # an *owned* buffer
    ref n t         # a read-only reference to a buffer

each represents a buffer of n elements of type t.
The GPU backend treats both as *device* buffers.
currently there is no host/device distinction at the language level.

Some design notes:

- `buf` cannot be arbitrarily discard, it must be explicitly deallocated. There must be no ops that consume a buf without doing this.
- `ref` can be freely copied and discarded, but one cannot create a ref (yet)

## Indices

## Finite Spaces

These are the least well defined. However, 


# Examples

The following are examples of various programs involving buffers.

## buf.head

One primitive for indices is the `index.zero` function, which takes a proof of
non-zero size, and gives the zero index.

    index.zero : (|- size(s) > 0) -> val(ix s)

We also need to be able to look up elements in a buf or ref at their index:

    ref.ix : val(Ix n) ● val(ref n t) -> val(t)

Using this, we can write the `ref.head` function, returning the first element of a non-zero buffer.

    ref.head : (|- size(n) > 0) ● val(ref n t) -> val(t)
    ref.head = ({index.zero id} ref.ix)

But where should we get the proof `|- size(n) > 0` from?


## Multiplying by identity matrix

(TODO: this example is complicated and incomplete!)

Suppose we define the f32 identity matrix for any square 2D finite space as follows

    id : Fin (n, n) => f32
    id (i, j) = f32.from-bool (ix.eq i j)

This compares i and j, and returns a f32 1 if they are equal.
Now suppose we want to multiply `id` by an input buffer.

There's a few pieces missing:

- The boundaries of the program can only accept a `buf n t` for free `n` with no 

    # multiply a n×n-sized buffer interpreted in row-major order by the identity matrix
    mul-id : buf (n*n) f32 -> (Ix (n, n) => f32)
    mul-id b = ???

Conceptually, this is precisely what we want!
However, something is missing:

The type `buf (n * n) f32` is not monomorphisable


    # need a primitive
    index.zero : (|- n > 0) -> val(t)

    # need
    assert.positive : val(n) -> (|- n > 0)

    array.head-or-zero : buf n t -> t
    array.head-or-zero = if n > 0 {
        // we have (|- n > 0) here
    } else {
        // we have (|- ¬ n > 0) here
    }

    // contract here is that buf has size n
    void array_head_or_zero(*t buf, size_t n, *t result) {
        if n == 0 {
            *result = 0
        } else {
            *result = buf[0]
        }
    }

    // first problem: buf n t is not monomorphizable because n is free

    foo : buf (n + m) t


- mem (internally is just a pair of a pointer and a len)



    # now we transform n → (a + b)
    u32.from-mem : mem -> buf n u32


    # this is similar to reshape(?)
    buf.substitute : val(buf x t) ● (|- x = y) -> val(buf y t)

    suppose have p : |- a + b = n
    buf.substitute : 
