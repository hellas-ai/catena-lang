# Buffers, Indexes, and Finite Sets

This document concerns the design of *buffers*, *indices*, and *finite spaces* in catena.
In short:

- Buffers are memory regions with dependent-typed size
- Indices represent *positions* in a buffer
- Finite spaces are the sets representing the *sizes* of buffers and indices

A quick example: we may represent a 2D matrix `x` of `f32` elements with dimensions `(a, b)` as

    x : Ix (a, b) => f32

Now suppose this is backed by a buffer; the closure must internally represent the indexing into this buffer explicitly.
For example, let's say our buffer `m` is a single element `c`, and `x` is the "broadcast" version.
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
- `ref n t`: same as `buf`, but merely a read-only reference; can be copied freely but not written.
- `Ix s`: an index (value) in some finite set s

There is a minor hiccup with this design: *entrypoints*.
Consider the "ref.head" example below

# Detail

Let's look at each in more detail.

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

The following examples demonstrate how we can write various 

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
