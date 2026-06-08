# Catena: Language Overview

There are two main levels:

- Catena DSL (with closures)
- Catena Base (Closures converted)

This document enumerates the core features of the language.

## Ascriptions and Values

- `val(t)` represents a runtime value of type t
- `x : t` is a runtime value of type `t`, with a type-level name `x`.

## Testing and Comparison

For use with `assert`, tests and comparisons also return *witnesses*.
For example:

    u64.gt : (x : u64) ● (y : u64) -> (b : bool) ● (|- b = true => x > y) ● (|- b = false => ¬ (x > y))

Comparing two `u64` values returns:

- A runtime boolean value, named `b`
- A proof that if `b` is true, then `x > y`
- A proof that if `b` is false, then `¬ (x > y)`

For example, one can then pass the boolean and first proof to `assert` to obtain a proof `|- x > y`

## Buffers

- `cap.own` is the capability for owned memory
- `cap.ref` is the capability for borrowed/read-only memory
- `mem c` is an untyped memory block with capability `c`; equivalent to `(void*, size_t)` in C
- `buf c n t` is the type of buffers with capability `c`, `n` elements, and element type `t`

Common aliases:

- `mem` means `mem cap.own`
- `mem.ref` means `mem cap.ref`
- `buf n t` means `buf cap.own n t`
- `ref n t` means `buf cap.ref n t`

TODO: `n` should probably be enforced to be something that lowers explicitly to `size_t`.
For example, an `extent` type ~= `u64`, with saturating operations.

## Borrowing

TODO: ownership, fractional borrowing, etc.
Owned memory/buffers cannot be discarded; `cap.ref` values are borrowed views.

## Partial Function semantics

Catena programs are *partial* functions.
There is one way to cause a crash: `assert`

    assert : (b : bool) -> (|- b = true)

    # NOTE: earlier version had this:
    # assert : (b : bool) ● (|- b = true => p) -> |- p

`assert` takes a runtime boolean value `b`,
and allows us to conclude that `|- b = true`.

Notice this is of course not true in general:

- When `b` is false, the program will crash.
- When `b` is true, the user is able to use the proof `|- b = true`

`assert` is intended to be the *only* partial operation.

## Branches

TODO

# Functions

There are *two* types of functions:

- `A => B` is a closure. It may have an implicit captured environment
- `A -> B` is a *function pointer*. It has no captured variables.

*Closures* are automatically lowered to *Converted Closures*:

- `A => B` with implicit environment `X` becomes
- `X ● (X * A -> B)` - a function pointer plus a stored environment value `X`

## Definitions and Names

A *definition* is the (conservative) extension of the core language with a new
symbol plus a rewrite mapping that symbol to/from an arrow in the core theory.

For each *definition* `foo`, catena's elaborator adds `name.foo`: its fully-curried variant.
