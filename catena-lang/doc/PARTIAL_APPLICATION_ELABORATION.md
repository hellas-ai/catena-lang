# Partial-Application Elaboration

An operation named `partial.f.N`, where `N` is decimal, denotes `f` with its
first `N` input wires captured. The elaborator generates this arrow only when
the operation occurs in a definition body.

For an arrow morally shaped as:

```text
f : A1 * (A2 * ... * Ak) -> B
```

the generated arrow is morally:

```text
partial.f.N :
  A1, ..., AN
    -> (A(N+1) * (... * Ak) => B)
```

As with `name.f`, any explicit source context needed by the original type maps
is retained as an input to the generated operation.

The generated definition uses the CMC structure rather than adding a new
runtime primitive. Starting from the packed remaining suffix, it prepends the
captured inputs from right to left. Each step `defer`s one captured input,
tensors it with an identity closure for the current suffix, accounts for the
left unit, and produces the next right-associated suffix. It then `compose`s
the completed domain with `name.f lift`. The boundary cases simplify naturally:

- `partial.f.0` is `name.f lift`.
- `partial.f.k` captures the full domain and produces a `1 => B` closure.

Malformed names, missing target arrows, and values of `N` larger than the input
arity are elaboration errors.
