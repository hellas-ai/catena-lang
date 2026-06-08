# Catena DSL

The catena language is split into two parts:

- DSL (`catena-dsl`): a high-level language built on "dataflow with higher-order linear functions"
- IR (`catena`): a low-level IR built around interleaving of dataflow and controlflow graphs

(NOTE: "linear functions" means the PLT sense - uncopyable - not like "affine")

Both are intended to be usable as "surface" languages, but the DSL is intended
as a more "algebraic" specification of programs permitting easier optimization,
primarily *fusion*.

Note that while both use a representation of programs as *cospans of hypergraphs*
(open hypergraphs), there are key theoretical differences, which we outline next.

Importantly, note that in the short term we will include direct CUDA codegen from catena-DSL *in addition* to the IR
because two developers are working concurrently.
So our job is to implement the full pipeline `catena-dsl -> CUDA` without going via IR for now.

# High-level overview of DSL and IR

The Catena DSL models *dataflow with higher-order linear functions*.
This means we have:

- Programs are morphisms in a symmetric monoidal category with *product* as tensor
- There are function types:
    - `A ~> B` standing for *functions* (NOT closures)
    - `A => B` are *closures* (can have partially-applied functions with captured vars)

In contrast, the IR models *interleaved data and control-flow*
This means programs are either

- A morphism in the *product* SMC (dataflow)
- A morphism in the *coproduct* SMC (control flow)

where the product distributes over the coproduct.
Essentially, the IR models a *rig category*.

This document focuses only the DSL.

# Catena DSL Base

Within "Catena DSL" there are to be several passes (e.g., function inlining,
type erasure, and so on.)
For concreteness, we first characterise the *end result* of these passes: the
"base" of the tower of theories.

To quickly summarise, in the `base` theory:

- All closures have been replaced with *functions*
- Closure eliminators like `reduce : (A => B) -> B` are replaced by explicit-context versions
    `reduce-fn : X * (X * A -> B) -> B` where `X` is the explicit context

See `dsl/base/*.hex` for the stdlib of the "base" theory.
Notice it does not include the `cmc.hex` theory.
