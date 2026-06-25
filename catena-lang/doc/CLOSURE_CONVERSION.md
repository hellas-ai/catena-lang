# Closure Conversion

If a catena program with closures is written using only the primitives `compose`,
`tensor`, `defer`, `run` and `lift`, then it can be fully "inlined" to remove
all closure types -- provided its boundary has no closure-typed objects.

Consider the following program:

![](closure-conversion-run-not.svg)

Applying the "forget closures" pass leads to this simplified code:

![](closure-conversion-run-not-forgotten.svg)

However, closures cannot always be fully removed this way. For example, when:

1. A closure type appears on a program boundary
2. A closure type is *eliminated* by an operation or definition (like `if` or `reduce`)

This document provides a solution, outlined below.

# Closure Conversion

- A program must be *closure-converted* if it has a `=>` type on:
    - A global boundary
    - A non-CMC operation
- Any such node must be closure converted (others are left unchanged)
- Closure conversion happens as follows
    - Any such `=>` type marked as 'for conversion'
    - Traverse leftwards in the hypergraph until one meets `defer` and `lift` ops which delimit the scope
    - Cut that region from the hypergraph
    - Place it in a new definition (e.g. anonymous.closure.i)
    - Replace it in the graph

The *type* of the replaced closure is figured as follows.

- Choose a new unique name for a definition: `closure.i`
    - (assume `closure` prefix is reserved for now)
- On the left hand side of the morphism, we have a number of maps in one of two forms:
    - `defer : A -> (I => A)`
    - `lift : (A -> B) -> (A => B)`
- Then...
- Order these (somewhat arbitrarily) by edge ID, combined sources define interface of `closure.i`

# Conversion for *definitions* and *primitives on closures*.

We propose the following additional primitives involving closures:

    reduce (zero : A) (add : A * A => A) (Ix n => A) (n : u64) -> A

    materialize : (Ix n => A) (n : u64) -> buf cap.own A

    if (A => B) ● (A => B) ● Bool ● A -> B

In addition, we want *user code* to be able to use closures
(this is important for lowering catgrad programs, since we use `Ix n => Dtype`
as the lowered type of a tensor with type `(Shape, Dtype)`.)

How should closure conversion work for these? We'll start with the 3 primitives above,
and consider user definitions later.

## Primitives `reduce`, `materialize` and `if`

Each of the primitives goes to its "closure converted" variant,
currently with the same name + `c` suffix.
So, ...


    # if
    if (A => B) ● (A => B) ● Bool ● A -> B

    # ... lowers to 'ifc'
    ifc (X, X ● A => B) ● (Y, Y ● A => B) ● Bool ● A -> B

    materialize : (Ix n => A) (n : u64) -> buf cap.own A
    materializec : (X, X ● Ix n -> A) (n : u64) -> buf cap.own A

    reduce (zero : A) (add : A ● A => A) (Ix n => A) (n : u64) -> A
    reducec (zero : A) (X, add : X ● A ● A -> A) (Y, Y ● Ix n => A) (n : u64) -> A

## User definitions

# A note on conditionals

Normally, when `(A => B)` lowers to `(X, X ● A -> B)`, we lower `if` to

    ifc (X, X ● A => B) ● (Y, Y ● A => B) ● Bool ● A -> B

However, if we take instead `select` on closures

    closure.select Bool ● (A => B) ● (A => B) -> (A => B)

And then `(A => B)` lowers to closures `(X, X ● A -> B)`, we get

    if Bool ● (X, X ● A => B) ● (Y, Y ● A => B) -> (???, ??? ● A -> B)

The problem above is that we need a dependent type:

    if (b : Bool) ● (X, X ● A => B) ● (Y, Y ● A => B) -> (b ? X : Y, (b ? X : Y) ● A -> B)

Meaning we need a *type level dependent sum* - the returned closure env still
depends on b.

# Implementation

This section describes the *implementation* of closure conversion.
Closure conversion is implemented *for each definition*.
So for the purposes of this section, fix a particular definition `d`.

- For each operation `x` in `d`
    - For each *source node* `w` of `x`
        - If the type of `w` declared by `x` is not a closure, continue. Otherwise...
        - Assume type of the closure is `A => B`
        - Cut the "closure region" `c` (see below) ending at `w` from the graph
        - Add a new definition `closure.d.x_id.w_id` whose body is `(c × id) ; ev`
        - ... and whose type is `X ● A -> B`
        - Run name elaboration on closures
        - ... and then replace the "cut" closure with `id_X ● name.closure.d.x_id.w_id`

The "closure region" is:

- Starting at a node `w` labeled `A => B`,
- "flood fill" left until meeting a `defer` or `name` operation
- The region includes `defer` and `name`
