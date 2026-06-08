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
