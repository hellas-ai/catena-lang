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

This document provides a solution, by:


