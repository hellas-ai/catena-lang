# Catena: Language Overview

There are two main levels:

- Catena DSL (with closures)
- Catena Base (Closures converted)

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
