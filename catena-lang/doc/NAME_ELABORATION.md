# Name Symbol Elaboration

For each arrow `f : A -> B` in a theory like `program`, the elaborator adds a
new arrow:

```text
name.f : 1 -> (A -> B)
```

More precisely, `name.f` is a value-level name for the function pointer denoted
by `f`.

Important: this currently produces `->` function pointers, not `=>` closures.
If a later phase needs a closure, it must use `lift`.

For example, if the input contains:

```text
bool.not : bool -> bool
bool.and : {bool bool} -> bool
```

then elaboration adds arrows morally equivalent to:

```text
name.bool.not : 1 -> (bool -> bool)
name.bool.and : 1 -> ({bool bool} -> bool)
```

The raw Hexpr written into the theory is more verbose than this. It explicitly:

- copies any explicit source metavariables of `f`
- packs multi-wire domains/codomains into `*`
- applies the `->` type constructor at the end

# Why Sources Matter

`name.f` is not always nullary.

If an arrow already has explicit source variables in its type map, those
variables remain as inputs to `name.f`. Intuitively, `name.f` names a family of
function pointers indexed by those source variables.

So the general shape is better thought of as:

```text
f      : X -> Y
name.f : M -> (X -> Y)
```

where `M` is whatever explicit source context was already present in the raw
type map.

# Packing Objects

The elaborator needs a single object on each side of `->`, but Catena type maps
can denote zero, one, or many wires. So it packs them as follows:

- `0` wires becomes `1`
- `1` wire stays as-is
- `2` wires becomes `*`
- `n > 2` wires becomes a left-associated product built from `*`

So a binary function like:

```text
bool.and : {bool bool} -> bool
```

is named as a single function pointer from the packed product object
`{bool bool}` to `bool`.

# Which Theories Are Elaborated

The top-level elaborator skips theories whose syntax category is `nat`:

```rust
.filter(|(_, theory)| theory.syntax_category.as_str() != "nat")
```

This means syntax theories themselves are not given `name.*` arrows. The pass is
intended for user/program theories written *in* that syntax.

# Current Scope

At the moment, elaboration does not yet:

- lower closures
- rewrite DSL surface constructs beyond extension folding
- introduce explicit closure environments

So the current role of elaboration is narrow but important:

- normalize extensions
- synthesize first-class names for arrows as function pointers

Later passes build on that, especially:

- checking
- `forget_closures`
- codegen

