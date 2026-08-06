# The Closure Product Boundary Problem

## The problem

Consider a closure with a product domain:

```text
A * B => C
```

The existing forgetting rule recursively removes both the closure and the
product structure:

```text
A * B => C
    │
    ▼
[A, B, C]
```

Closure conversion, however, models a closure marker with one domain endpoint
and one codomain endpoint:

```text
[domain, codomain] ──▶ !closure ──▶ opaque closure
```

For `A * B => C`, the intended domain endpoint has type `A * B`; it is not two
independent closure-domain endpoints. Flattening it to `A, B` loses the fact
that the closure receives one product argument.

The previous implementation repaired that mismatch by reversing the boundary
of a product eliminator:

```text
A ─┐
   ├──▶ [flipped *.elim] ──▶ A*B ──▶ [body] ──▶ C
B ─┘                         │                  │
                             └──▶ !closure ◀───┘
```

This does not look correct and causes issues with closure conversion because `flipped *.elim` is interpreted as an escaped wire/edge. The correct forget should give.

```text
 A*B ──▶ [body] ──▶ C
 │                  │
 └──▶ !closure ◀───┘
```

## Attempted solution 1: preserve products on boundaries

It does not work, requires too many changes.

## Attempted solution 2: preserve closure sides only

The attempted solution gives products a contextual interpretation:

- A standalone product is forgotten into its components.
- A product which is the domain or result of a closure remains one object.
- The closure constructor itself is still erased.
- Units retain their existing zero-wire external representation.

The central example becomes:

```text
forget(A * B => C) = [A * B, C]
```

The closure marker receives this representation directly:

```text
[A * B, C]
      │
      ▼
  !closure
      │
      ▼
(A * B => C)
```

No flipped eliminator is needed at this simple boundary.

It seems to work, but again it requires some changes in some basic assumptions on the code base. Below we document those changes.

### Boundary representations

The attempted representation distinguishes three related views of an object:

1. Its source type.
2. Its ordinary forgotten boundary.
3. Its representation when it is one side of a closure.

For `T = A * B`, these views are:

```text
source type:                  A * B
ordinary forgotten boundary: [A, B]
closure-side boundary:       [A * B]
```

Its proposed object rules are:

```text
forget(A * B)      = [A, B]
forget(A * B => C) = [A * B, C]
forget(A => B * C) = [A, B * C]
```

Products are therefore preserved by context, not globally:

```text
standalone product boundary: flatten
closure-side product:        preserve
```

Units keep their existing zero-wire interpretation:

```text
forget(1)      = []
forget(1 => A) = [A]
```

A closure marker has one internal domain node and one internal codomain
node. For `1 => A`, a temporary unit node can be constructed internally even
though it is absent from the external forgotten boundary:

```text
[] ──unit.intro──▶ 1
                     ╲
                      !closure ──▶ (1 => A)
                     ╱
                    A
```

### What remains unchanged

The attempted solution removes only the artificial flipped eliminator at a
closure boundary. It does not remove normal product operations from the
pipeline. An ordinary operation still receives adapters when its source-level
ABI expects a product:

```text
f : A * B -> C

[A, B] ──*.intro──▶ A * B ──f──▶ C
```

Similarly, an ordinary product result is flattened in the normal direction:

```text
f ──▶ A * B ──*.elim──▶ [A, B]
```

These operations remain genuine construction and elimination. The contextual
rule distinguishes them from a closure-side product which is already present
as one object.

### Why the attempted solution is not a local change

The desired object rule looks local:

```text
inside a closure side: preserve A * B
everywhere else:       flatten A * B
```

The compiler pipeline, however, previously relied on one context-independent
flattening convention. Closed-monoidal operations, operation adapters,
named-call inlining, and region conversion all assumed that the same product
had the same flattened representation wherever it appeared. Making the object
representation depend on whether it is a closure side changes those
assumptions. The following sections describe the resulting consequences of the
attempted solution.

### Consequences for `forget_closures`

The object action of `forget_closures` distinguishes ordinary products from
closure sides. A closure is erased into its domain and result, with each
non-unit side packed into one logical object. Nested closures are recursively
erased inside that packed representation.

Source and target adapters consequently have different jobs:

- An ordinary product boundary is adapted between flat wires and the packed
  source-level object expected by an operation.
- A product closure side is already packed and is connected directly.
- A closure consumed by an ordinary operation is bracketed by a marker whose
  internal boundary has one domain node and one codomain node.
- Nested closure structure may require real, correctly directed structural
  adapters, but a simple product closure side requires none.

The generic rule "flatten every product after an operation" is no longer
sufficient. Target adapters also need the original type context so that they
flatten standalone products while retaining closure-side products.

### Consequences for closed-monoidal operations

The original mappings of `defer`, `run`, `compose`, `tensor`, `lift`, and
`name.*` relied on closure sides and standalone objects having the same fully
flattened boundary. Preserving closure-side products removes that equality, so
these operations expose the appropriate boundary explicitly.

#### `defer`

For a product result:

```text
defer : A * B -> (1 => A * B)
```

the mapped boundaries are:

```text
source: [A, B]
target: [A * B]
```

The product is constructed because it becomes one side of a closure:

```text
[A, B] ──*.intro──▶ [A * B]
```

#### `run`

`run` exposes a preserved closure result as an ordinary result:

```text
run : (1 => A * B) -> A * B

[A * B] ──*.elim──▶ [A, B]
```

The eliminator is normal and correctly directed.

#### `compose`

For:

```text
(A => B) * (B => C) -> (A => C)
```

the forgotten interface is conceptually:

```text
[A, B, B, C] -> [A, C]
```

Each letter denotes one closure-side object. If `B = X * Y`, the interface is:

```text
[A, X * Y, X * Y, C] -> [A, C]
```

The two packed intermediate objects meet at the cap without first being
flattened.

#### `tensor`

For:

```text
(A0 => B0) * (A1 => B1)
    ->
(A0 * A1 => B0 * B1)
```

the input and output boundaries are:

```text
[A0, B0, A1, B1]
    ->
[A0 * A1, B0 * B1]
```

The mapping reorders the inputs and genuinely constructs the two new product
sides:

```text
A0 ─┐
    ├─*.intro──▶ A0 * A1
A1 ─┘

B0 ─┐
    ├─*.intro──▶ B0 * B1
B1 ─┘
```

Unit sides are internally materialized when tensor needs them to construct a
source-level product, but they remain absent from the external forgotten
boundary.

#### `lift` and `name.*`

For a function with a product argument:

```text
f : A * B -> C
```

the forgotten closure must preserve a directed path from the packed domain to
the codomain:

```text
A * B ───────────────┐
  │                  │
  └──▶ eval(f) ──▶ C │
                     ▼
                  !closure
```

One occurrence of the packed domain is the closure delimiter. The occurrence
fed to `eval` is opened only as required by the ordinary function ABI. The
result is packed when it is a product closure result.

This ordering matters for region discovery. Packing independent flat inputs
after evaluating the function would make the packed domain and codomain
sibling outputs rather than the endpoints of a path:

```text
flat inputs ──▶ body ──▶ flat outputs
     │                         │
     ▼                         ▼
packed domain             packed result
```

That graph has no directed closure body from the packed domain to the packed
result.

### Consequences for closure-specific named-call inlining

The named-call inliner recovers the forgotten boundary around `eval`. It can no
longer open every product merely because the node has product type.

Structural adjacency determines the interpretation:

```text
product input produced by *.intro:
    open it

product output consumed by *.elim:
    open it

product with no adjacent structural adapter:
    retain it as a closure-side object
```

For an ordinary operation adapter:

```text
[A, B] ──*.intro──▶ A * B ──▶ eval
```

the inliner follows `*.intro` backward and recovers `[A, B]`.

For a preserved closure side:

```text
A * B ──▶ !closure
```

there is no structural adapter, so `A * B` remains one boundary object.

Adapter direction is significant:

- Input recovery follows a producing `*.intro`.
- Output recovery follows a consuming `*.elim`.
- Introduction and elimination are not treated as interchangeable.
- No inverse product adapter is manufactured when an adapter is absent.

When the marker contains an internal unit endpoint, named-call boundary
recovery omits it to retain the external rule `forget(1 => A) = [A]`.

### Consequences for region discovery

Region discovery includes branches which consume and discard all or part of a
closure argument. A fully discarded product is a valid example:

```text
domain ──*.elim──┬──▶ unit.elim
                 └──▶ unit.elim
```

Both outputs are discarded, so the eliminator belongs to the closure region.

A multi-output operation can instead have one discarded and one live result:

```text
value ──*.elim──┬──▶ unit.elim
                └──▶ live output ──▶ outside operation
```

Backward traversal from the sink must not classify the live sibling as part of
the discarded branch. A producer belongs to the discarded branch only when
all its outputs have already been classified as discarded:

```text
all outputs discarded -> include producer
one output still live  -> do not include producer
```

Without this condition, the live output is marked internal and region
replacement later reports it as an escaping wire.

### Consequences for region replacement

With nested closures, a closure domain can also participate in structural
plumbing belonging to an enclosing closure:

```text
outside structural operation
            │
            ▼
       closure domain ──▶ inner body
```

The domain is a cut endpoint, not always a node private to the inner region.
If an ordinary edge outside the region still uses that domain, replacement
retains it. If no outside edge uses it, it is deleted with the rest of the
region as before.

The same shared-domain predicate governs both decisions:

1. Whether a region is ready for extraction.
2. Whether the domain node is included in the deletion plan.

Using the same rule avoids two failure modes:

- A nested region is never considered ready because structural plumbing uses
  its domain.
- A region is considered ready but replacement deletes a domain node still
  referenced by the enclosing graph.

### Scope of the change

The attempted solution preserves products only at closure boundaries. That
limitation keeps the ordinary runtime interface and the later product-unpacking
pipeline unchanged:

```text
ordinary A * B:      [A, B]
closure-side A * B:  [A * B]
```

Preserving products everywhere would be a substantially larger change. In
that model:

- `forget(A * B)` would become `[A * B]` globally.
- Product introduction and elimination could no longer generally map to
  identities during closure forgetting.
- Every ordinary operation boundary would need a packed-product ABI or a
  separate lowering convention.
- Named-call inlining could not use the presence of structural adapters to
  distinguish ordinary and closure-side products.
- The later product-unpacking pass and recorded boundary sizes would need a
  new contract.
- Closure-converted primitive interfaces and code generation would need to
  agree on where product flattening finally occurs.

The closure-only rule avoids that broader pipeline redesign, but it is not a
one-function change. Its complexity is concentrated in the places where
closure boundaries are created, interpreted, inlined, discovered as regions,
and replaced.

### Invariants of the attempted solution

The attempted representation is consistent only if the pipeline follows these
boundary invariants:

```text
forget(A * B)      = [A, B]
forget(A * B => C) = [A * B, C]
forget(A => B * C) = [A, B * C]
forget(1 => A)     = [A]
```

Every closure marker has an internal boundary containing exactly one domain
node and one codomain node:

```text
[domain, codomain] ──▶ !closure ──▶ opaque closure
```

For a simple product closure side, the external product node is the marker's
product node, so no structural adapter is present. Standalone products retain
their ordinary flat boundary and use correctly directed structural adapters
only where an operation ABI requires them.
