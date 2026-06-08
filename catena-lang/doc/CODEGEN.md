# Code Generation

Catena DSL's codegen has a few overlapping concerns:

- Representation lowering: we need C runtime representations for all metacat types
- Monomorphisation & Entrypoints
    - A function with type parameters may requiring multiple versions of the 'same' templated C function
    - How do we decide which definitions are exposed as C functions with a predictable ABI?
- Host/device distinction: some functions may be needed on host, device, or both!
- Kernels: some operations (gpu materialize) cannot be run as a normal C function, but MUST be launched as a kernel.

The following subsections describe each problem in more detail, and sketch
a solution.

## Representation Lowering

We need to represent each catena type as a C type.
For example:


    val(Ix n)   ~> u64                  # TODO: indexes over space n map to u64
    val(f32 )   ~> float
    val(bool)   ~> bool

    val(buf t)  ~> *t                   # pointer to t, representing allocated array
    a -> b      ~> ERROR                # we do *not* support runtime function values. These must be statically known.
    a => b      ~> ERROR                # closures must not survive lowering
    f32         ~> Erased               # empty (no representation)

    a * b       ~> lower(a), lower(b)   # products maximally expanded into scalars

Thus, we can map each interface `Vec<Type>` to a `Vec<LoweredType>`;
when rendering, `Erased` values will simply not appear.

## Monomorphisation and Entrypoints

Monomorphisation and entrypoints are two (very related) problems.
Consider the polymorphic identity function

    id : A -> A

What C code should this serialize to?

    void id(??? x0, ???* r0) {
        *r0 = x
    }

Clearly we cannot synthesize a C function for the general declared type of `id`.
However, many of its *specialisations* can be synthesized.
For example, if we use it with `A = val(bool)`, we'd expect

    void id__bool(bool x0, bool* r0) {
        *r0 = x
    }

Now, when generating code intended to be called externally, we also want:

- A predictable function symbol name ("id" would be perfect!)
- A *monomorphic* type (i.e., there's only one variant to call!)
- The *definition* to only call *specialised* instantiations

This third point means that every *call* to an operation must be *monomorphic*.
To sketch our solution:

- A "monomorphic" type is monomorphic for codegen if representation lowering succeeds:
    - Every retained runtime component has a concrete `CType`
    - Any remaining polymorphism is erased
    - E.g., `t ● val(buf t)` is not monomorphic, but `t ● val(buf bool)` is (if t is erased). See the table below for more examples.
- Codegen also records a list of *monomorphic entrypoints*
    - (an entrypoint is simply a definition whose type is already monomorphic)
- Within each definition body...
    - Polymorphic usages of a definition cause an error
    - Each monomorphic usage of an operation (or definition) generates a specialisation

| Type | Codegen-monomorphic? | Reason |
| --- | --- | --- |
| `val(bool)` | yes | lowers to `bool` |
| `val(buf bool)` | yes | lowers to `bool *` |
| `val(buf t)` | no | retained buffer element type is unknown |
| `t` | yes, for ABI purposes | `t` is erased |
| `t * val(bool)` | yes, for ABI purposes | `t` is erased; retained component lowers to `bool` |
| `val(t)` | no | runtime value type is unknown |
| `a -> b` | no | runtime function values are not represented |

**Solution Detail**

Entrypoints are exactly those definitions declared with a monomorphic type.
Then monomorphize any polymorphic definitions by generating the same 'template'
implementation for each set of type parameters used -- the "specialization
key".
For example, suppose we have

    (def program id : a -> a = [x])

And later we use that in another program:

    (def program bool-id : (bool val) -> (bool val) = id)

So then doing codegen for `bool-id`, we would only generate the *bool*
specialisation of `id`.

    // append a unique id to distinguish from other instantiations with
    // different types.
    void id_0(bool x0, bool* r0) { ... }

So in general, codegen does *not* synthesize a definition unless it is actually
used with a specific (monomorphic) type.
In fact, what we do is:

- Find entrypoints: `program` definitions whose declared interface is already monomorphic
- Seed a worklist with those entrypoint instances
- For each retained internal op use:
    - Primitive ops are rendered inline or via prelude support
    - `program` definition uses enqueue a specialised definition instance
    - Retained polymorphic uses are errors
    - Erased polymorphic uses are ignored

Codegen then synthesizes a C function for each pair of `(op_name,
specialization_key)`, where the specialization key records the concrete source
and target interface types used at that occurrence, plus any erased static
operands that affect generated code, such as direct function symbols.
Entrypoints keep predictable symbol names; internal specialisations use
deterministic mangled names.

For now, function symbols do not propagate through arbitrary dataflow.
`gpu.materialize` only accepts an immediate direct function symbol; if a function
symbol flows through `if`, product construction, `eval`, etc., codegen rejects it
until symbolic propagation is implemented.

## Host/Device Distinction

Now suppose we want to use `gpu.materialize` to perform a simple copy.
Thus, the kernel to launch is (more or less) the identity function.

Originally, we passed a function pointer (`->` type).
However, this has the problem that function pointers on host/device have
different address spaces.

**Solution**

Function types `->` are no longer represented at runtime.
Instead, we try to "propagate" the function symbols through the program.
The initial version of this is to simply mark the output node of any `->`
constant without further propagating.

Moreover, ordinary synthesized functions are emitted as `__host__ __device__`.
Generated `gpu.materialize` kernels are emitted as `__global__`. Entrypoint
wrapper functions are host-callable and launch kernels where needed.

## Kernel Launches

The motivating example here is `gpu.materialize`.
When concretized with a specific function symbol, the actual definition must be
a `__global__` (i.e., a kernel).
But this is not just another C program to call: it must be launched as a kernel.

**Solution**

synthesize both the global definition -- prefixed with `__global__` -- as well
as a 'wrapper function' that calls it.

# Deficiencies

Currently we make our lives easier by taking some shortcuts:

- We use the globally set hip device instead of threading it everywhere
