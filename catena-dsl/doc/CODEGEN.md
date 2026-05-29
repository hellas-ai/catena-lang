# Code Generation

Catena DSL's codegen has a few overlapping concerns:

- Representation lowering: we need C runtime representations for all metacat types
- Monomorphisation: a function with type parameters may requiring multiple versions of the 'same' templated C function
- Host/device distinction: some functions may be needed on host, device, or both!
- Kernels: some operations (gpu materialize) cannot be run as a normal C function, but MUST be launched as a kernel.

The following subsections describe each problem in more detail, and sketch
a solution.

## Representation Lowering

We need to represent each catena type as a C type.
For example:


    val(Ix n)   ~> u64                  # Indexes over space n map to u64
    val(f32 )   ~> float
    val(bool)   ~> bool

    val(buf t)  ~> *t                   # pointer to t, representing allocated array
    a -> b      ~> lower(a) -> lower(b) # here -> is short for a c function pointer
    a => b      ~> ERROR                # closures must not survive lowering
    f32         ~> Erased               # empty (no representation)

    a * b       ~> lower(a), lower(b)   # products maximally expanded into scalars

Thus, we can map each interface `Vec<Type>` to a `Vec<LoweredType>`;
when rendering, `Erased` values will simply not appear.

## Monomorphisation

Consider the polymorphic identity function

    id : A -> A

What C code should this serialize to?

    void id(??? x0, ???* r0) {
        *r0 = x
    }

C++ templates aside, we clearly need to generate this code with a type.
Thus, if an annotated definition

**Solution**

Monomorphize by generating the same 'template' implementation for each set of
type parameters used -- the "specialization key".
For example, suppose we have

    (def program id : a -> a = [x])

And later we use that in another program:

    (def program bool-id : (bool val) -> (bool val) = id)

So then doing codegen for `bool-id`, we would only generate the *bool*
specialisation of `id`.

    // append a unique id to distinguish from other instantiations with
    // different types.
    void id_0(bool x0, bool* r0) { ... }

So in general, codegen does *not* synthesize a definition unless it is actually used with a specific type.
In fact, what we do is:

- For each op, definition, create a mapping `Name → List<Vec<Type>>`, where `Type` is a `Tree`
- This records all the different types the op is instantiated with
- Fill this dict by...
    - Walk all definitions
    - For each internal op (including definitions called)...
    - Record the *instantiated types* passed to each in the outer mapping

Codegen then synthesizes a C function for each pair of `(op_name, type)`, using
a unique integer (the position in the above list) to keep them distinct.

## Host/Device Distinction

Now suppose we want to use `gpu.materialize` to perform a simple copy.
Thus, the kernel to launch is (more or less) the identity function.

However, we must pass a *device* pointer; this requires a declaration like

    __device__ void id(??? x0, ???* r0) {
        *r0 = x
    }

The problem here is that catena's surface language does not distinguish between
host and device pointers, but Hip/CUDA do.

**Solution**

The runtime representation of function pointers is a *pair* of:

- Host fn pointer
- Device fn pointer

When *using* a function pointer, we pick whichever is correct based on the context we're in.

## Kernel Launches

The motivating example here is `gpu.materialize`.
Assuming we pass a genuine device pointer argument defining the 'innards' of
the kernel, how should we synthesize this?
The actual definition must be a `__global__` (i.e., a kernel).

The solution is we synthesize both the global definition -- prefixed with
`__global__` -- as well as a 'wrapper function' that calls it.

# Deficiencies

Currently we make our lives easier by taking some shortcuts:

- We use the globally set hip device instead of threading it everywhere
