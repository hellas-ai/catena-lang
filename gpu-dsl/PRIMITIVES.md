# GPU kernel primitives

This is a small, language-independent interface for global-memory GPU kernels. The notation follows Catena: `●` is a product, `->` is a function, `=>` is a closure, `|-` is for proofs, and `val(t)` is a runtime value.

Shared memory, barriers, and tiling are deliberately left out.

## Supporting types

Here, I use shapes to represent index spaces. We might want them or not. Here, they are just usuful to help understand what values are intended to be.

Finite logical spaces and their indices are:

```text
space ::= d1(n) | d2(rows, cols) | d3(depth, rows, cols)

ix(d1(n))                 = ix n
ix(d2(rows, cols))        = ix rows ● ix cols
ix(d3(depth, rows, cols)) = ix depth ● ix rows ● ix cols
```

An `ix n` is an integer in `[0, n)`, so indexing is safe by construction.
`vec n t` is a vector containing exactly `n` values of type `t`.

Launch geometry is CUDA/HIP-shaped:

```text
dim3 := { x : u64, y : u64, z : u64 }

launch-config := {
  grid  : dim3,
  block : dim3
}

coord(d : dim3) := {
  x : ix d.x,
  y : ix d.y,
  z : ix d.z
}

thread(config : launch-config) := {
  block-index  : coord(config.grid),
  thread-index : coord(config.block)
}
```

A layout maps logical indices to cells of a flat buffer:

```text
layout(s : space, n : u64) := {
  offset : val(ix s) -> val(ix n)
}
```

## Schedule

A schedule assigns every logical output to one physical thread:

```text
schedule(config : launch-config, s : space) := {
  owner : val(ix s) -> val(thread config)
}
```

Since `owner` is a total function, every output has exactly one owner. A thread may own zero, one, or several outputs.

The schedule is only a mapping; it does not run the kernel. `gpu.launch` uses it to collect, for every physical thread, the ordered vector of indices owned by that thread.

For example, given a positive `block-size`, a linear schedule uses this
one-dimensional configuration:

```text
grid.x  = max(1, ceil-div(size, block-size))
block.x = block-size
grid.y = grid.z = block.y = block.z = 1
```

It assigns logical index `i` to:

```text
owner(i) = thread {
  block-index  = (i / block-size, 0, 0),
  thread-index = (i % block-size, 0, 0)
}
```

With this schedule, an active thread owns one logical index. Threads introduced by rounding the grid size up own no indices and receive an empty vector.

## Kernel

```text
kernel(config : launch-config, s : space, t : type) :=
  forall count.
  val(thread config) ● val(vec count (ix s))
  => val(vec count t)
```

A kernel is a closure from the current physical thread and all logical outputs owned by that thread to one value for each output. A thread may receive an empty vector (i.e. inactive threads). The kernel may capture read-only input buffers.

For example, global input reads use the ordinary Catena buffer operation:

```text
ref.get[n, t] : ref n t ● val(ix n) -> val(t)
```

There is no output-store primitive in a kernel body. The launch primitive owns that operation.

## Launch

```text
gpu.launch[s, n, t] :
  (config : launch-config) ●
  buf n t ●
  layout(s, n) ●
  schedule(config, s) ●
  kernel(config, s, t)
  -> buf n t
```

`gpu.launch` collects the logical indices assigned to each physical thread and runs the kernel once per thread. It stores each returned value at the layout offset of the corresponding logical index. The updated owned output buffer is returned. Execution order between different threads is unspecified.

Here, we specify the layout for the output buffer. Alternatively, we can follow the approach we used in previous implementations where `launch` takes a linear buffer and we build a helper function that compute the layout before calling `launch`.

## Defining `materialize`

`materialize` evaluates an indexed producer and stores its values in a new
linear buffer:

```text
materialize[n, t] :
  (n : u64) ● (val(ix n) => val(t))
  -> buf n t
```

It uses ordinary buffer allocation and vector mapping:

```text
buf.alloc[n, t] : (n : u64) -> buf n t

vec.map[n, a, b] :
  (val(a) => val(b)) ● val(vec n a)
  -> val(vec n b)
```

We can define `materialize` entirely in terms of these operations:

```text
def materialize[n, t](
  n        : u64,
  producer : val(ix n) => val(t)
) -> buf n t =
  let config   = linear-config(n)
  let output   = buf.alloc[n, t](n)
  let layout   = linear-layout(n)
  let schedule = linear-schedule(config, n)

  let kernel = \_thread indices ->
    vec.map(producer, indices)

  gpu.launch(config, output, layout, schedule, kernel)
```

Thus `materialize` needs no additional GPU primitive beyond `gpu.launch`;
allocation and `vec.map` come from the surrounding language.
