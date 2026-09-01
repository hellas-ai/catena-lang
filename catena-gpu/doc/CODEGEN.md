# Code generation

Catena GPU keeps the Hex language small and typed, then lowers it to simple CUDA or HIP C++.

## Specialization

Hex definitions may be generic, but generated CUDA/HIP functions are always concrete. Code generation starts from concrete program entrypoints and processes a small specialization queue. A direct call or `name.*` function value adds the required concrete instance; an existing instance with the same runtime input and output types is reused.

Generic definitions that are never used are not emitted. The generated source therefore uses ordinary concrete function signatures and does not rely on C++ templates.

## Logical dimensions and backend representation

Hex distinguishes one-, two-, and three-dimensional shapes:

```text
shape.1d(width)
shape.2d(width, height)
shape.3d(width, height, depth)
```

An `ix(shape)` therefore carries a logical rank and bounds in its type. This prevents, for example, using a 2D index where a bounded buffer-cell index is required.

The rank-specific forms also keep Hex code simple. A 1D program supplies only `x`, and a 2D program supplies only `x` and `y`; neither has to manually add unused coordinates or launch dimensions. Code generation performs the padding required by the three-dimensional CUDA/HIP representation.

CUDA and HIP expose three-dimensional launch objects, so code generation uses one uniform representation internally:

```cpp
typedef struct {
    uint64_t first;
    uint64_t second;
    uint64_t third;
} catena_ix_t;
```

Unused coordinates are zero. A Hex 2D index `(x, y)` becomes `{x, y, 0}`. This representation does not imply a memory layout: row-major flattening remains an explicit Hex computation.

Grid and block dimensions are also emitted as backend 3D objects. Missing dimensions are padded with one because they are extents rather than coordinates:

```text
Hex grid.1d(gx, bx) -> dim3(gx, 1, 1), dim3(bx, 1, 1)
Hex grid.2d(...)    -> dim3(gx, gy, 1), dim3(bx, by, 1)
```

Every supplied launch extent has type `positive-u32`. Hex obtains it from a `u32` value and a proof that the value is positive. This rules out zero dimensions and matches the backend representation without a narrowing conversion.

A 1D grid also produces its total global extent as `u64`, after widening both factors. The 1D ownership policy needs that value to prove that every buffer cell has a corresponding thread. A 2D or 3D grid does not flatten its extents into one total: its global shape already retains the separate axis bounds.

## Common shape and index transformations

| Hex value | Generated representation |
| --- | --- |
| `ix(shape.1d(n))` with coordinate `x` | `{x, 0, 0}` |
| `ix(shape.2d(nx, ny))` with coordinates `(x, y)` | `{x, y, 0}` |
| `ix(shape.3d(nx, ny, nz))` with coordinates `(x, y, z)` | `{x, y, z}` |
| 1D grid or block extent `n` | `{n, 1, 1}` |
| 2D grid or block extent `(nx, ny)` | `{nx, ny, 1}` |
| 3D grid or block extent `(nx, ny, nz)` | `{nx, ny, nz}` |

The corresponding operations lower as follows:

- `shape.1d.size(shape)` exposes the static bound to Hex and produces no runtime code.
- `positive-u32.intro(value, proof)` keeps `value` as `uint32_t`; the positivity proof is erased.
- `ix.2d.from-components(x, y)` copies the coordinates into `{x, y, 0}`.
- `ix.2d.split(index)` projects `first` and `second`, producing two bounded 1D indices.
- `ix.1d.value(index)` projects `first` as a `u64`.
- `u64.to-ix(value, proof)` produces `{value, 0, 0}`. The proof establishes the bound in Hex and is erased.
- `gpu.thread.in-grid.index` selects the thread's global `(x, y, z)` coordinates.
- `gpu.thread.in-block.index` selects `threadIdx.(x, y, z)`.
- `gpu.block.index` selects `blockIdx.(x, y, z)`.

Shapes and bounds exist only in Hex types; they do not add fields to `catena_ix_t`. A generic index is never implicitly flattened. For example, row-major matrix layout is an explicit computation:

```text
linear = row * columns + column
```

This keeps logical coordinates independent from the memory layout selected by the program.

## Global reads and writes

Global buffers are always linear allocations. Their cells are addressed by a bounded 1D index, regardless of whether the program interprets the contents as a vector, matrix, or higher-dimensional object:

```text
buffer : gpu.global(capability, length, element-type)
cell   : ix(shape.1d(length))
```

Consequently, the primitive operations are linear:

```text
gpu.global.read(buffer, cell)
gpu.global.write(thread, buffer, cell, value, ownership-proof)
```

Reads use `cap.ref`, so they need no scheduling proof. Writes use `cap.own` and require evidence that the current thread owns the selected cell.

Code generation uses `cell.first` as the physical offset:

```cpp
value = buffer[cell.first];
buffer[cell.first] = value;
```

A multidimensional index cannot be passed directly to these operations. A storage adapter first chooses a layout and produces a linear offset. For a 2D row-major matrix:

```text
matrix-index : ix(shape.2d(rows, columns))
offset       = row * columns + column
cell         : ix(shape.1d(buffer-length))
```

The adapter checks that the offset fits the allocation before constructing `cell`. `gpu.global.matrix.2d.row-major.read` and `gpu.global.matrix.2d.row-major.write` are such adapters; they do not introduce a different kind of buffer.

This separation keeps global memory simple and makes layout a visible program choice. The same linear buffer can be interpreted through a different layout adapter without changing its runtime representation.

## Kernel launch

`gpu.launch` is generic over the kernel arguments. A particular kernel definition decides which buffers, lengths, schedules, and scalar values it receives. The kernel itself is passed as a direct `name.*` function value.

Code generation creates a GPU wrapper for each launch. In simplified form:

```cpp
__global__ void launch_wrapper(kernel_arguments...) {
    global_index = {
        blockIdx.x * blockDim.x + threadIdx.x,
        blockIdx.y * blockDim.y + threadIdx.y,
        blockIdx.z * blockDim.z + threadIdx.z
    };

    in_block_index = {threadIdx.x, threadIdx.y, threadIdx.z};
    block_index = {blockIdx.x, blockIdx.y, blockIdx.z};
    thread = {global_index, in_block_index, block_index};

    hex_kernel(kernel_arguments..., thread);
}
```

The host function launches this wrapper using the grid and block dimensions produced by `gpu.grid.*.intro`:

```cpp
launch_wrapper<<<grid_dim, block_dim>>>(kernel_arguments...);
catena_gpu_synchronize();
```

Each explicit dimension is already a positive `uint32_t`. For a 1D grid, the required global extent is calculated separately with 64-bit multiplication:

```text
global-size = u64(grid.x) * u64(block.x)
```

For 2D and 3D grids, each global axis remains separate:

```text
global-x = grid.x * block.x
global-y = grid.y * block.y
global-z = grid.z * block.z
```

Each per-axis product fits in `u64` because it multiplies two `u32` values. Codegen does not multiply the axes together, avoiding overflow in an unnecessary flattened thread count.

Logical 1D and 2D launch dimensions have already been padded for the backend: unused extents are one. Kernel code therefore receives a uniform 3D thread object while its Hex type still exposes the original logical shape.

Kernel arguments are forwarded as ordinary runtime parameters. Type-only information, static names, and proofs are erased. After synchronization, `gpu.launch` returns the runtime arguments unchanged; mutations to owned global buffers are visible through those same handles.

A launched kernel must be a direct function definition and must not reach another `gpu.launch`, including through helper functions or folds. Code generation checks this restriction before rendering the module.

## Scheduling and ownership

Scheduling is checked by Hex types but remains a small runtime value because the kernel must query the policy selected for each owned buffer:

```cpp
typedef struct {
    catena_scheduling_kind_t kind;
    uint64_t matrix_columns;
} catena_scheduling_t;
```

The current schedule constructors lower as follows:

- `gpu.scheduling.own-each` stores the 1D `OWN_EACH` policy.
- `gpu.scheduling.2d.row-major.own-each(columns)` stores the 2D row-major policy and its column count.
- Static buffer, grid, and schedule names are erased. Their role is to prevent mixing unrelated values during type checking.

`gpu.scheduling.can-own(schedule, thread, cell)` becomes a runtime policy query. For the current policies, the generated checks are equivalent to:

```text
own-each:
    thread.x == cell.first

2d row-major own-each:
    thread.x == cell.first % columns
    and thread.y == cell.first / columns
```

The Boolean decision remains at runtime. The positive ownership proof exists only in Hex and is erased. An `assert` on a false decision becomes a GPU trap; a predicated kernel can instead handle the false branch and skip the write.

After Hex has established ownership, `gpu.global.write(thread, buffer, cell, value, proof)` becomes the direct memory operation:

```cpp
buffer[cell.first] = value;
```

The thread and proof are not needed by that final instruction: they were needed to make the write type-check. A `cap.ref` buffer is read-only, so `gpu.global.read` needs no schedule or ownership query and becomes a direct indexed load.

Code generation does not prove race freedom again. It relies on schedules being constructible only through policies that are race-free by construction, while retaining the selected policy at runtime so the kernel cannot query a different resolver accidentally.
