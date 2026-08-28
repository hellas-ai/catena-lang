# Code generation

Catena GPU keeps the Hex language small and typed, then lowers it to simple CUDA or HIP C++.

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
