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
