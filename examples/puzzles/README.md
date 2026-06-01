# Metacat GPU puzzles

GPU puzzles in Metacat to test the language/compiler expressivity.

Inspirations

- https://puzzles.modular.com
- https://github.com/srush/gpu-puzzles
- https://github.com/gpu-mode/Triton-Puzzles

### What we can improve

- generated code has a lot of `auto output_summed = output`. The reason is that the hypergraph representation is basically SSA, that is, each assignment creates a new variable. In CUDA code we should identify wires that represent the same variable and remove those expressions.

* Should we allocate global memory in launcher? Now it looks quite unsafe since we compute the size in the launcher, but we don't allocate the memory.

### Broadcast proof

`user.f32.broadcast-add-singleton-matrix-inputs` proves memory safety with `examples/puzzles/broadcast.proof.hex`.

The output view is `ij = (i, j)` for an output shaped
`[grid_rows * tile_rows, grid_cols * tile_cols]`. Proving `ij` safe for the
output gives `i < grid_rows * tile_rows` and `j < grid_cols * tile_cols`.

The singleton input views are explicit reshapes:

- `reshape(j, shape.row-mul(grid_cols, tile_cols)) = (0, j)`
- `reshape(i, shape.col-mul(grid_rows, tile_rows)) = (i, 0)`

Therefore the row-vector access is safe because `0 < 1` and
`j < grid_cols * tile_cols`; the column-vector access is safe because
`i < grid_rows * tile_rows` and `0 < 1`.
