# Metacat GPU puzzles

GPU puzzles in Metacat to test the language/compiler expressivity.

Inspirations

- https://puzzles.modular.com
- https://github.com/srush/gpu-puzzles
- https://github.com/gpu-mode/Triton-Puzzles

### What we can improve

- generated code has a lot of `auto output_summed = output`. The reason is that the hypergraph representation is basically SSA, that is, each assignment creates a new variable. In CUDA code we should identify wires that represent the same variable and remove those expressions.

* Should we allocate global memory in launcher? Now it looks quite unsafe since we compute the size in the launcher, but we don't allocate the memory.
* We add guards to ensure that indices are not out of bound. This could happen if launches have extra threads. In general, threads/blocks are configured at run time. I don't know if we can be smarter and detect unsafety at compile time.

- Right now the compiler is conservative and emits guards for each view. To remove redundant input guards, we’d need to represent view expressions symbolically and check implication against existing guards.
  - For example, in `user.f32.broadcast-add-matrix-inputs`, the output view is `ij = (i, j)` for an output shaped `[tile_rows, tile_cols]`. The output guard proves `i < tile_rows` and `j < tile_cols`. The input views are built as `(0, j)` for an input shaped `[1, tile_cols]` and `(i, 0)` for an input shaped `[tile_rows, 1]`. Once the output guard holds, those input accesses are already safe: `0 < 1`, `j < tile_cols`, and `i < tile_rows`. The extra input checks are therefore redundant, but the compiler currently does not prove that relationship.
