# Metacat GPU puzzles

GPU puzzles in Metacat to test the language/compiler expressivity.

Inspirations

- https://puzzles.modular.com
- https://github.com/srush/gpu-puzzles
- https://github.com/gpu-mode/Triton-Puzzles

### What we can improve

- generated code has a lot of `auto output_summed = output`. The reason is that the hypergraph representation is basically SSA, that is, each assignment creates a new variable. In CUDA code we should identify wires that represent the same variable and remove those expressions.

* Should we allocate global memory in launcher? Now it looks quite unsafe since we compute the size in the launcher, but we don't allocate the memory.
