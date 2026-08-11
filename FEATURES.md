# Catena performance features

## `materializec.into`

```text
materializec.into[X, C, N, T] :
  Bufᵒʷⁿ(C, T)
  ⊗ ⟦C : U64⟧
  ⊗ Val(U64)
  ⊗ ⟦N : U64⟧
  ⊗ X
  ⊗ (X ⊗ Ix(N) ⊸ Val(T))
  ⊸ Bufᵒʷⁿ(C, T)
```

The unindexed `Val(U64)` is the write offset. The operation requires
`offset + N ≤ C`, consumes the unique buffer owner, writes
`producer(x, i)` to `buffer[offset + i]` for every `i : Ix(N)`, and returns
ownership of the same capacity-`C` buffer. Elements outside that range are
unchanged.

`⊗` is the monoidal product, `⊸` is a linear map, and `⟦C : U64⟧` denotes a
runtime value witnessing the type-level natural `C`.

## `materializec.borrow`

```text
materializec.borrow[X, C, N, A, B] :
  Bufᵒʷⁿ(C, A)
  ⊗ ⟦C : U64⟧
  ⊗ ⟦N : U64⟧
  ⊗ X
  ⊗ (Bufʳᵉᶠ(C, A) ⊗ X ⊗ Ix(N) ⊸ Val(B))
  ⊸ Bufᵒʷⁿ(C, A) ⊗ Bufᵒʷⁿ(N, B)
```

The operation temporarily derives a read-only reference from the unique source
owner. The producer uses that reference to construct `N` output elements. The
reference cannot escape the materialization; after its GPU work is enqueued,
the operation returns both the unchanged source owner and the new owned output
buffer.

## `materializec.reduce-f32`

```text
materializec.reduce-f32[X, N, K] :
  ⟦N : U64⟧
  ⊗ ⟦K : U64⟧
  ⊗ X
  ⊗ (X ⊗ Ix(N) ⊗ Ix(K) ⊸ Val(F32))
  ⊗ (X ⊗ Ix(N) ⊗ Val(F32) ⊸ Val(F32))
  ⊸ Bufᵒʷⁿ(N, F32)
```

For each `i : Ix(N)`, the first map produces `K` reduction terms. They are
summed by the exact logical tree `(0 + 1), (2 + 3), ...`; an unmatched final
term is carried forward unchanged, and the rule repeats until one value
remains. The empty reduction is `+0.0`. The second map applies an epilogue to
the resulting sum and produces output `i`.

This logical tree is independent of GPU thread count and scheduling, making
the result bit-deterministic despite floating-point non-associativity. The
current GPU implementation assigns one block per output and supports
`K ≤ 4096`.

## `materializec.borrow-reduce-f32`

```text
materializec.borrow-reduce-f32[X, C, N, K, A] :
  Bufᵒʷⁿ(C, A)
  ⊗ ⟦C : U64⟧
  ⊗ ⟦N : U64⟧
  ⊗ ⟦K : U64⟧
  ⊗ X
  ⊗ (Bufʳᵉᶠ(C, A) ⊗ X ⊗ Ix(N) ⊗ Ix(K) ⊸ Val(F32))
  ⊗ (Bufʳᵉᶠ(C, A) ⊗ X ⊗ Ix(N) ⊗ Val(F32) ⊸ Val(F32))
  ⊸ Bufᵒʷⁿ(C, A) ⊗ Bufᵒʷⁿ(N, F32)
```

This is `materializec.reduce-f32` with a scoped read-only borrow of an owned
source buffer available to both the term map and epilogue. The borrow cannot
escape. The operation returns the unchanged source owner together with the
new owned output buffer.

For each output, reduction uses the same exact adjacent-pair tree as
`materializec.reduce-f32`, including carrying unmatched terms unchanged and
using `+0.0` for an empty domain. GPU scheduling therefore cannot alter the
resulting bits. The current implementation supports `K ≤ 8192`, selecting
64, 128, 256, or 512 physical threads without changing that logical tree.

## `materializec.softmax-f32`

```text
materializec.softmax-f32[N] :
  Bufᵒʷⁿ(N, F32)
  ⊗ ⟦N : U64⟧
  ⊗ Val(U64)
  ⊗ (Val(F32) ⊸ Val(F32))
  ⊸ Bufᵒʷⁿ(N, F32)
```

The unindexed `Val(U64)` is the number of columns; it must be non-zero and
divide `N` exactly. The final map supplies the exponential operation. Each
contiguous row is normalized in place, consuming and returning ownership of
the same buffer.

Both row reductions use the exact adjacent-pair tree defined by
`materializec.reduce-f32`. Maximum selection retains the left operand unless
the right is strictly greater. An input with bits `0xFF800000` (negative
infinity) produces a `+0.0` numerator without invoking the exponential; every
other numerator is `exp(value - maximum)`. A zero denominator produces an
all-`+0.0` row. The current GPU implementation supports at most 8192 columns.

## `materializec.borrow-argmax-f32`

```text
materializec.borrow-argmax-f32[N, One] :
  Bufᵒʷⁿ(N, F32)
  ⊗ ⟦N : U64⟧
  ⊗ ⟦One : U64⟧
  ⊸ Bufᵒʷⁿ(N, F32) ⊗ Bufᵒʷⁿ(One, U64)
```

The input must be non-empty and `One = 1`. The operation temporarily borrows
the input, returns its unchanged owner, and emits the selected index in a
one-element owned buffer.

Selection uses an IEEE-bit total order spanning negative NaNs through positive
NaNs. Equal bit patterns select the greatest index. These rules completely
specify the result independently of GPU scheduling.

# Qwen

The following features were added while implementing and optimizing
Qwen3-30B-A3B-Base. They live in the exploratory compiler fork and are grouped
separately from the features that were already present for SmolLM2.

## `materializec.reduce-f32-pair`

```text
materializec.reduce-f32-pair[X, N, K] :
  ⟦N : U64⟧
  ⊗ ⟦K : U64⟧
  ⊗ X
  ⊗ (X ⊗ Ix(N) ⊗ Ix(K) ⊸ Val(F32) ⊗ Val(F32))
  ⊗ (X ⊗ Ix(N) ⊗ (Val(F32) ⊗ Val(F32))
       ⊸ Val(F32) ⊗ Val(F32))
  ⊸ Bufᵒʷⁿ(N, F32) ⊗ Bufᵒʷⁿ(N, F32)
```

The term map produces two terms for each `(output, reduction)` index. Their
left and right components are reduced independently using the exact tree from
`materializec.reduce-f32`; an empty domain supplies `(+0.0, +0.0)`. The
epilogue consumes both sums and produces the corresponding pair of outputs.

One kernel shares term setup and synchronization barriers between the two
trees without changing either tree's arithmetic. The current implementation
supports `K ≤ 4096`.

## `materializec.borrow-routed-bf16-gemv-pair`

```text
materializec.borrow-routed-bf16-gemv-pair[I, G, U, S, N, K] :
  Bufᵒʷⁿ(I, F32)
  ⊗ ⟦I : U64⟧
  ⊗ Bufʳᵉᶠ(G, BF16)
  ⊗ ⟦G : U64⟧
  ⊗ Bufʳᵉᶠ(U, BF16)
  ⊗ ⟦U : U64⟧
  ⊗ Bufᵒʷⁿ(S, U64)
  ⊗ ⟦S : U64⟧
  ⊗ ⟦N : U64⟧
  ⊗ ⟦K : U64⟧
  ⊗ Val(U64)
  ⊗ Val(U64)
  ⊸ Bufᵒʷⁿ(I, F32)
    ⊗ Bufᵒʷⁿ(S, U64)
    ⊗ Bufᵒʷⁿ(N, F32)
    ⊗ Bufᵒʷⁿ(N, F32)
```

The two unindexed values are `slots` and `output_features`. Let
`rows = I / K`. The operation requires `0 < K ≤ 4096`, `slots > 0`,
`output_features > 0`, `I = rows × K`, `S = rows × slots`, and
`N = S × output_features`. Gate and up capacities must be equal and divisible
by `output_features × K`; the quotient is the expert count.

Weights are BF16 tensors laid out as
`[expert, output-feature, reduction-feature]`. Selected expert IDs are U64
values laid out as `[row, slot]`, and each ID must be smaller than the expert
count. Both F32 outputs use `[row, slot, output-feature]` layout.

The operation temporarily borrows the F32 input and selected-expert buffers,
then returns both owners with gate and up results. Expert lookup, row/channel
division, and weight-base calculation occur once per output block. BF16
conversion and F32 multiplication remain separate for gate and up, and both
outputs use the exact adjacent-pair tree from `materializec.reduce-f32-pair`.

## `materializec.borrow-topk-f32`

```text
materializec.borrow-topk-f32[N, R, O] :
  Bufᵒʷⁿ(N, F32)
  ⊗ ⟦N : U64⟧
  ⊗ ⟦R : U64⟧
  ⊗ ⟦O : U64⟧
  ⊗ Val(U64)
  ⊗ Val(U64)
  ⊸ Bufᵒʷⁿ(N, F32) ⊗ Bufᵒʷⁿ(O, U64)
```

The two unindexed values are `columns` and `k`. The operation requires
`columns > 0`, `N = R × columns`, `O = R × k`, and `1 ≤ k ≤ 8`.
It temporarily borrows the source, returns its unchanged owner, and emits
row-major top-`k` column indices. Callers requiring `k` valid indices per row
must additionally ensure `k ≤ columns`.

Values are ranked in descending IEEE-bit total order after canonicalizing both
signed zeros to `+0.0`; equal ordering keys retain the lower column index.
Selection does not alter any source bits.

For rows through 128 columns, the GPU lowering uses one 128-thread block and a
synchronized bitonic sort. Wider rows use a serial stable-insertion fallback.
Both paths use the same ordering key and tie rule; neither path uses atomics or
scheduling-dependent writes.

## Deterministic reduction scheduling refinements

Qwen's 6144-term MoE down projection motivated a 512-thread launch for
`materializec.borrow-reduce-f32` when `K > 4096`. Thread count changes only
which lane materializes a term and performs an independent indexed addition;
the logical adjacent-pair tree is unchanged.

The Qwen work also replaces late whole-block barriers with explicit warp/wave
barriers in `materializec.reduce-f32`, `materializec.reduce-f32-pair`,
`materializec.borrow-reduce-f32`, and the routed BF16 pair. Once
`stride ≥ 256`, the maximum supported reduction length of 8192 leaves at most
16 active lanes, all within the first 32-lane CUDA warp or 64-lane AMD wave.
Every addition retains the same lane, logical index, left operand, right
operand, and parenthesization.

GPU tests compare every output bit with the canonical CPU tree at reduction
lengths 257, 2048, and 6144, including cancellation-sensitive values and
signed zero. The maximum-active-lane argument is part of the feature's
determinism requirement; increasing the supported reduction length requires
re-establishing it.

## GPU module save/load hook

`CATENA_GPU_MODULE_SAVE=<path>` copies a newly compiled GPU module to `path`.
`CATENA_GPU_MODULE_LOAD=<path>` loads that module instead of invoking the GPU
compiler. This opt-in runtime hook supports profiling tools that cannot safely
nest a `hipcc` invocation; normal runtime construction does not use it, and
the caller is responsible for cache validity.
