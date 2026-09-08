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
materializec.reduce-f32[X, N, K, T] :
  ⟦N : U64⟧
  ⊗ ⟦K : U64⟧
  ⊗ X
  ⊗ (X ⊗ Ix(N) ⊗ Ix(K) ⊸ Val(F32))
  ⊗ (X ⊗ Ix(N) ⊗ Val(F32) ⊸ Val(T))
  ⊸ Bufᵒʷⁿ(N, T)
```

For each `i : Ix(N)`, the first map produces `K` reduction terms. They are
summed by the exact logical tree `(0 + 1), (2 + 3), ...`; an unmatched final
term is carried forward unchanged, and the rule repeats until one value
remains. The empty reduction is `+0.0`. The second map applies an epilogue to
the resulting sum and produces output `i`.

The GPU backend accepts `T = F32` or `T = BF16`. Terms and the complete
reduction tree remain F32; only the epilogue result and output storage vary.

This logical tree is independent of GPU thread count and scheduling, making
the result bit-deterministic despite floating-point non-associativity. The
current GPU implementation assigns one block per output and supports
`K ≤ 16384`.

## `materializec.borrow-reduce-f32`

```text
materializec.borrow-reduce-f32[X, C, N, K, A, T] :
  Bufᵒʷⁿ(C, A)
  ⊗ ⟦C : U64⟧
  ⊗ ⟦N : U64⟧
  ⊗ ⟦K : U64⟧
  ⊗ X
  ⊗ (Bufʳᵉᶠ(C, A) ⊗ X ⊗ Ix(N) ⊗ Ix(K) ⊸ Val(F32))
  ⊗ (Bufʳᵉᶠ(C, A) ⊗ X ⊗ Ix(N) ⊗ Val(F32) ⊸ Val(T))
  ⊸ Bufᵒʷⁿ(C, A) ⊗ Bufᵒʷⁿ(N, T)
```

This is `materializec.reduce-f32` with a scoped read-only borrow of an owned
source buffer available to both the term map and epilogue. The borrow cannot
escape. The operation returns the unchanged source owner together with the
new owned F32 or BF16 output buffer. Source element type `A` is independent of
output type `T`.

For each output, reduction uses the same exact adjacent-pair tree as
`materializec.reduce-f32`, including carrying unmatched terms unchanged and
using `+0.0` for an empty domain. GPU scheduling therefore cannot alter the
resulting bits. The current implementation supports `K ≤ 16384`, selecting
64, 128, 256, or 512 physical threads without changing that logical tree.

## `materializec.softmax-f32`

```text
materializec.softmax-f32[N, T] :
  Bufᵒʷⁿ(N, T)
  ⊗ ⟦N : U64⟧
  ⊗ Val(U64)
  ⊗ (Val(F32) ⊸ Val(F32))
  ⊸ Bufᵒʷⁿ(N, T)
```

The GPU backend accepts `T = F32` or `T = BF16`; maximum, exponential, and
denominator arithmetic is F32, and normalized values are stored as `T`.

The unindexed `Val(U64)` is the number of columns; it must be non-zero and
divide `N` exactly. The final map supplies the exponential operation. Each
contiguous row is normalized in place, consuming and returning ownership of
the same buffer.

Both row reductions use the exact adjacent-pair tree defined by
`materializec.reduce-f32`. Maximum selection retains the left operand unless
the right is strictly greater. A negative-infinity input produces a `+0.0`
numerator without invoking the exponential; every other numerator is
`exp(value - maximum)`. A zero denominator produces an all-`+0.0` row. The
current GPU implementation supports at most 16384 columns.

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
Qwen3 and Qwen3.5 family dense and MoE architectures. They live in the exploratory
compiler fork and are grouped separately from the features that were already present
for SmolLM2.

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
supports `K ≤ 8192`.

## `materializec.bf16-gemm-wmma`

```text
materializec.bf16-gemm-wmma[I, W, N, K] :
  Bufᵒʷⁿ(I, BF16)
  ⊗ ⟦I : U64⟧
  ⊗ Bufʳᵉᶠ(W, BF16)
  ⊗ ⟦W : U64⟧
  ⊗ ⟦N : U64⟧
  ⊗ ⟦K : U64⟧
  ⊗ Val(U64)
  ⊗ Val(U64)
  ⊸ Bufᵒʷⁿ(N, BF16)
```

The final values are `input_row` and `output_features`. The operation has the
same row-major shape, capacity, ownership, and BF16-output contract as
`materializec.bf16-gemv`: it computes selected rows of the input matrix times
the transpose of the row-major weight matrix, accumulates in F32, and rounds
once to BF16.

This operation is HIP/gfx11-only. For 64 or more input rows, one 32-lane wave
computes each 16×16 output tile with rocWMMA BF16 16×16×16 instructions and
F32 accumulator fragments. Shared-memory edge tiles zero-pad row, column, and
reduction tails. Inputs with fewer than 64 rows use the existing packed-BF16
SIMT kernel, which keeps token-at-a-time decode on its established path.

Unlike the regular primitive, this fast operation does not promise the
canonical adjacent-pair reduction order: WMMA groups and accumulates products
according to the matrix instruction. It is therefore selected explicitly by
the caller when that numerical tradeoff is acceptable.

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
`rows = I / K`. The operation requires `0 < K ≤ 8192`, `slots > 0`,
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

An annotated CUDA reading copy of the generated kernel is available at
[`catena-lang/doc/borrow_routed_bf16_gemv_pair.cu`](catena-lang/doc/borrow_routed_bf16_gemv_pair.cu).

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

## `materializec.bf16-gemv`

```text
materializec.bf16-gemv[I, W, N, K] :
  Bufᵒʷⁿ(I, BF16)
  ⊗ ⟦I : U64⟧
  ⊗ Bufʳᵉᶠ(W, BF16)
  ⊗ ⟦W : U64⟧
  ⊗ ⟦N : U64⟧
  ⊗ ⟦K : U64⟧
  ⊗ Val(U64)
  ⊗ Val(U64)
  ⊸ Bufᵒʷⁿ(N, BF16)
```

The unindexed values are `input_row` and `output_features`. The operation
multiplies selected rows of a row-major BF16 input by the transpose of a
row-major BF16 weight matrix. It requires non-zero `K`, `I` divisible by
`K`, `N` divisible by `output_features`, and capacities sufficient for the
selected input rows and `output_features × K` weights.

Each output uses 256 threads and an F32 block reduction. On supported AMD
gfx11 targets, packed BF16 pairs use `v_dot2_f32_bf16`; HIP targets without
that instruction and CUDA use scalar BF16-to-F32 products. An odd final
element is handled separately, and the completed F32 sum is rounded once to
BF16.

## `Runtime::mem_bf16_zeroed`

`Runtime::mem_bf16_zeroed(elements)` allocates an application-owned device
buffer of `elements × 2` bytes and initializes it to zero. Allocation size is
checked for overflow. It is the BF16 counterpart of `mem_f32_zeroed` and is
used for persistent BF16 attention and recurrent-state storage without a host
staging buffer.

## `materializec.borrow-argmax-bf16`

```text
materializec.borrow-argmax-bf16[N, One] :
  Bufᵒʷⁿ(N, BF16)
  ⊗ ⟦N : U64⟧
  ⊗ ⟦One : U64⟧
  ⊸ Bufᵒʷⁿ(N, BF16) ⊗ Bufᵒʷⁿ(One, U64)
```

The capacity and ownership rules match `borrow-argmax-f32`. BF16 bit patterns
are transformed into total-order keys directly, without first materializing an
F32 logits buffer. Equal bit patterns select the greatest index.

## Fused gated DeltaNet scans

The compiler exposes `materializec.borrow-gated-delta-net-f32` and
`materializec.borrow-gated-delta-net-bf16` as matching storage variants:

```text
materializec.borrow-gated-delta-net-{f32,bf16}[
  State, Q, K, V, Gate, Beta, Out
] :
  Bufᵒʷⁿ(State, S) ⊗ ⟦State : U64⟧
  ⊗ Bufᵒʷⁿ(Q, S) ⊗ ⟦Q : U64⟧
  ⊗ Bufᵒʷⁿ(K, S) ⊗ ⟦K : U64⟧
  ⊗ Bufᵒʷⁿ(V, S) ⊗ ⟦V : U64⟧
  ⊗ Bufᵒʷⁿ(Gate, S) ⊗ ⟦Gate : U64⟧
  ⊗ Bufᵒʷⁿ(Beta, S) ⊗ ⟦Beta : U64⟧
  ⊗ Val(U64)  # state_offset
  ⊗ Val(U64)  # tokens
  ⊗ Val(U64)  # key_heads
  ⊗ Val(U64)  # value_heads
  ⊗ Val(U64)  # dim
  ⊗ ⟦Out : U64⟧
  ⊸ Bufᵒʷⁿ(State, S) ⊗ Bufᵒʷⁿ(Q, S) ⊗ Bufᵒʷⁿ(K, S)
    ⊗ Bufᵒʷⁿ(V, S) ⊗ Bufᵒʷⁿ(Gate, S) ⊗ Bufᵒʷⁿ(Beta, S)
    ⊗ Bufᵒʷⁿ(Out, S)
```

`S` is F32 for `borrow-gated-delta-net-f32` and BF16 for
`borrow-gated-delta-net-bf16`. The five scalar values are `state_offset`,
`tokens`, `key_heads`, `value_heads`, and `dim`. Q and K use
`[token, key-head, dim]`; V/output use `[token, value-head, dim]`; gate and
beta use `[token, value-head]`. The state slice at `state_offset` contains
`value_heads × dim × dim` elements.

The operation requires non-zero head counts, `value_heads` divisible by
`key_heads`, exact input capacities, and `dim` equal to 16, 32, 64, or 128.
One kernel scans all supplied tokens while keeping each recurrent state row in
F32 registers and using warp/wave shuffles for its dot products. The BF16
variant widens every load and rounds only output and final-state stores. All
six source owners are returned after the launch.

## Tiled cached-attention context

The compiler exposes `materializec.borrow-cached-context-f32` and
`materializec.borrow-cached-context-bf16`:

```text
materializec.borrow-cached-context-{f32,bf16}[C, P, O] :
  Bufᵒʷⁿ(C, S) ⊗ ⟦C : U64⟧
  ⊗ Bufᵒʷⁿ(P, S) ⊗ ⟦P : U64⟧
  ⊗ ⟦O : U64⟧
  ⊗ Val(U64)  # sequence
  ⊗ Val(U64)  # history
  ⊗ Val(U64)  # token_capacity
  ⊗ Val(U64)  # layer
  ⊗ Val(U64)  # query_heads
  ⊗ Val(U64)  # key_value_heads
  ⊗ Val(U64)  # head_dim
  ⊸ Bufᵒʷⁿ(C, S) ⊗ Bufᵒʷⁿ(P, S) ⊗ Bufᵒʷⁿ(O, S)
```

`S` follows the operation suffix. The seven scalar values are `sequence`,
`history`, `token_capacity`, `layer`, `query_heads`,
`key_value_heads`, and `head_dim`. The cache is
`[layer, K-or-V, token, KV-head, channel]`, probabilities are
`[query-position, query-head, key-position]`, and output is
`[query-position, query-head, channel]`. Capacity and overflow checks prove
that all three layouts fit; query heads must be divisible by KV heads,
`history ≤ token_capacity`, and `1 ≤ head_dim ≤ 1024`.

A `32 × 8` block assigns adjacent threads to 32 contiguous value channels
while eight key lanes traverse history. It reduces a fixed `8 × 32` F32
shared tile, giving coalesced V-cache reads and 1 KiB shared memory per block
instead of history-sized dynamic shared memory. BF16 loads are widened and
each completed context channel is rounded once. This is still exact global
attention and therefore still visits all `history` values.

## `materializec.borrow-routed-native-bf16-gemv-pair`

```text
materializec.borrow-routed-native-bf16-gemv-pair[I, G, U, S, N, K] :
  Bufᵒʷⁿ(I, BF16) ⊗ ⟦I : U64⟧
  ⊗ Bufʳᵉᶠ(G, BF16) ⊗ ⟦G : U64⟧
  ⊗ Bufʳᵉᶠ(U, BF16) ⊗ ⟦U : U64⟧
  ⊗ Bufᵒʷⁿ(S, U64) ⊗ ⟦S : U64⟧
  ⊗ ⟦N : U64⟧ ⊗ ⟦K : U64⟧
  ⊗ Val(U64) ⊗ Val(U64)
  ⊸ Bufᵒʷⁿ(I, BF16) ⊗ Bufᵒʷⁿ(S, U64)
    ⊗ Bufᵒʷⁿ(N, BF16) ⊗ Bufᵒʷⁿ(N, BF16)
```

This has the same `slots`, `output_features`, selected-expert layout, and
capacity relationships as `borrow-routed-bf16-gemv-pair`, but input and both
outputs remain BF16. Gate and up products have independent F32 accumulators;
supported AMD targets use packed BF16 dot instructions, and each final sum is
rounded once.

## `materializec.borrow-routed-native-bf16-gemv-residual`

```text
materializec.borrow-routed-native-bf16-gemv-residual[A, D, S, R, N, K] :
  Bufᵒʷⁿ(A, BF16) ⊗ ⟦A : U64⟧
  ⊗ Bufʳᵉᶠ(D, BF16) ⊗ ⟦D : U64⟧
  ⊗ Bufᵒʷⁿ(S, U64) ⊗ ⟦S : U64⟧
  ⊗ Bufᵒʷⁿ(R, BF16) ⊗ ⟦R : U64⟧
  ⊗ ⟦N : U64⟧ ⊗ ⟦K : U64⟧
  ⊗ Val(U64)  # slots
  ⊗ Val(U64)  # intermediate
  ⊗ Val(U64)  # output_features
  ⊸ Bufᵒʷⁿ(A, BF16) ⊗ Bufᵒʷⁿ(S, U64)
    ⊗ Bufᵒʷⁿ(R, BF16) ⊗ Bufᵒʷⁿ(N, BF16)
```

The scalar values are `slots`, `intermediate`, and `output_features`;
`K = slots × intermediate`. Active values are
`[row, slot, intermediate]`, selected IDs are `[row, slot]`, and down
weights are `[expert, output-feature, intermediate]`. The kernel accumulates
all routed expert products in F32, adds the BF16 residual in F32, rounds once
to BF16, and returns all three owned inputs with the output.

## Deterministic reduction scheduling refinements

Qwen's 6144-term MoE down projection motivated a 512-thread launch for
`materializec.borrow-reduce-f32` when `K > 4096`. Thread count changes only
which lane materializes a term and performs an independent indexed addition;
the logical adjacent-pair tree is unchanged.

The Qwen work also replaces late whole-block barriers with explicit warp/wave
barriers in `materializec.reduce-f32`, `materializec.reduce-f32-pair`,
`materializec.borrow-reduce-f32`, and the routed BF16 pair. Once
`stride ≥ 256`, the maximum supported reduction length of 16384 leaves at most
32 active lanes, all within the first 32-lane CUDA warp or 64-lane AMD wave.
Every addition retains the same lane, logical index, left operand, right
operand, and parenthesization.

GPU tests compare every output bit with the canonical CPU tree at reduction
lengths 257, 2048, 6144, 8192, and 16384, including cancellation-sensitive values and
signed zero. The maximum-active-lane argument is part of the feature's
determinism requirement; increasing the supported reduction length requires
re-establishing it.

## GPU module save/load hook

`CATENA_GPU_MODULE_SAVE=<path>` copies a newly compiled GPU module to `path`.
`CATENA_GPU_MODULE_LOAD=<path>` loads that module instead of invoking the GPU
compiler. This opt-in runtime hook supports profiling tools that cannot safely
nest a `hipcc` invocation; normal runtime construction does not use it, and
the caller is responsible for cache validity.
