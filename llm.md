# Hex helpers and backend additions for f32 LLMs

The experimental SmolLM2 port contains three distinct kinds of work:

1. Model helpers defined in Hex, composed from supported operations.
2. Backend support for missing memory effects, ABI access, and numeric conversions.
3. Convenience adapters implemented in codegen, even though their underlying
   capabilities already existed and could be exposed through Hex helpers.

Here, a **new primitive capability** means an operation the experimental backend
could not previously lower, rather than every new arrow name. Arithmetic,
conditionals, ordered folds, GPU launches, and typed global-memory access
already existed. The default backend also already supported many of the memory
and conversion operations added to the experimental path.

## Helpers written in Hex

These definitions live in the sibling model repository under
[`src/catena/experimental`](../catena-smollm2-135/src/catena/experimental).
They compose operations into model calculations; none has a model-specific Rust
lowering or C++ renderer. A helper can call a backend primitive without itself
being a primitive.

| Hex helper or group | What it implements | Rationale |
| --- | --- | --- |
| `smollm2.exp` and `stdlib.smollm2.exp-nonneg` | ln(2) range reduction, a degree-six polynomial, exponent reconstruction, and negative-input handling. | Preserve the reference model's exponential approximation for softmax and SiLU. The polynomial and its constants belong in Hex. |
| `smollm2.sqrt` | An exponent-derived initial estimate and four Newton-Raphson refinements. | Preserve the reference RMSNorm numerics across differently sized activations. The iterative algorithm belongs in Hex; bit representation access is supplied by the backend. |
| `smollm2.silu` | `x / (1 + exp(-x))`. | Build the SwiGLU activation from arithmetic and the Hex exponential helper. |
| `smollm2.embedding` and its kernel | Token-to-weight lookup producing `[sequence, 576]` activations. | Express embedding layout and indexing in the model, rather than adding an embedding compiler operation. |
| `smollm2.rms-inv` and its reduction step | Ordered sum of squares, division by hidden size, epsilon addition, square root, and reciprocal. | Define RMSNorm using a reusable reduction and scalar helpers. |
| `smollm2.linear-norm` and its reduction step | Learned normalization followed by a bias-free projection. | Preserve the reference multiplication order: hidden value × norm weight × inverse RMS × projection weight. |
| `stdlib.smollm2.rope-value` | Non-interleaved rotary channel pairing and rotation using the supplied sine/cosine table. | Keep the model's RoPE convention and tensor layouts explicit in Hex. |
| Attention score, row-max, denominator, and context helpers | Rotary Q/K dot products, grouped-query head mapping, causal masking, stable softmax, and weighted V sums. | Compose attention from indexed loads, scalar arithmetic, and ordered reductions; no attention backend primitive is needed. |
| `smollm2.linear-residual` | A projection followed by residual addition. | Define the projection's residual epilogue in Hex. |
| `smollm2.mlp-down-residual` | SiLU(gate) × up, down projection, and residual addition. | Define SwiGLU and its fused projection in Hex. |
| `stdlib.smollm2.free-before` | Release one temporary after a dependent output has been produced, then forward that output. | Express deallocation ordering through the graph. The helper sequences an existing primitive; the actual device free requires backend support. |
| `stdlib.smollm2.decoder-layer` | Compose attention, residuals, MLP work, and temporary cleanup. | Make layer structure and buffer lifetimes model code. |
| Final-logit helper and `smollm2` | Final normalization, tied-embedding vocabulary projection, and explicit composition of 30 decoder layers. | Keep the architecture, weight order, and public model interface outside the compiler. |

Kernel bodies and host wrappers are also Hex definitions. The host wrappers
allocate outputs and launch index callbacks; the callbacks calculate and write
individual output elements. Float calculations and intermediates remain f32.

## Missing operations supplied by backend additions

The declarations are in
[`execution.hex`](catena-lang/stdlib/experimental/execution.hex). Their native
semantics are implemented in
[`ops/raw.rs`](catena-lang/src/codegen/experimental/ops/raw.rs) and
[`prelude.rs`](catena-lang/src/codegen/experimental/prelude.rs).

These are the missing backend operations used by the current port to retain its
buffer ABI and scalar implementation. Declaring an arrow in Hex does not give
it native semantics: the corresponding lowering must exist. This describes
requirements of this implementation, rather than claiming that every numeric
conversion has no possible alternative arithmetic algorithm.

| Backend addition | Native code or effect | Why backend support is needed |
| --- | --- | --- |
| `stdlib.runtime.raw.alloc_f32` | Host call to `hipMalloc` or `cudaMalloc`, returning owned pointer/byte-length metadata. | The previous experimental interface had no allocation operation. Hex must obtain output and temporary buffers somewhere; arithmetic and launches alone cannot allocate device storage. |
| `stdlib.runtime.raw.free` | Host call to `hipFree` or `cudaFree`. | The previous experimental interface had no deallocation operation. Cleanup must release native allocations after their final consumer. |
| `stdlib.runtime.raw.bytes` | Read the ABI buffer's `.len` field as u64. | The existing typed-global conversion required an element count supplied by the caller. It did not expose the incoming ABI byte length, which this model uses to recover sequence and tensor sizes without adding shape arguments. |
| `stdlib.numeric.raw.to_f32` | Numeric C++ conversion from u32 or u64 to float. | Integer-to-f32 conversion was absent from experimental lowering. The existing exponential helper needs the rounded integer exponent converted back to f32 for range reduction. |
| `stdlib.numeric.raw.to_u32` | Assert `value <= UINT32_MAX`, then convert u64 to u32. | Existing experimental coercion supported widening u32 to u64, not this checked narrowing. It bridges u64 token IDs and ABI-derived sizes to u32 indexing. |
| `stdlib.numeric.raw.bits_u32` | Reinterpret an f32 bit pattern as u32 through a generated support function. | Experimental codegen already had the reverse `u32.bitcast-f32`, but lacked f32 bit-pattern extraction. The reference square-root estimate uses those exponent bits. |
| `stdlib.numeric.raw.round_u32` | `nearbyintf`, finite/range checks, and conversion to u32. | Rounded f32-to-u32 conversion was absent from experimental lowering. The reference exponential approximation uses it to choose the integer exponent. |

Allocation takes a u32 element count, computes bytes in u64, and checks that the
size fits `size_t`. A zero-sized allocation has a null pointer and zero length;
freeing a null pointer is a no-op. Allocation and free require host execution
and check GPU API errors. `nearbyintf` follows the rounding mode; the usual
default is nearest with ties to even.

### Additional ABI bridge, not required by the current model

`stdlib.runtime.raw.borrow` emits a `MemRef` with the same pointer and byte
length as an owned buffer, returning both the owner and reference. This adds a
bridge between ABI ownership representations. It can support read-only helper
interfaces, but the current experimental SmolLM2 Hex does not call it: its read
adapters accept owned buffers directly. It should not be counted as a necessary
primitive for this model.

## Codegen convenience adapters over existing capabilities

The following additions currently emit native code too, but that fact does not
make them new primitive capabilities. They simplify plain `val` interfaces in
the port. They have not yet been replaced with Hex wrappers.

| Current adapter | Existing capability or Hex implementation route | Rationale for the adapter |
| --- | --- | --- |
| `stdlib.numeric.raw.add`, `sub`, `mul`, `div`, `rem`, `max` | Existing `stdlib.numeric` arithmetic on named values and Numeric/Integral evidence. | Avoid carrying symbolic identities and evidence through every scalar calculation. Arithmetic itself was already supported. |
| `stdlib.numeric.raw.less`, `less_equal`, `equal`, `negate` | Existing numeric comparisons and negation. | Provide plain-value signatures for masks, indices, and scalar helpers. |
| `stdlib.numeric.raw.select` | Existing `core.if`/`core.if_guarded` with callbacks returning the alternatives. | Provide a compact value-selection operation. The current adapter selects already computed values; it does not make their producers lazy. |
| `stdlib.numeric.raw.to_u64` | Existing `stdlib.numeric.coerce` with u32-to-u64 subtype evidence. | Expose existing widening as a plain-value helper for size calculations. |
| `stdlib.numeric.raw.shr` | For a shift count below 32, unsigned division by `2^count`; the divisor can be built with a Hex fold. | Emit a direct right shift for the square-root estimate. A shift instruction is convenient, but does not add an essential capability beyond integer arithmetic and folds. |
| `core.fold.values` | Existing `core.fold.bounded` lowering, with suitable Hex adaptation of callback and invariant/proof operands. | Carry plain state through an ascending-index loop without exposing the proof/invariant interface to each model helper. |
| `unsafe.launch_linear` | Existing geometry construction, `unsafe.launch`, index extraction, and a conditional bounds guard. | Package a positive element count into 256-thread blocks and guard padded threads. GPU launch capability already existed. |
| `stdlib.runtime.raw.read_f32`, `read_u64` | Existing `stdlib.runtime.global_own_*`/`global_ref_*` bridges and `stdlib.gpu.memory.global_read`. | Read directly from ABI buffers with runtime element-size, alignment, count, and index checks. Loads were already supported. |
| `stdlib.runtime.raw.write_f32` | Existing typed-global bridge and `stdlib.gpu.memory.global_write`. | Write directly to an owned ABI output buffer with runtime checks. Stores were already supported. |

The existing global-memory bridges can be used once the element count is known;
for this unchanged buffer ABI, the new `bytes` operation supplies that missing
metadata. Replacing the direct read/write adapters with Hex helpers would still
need that metadata access.

`core.fold.values` preserves state shape and runtime field types and updates
multiple fields simultaneously. `unsafe.launch_linear` synchronizes before
returning, as the existing launcher does. These choices reproduce ordered
reductions and completed tensor materializations without adding model-specific
compiler algorithms.

Memory checks do not establish alias lifetimes, race freedom, or uniform barrier
participation. Hex definitions must preserve buffer owners until their final
use and assign writes appropriately. Ordinary integer arithmetic is not
checked for overflow; the model widens size products before checked narrowing.
Separate f32 multiply/add expressions and the existing native compilation flags
preserve the reference arithmetic order without implicit contraction to FMA.

## Integration and validation

The model combines its Hex definitions with the experimental library, runs the
shared front end, and selects `CodegenKind::Experimental`. These additions do
not change `RuntimeModule`, `Runtime::load`, or `Artifact::exec`, and do not
change the default backend. Weight loading, tokenization, and construction of
RoPE tables remain application responsibilities.

Compiler tests cover type errors, host/device placement, erased-result effects,
checked narrowing, and HIP/CUDA support. Model tests cover full compilation,
public buffer ABI compatibility, and opt-in numerical comparisons with the
reference implementation. Native HIP compilation and host scalar checks have
passed; full GPU inference parity still requires a HIP-capable machine.
