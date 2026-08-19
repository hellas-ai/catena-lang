// Annotated reading copy of the CUDA kernel emitted for
// `materializec.borrow-routed-bf16-gemv-pair`.
//
// The source of truth is `render_borrow_routed_bf16_gemv_pair_kernel` in
// `src/codegen/ops/materializec.rs`.  Catena normally emits this kernel into a
// larger generated CUDA translation unit and gives it a program-specific name.
// This file supplies just enough of Catena's CUDA prelude to make the copied
// kernel understandable (and independently compilable with `nvcc -c`).
//
// Tensor shapes and row-major layouts:
//
//   input       [rows][reduction_len]                         F32
//   selected    [rows][slots]                                U64 expert IDs
//   gate_weight [expert_count][output_features][reduction_len] BF16
//   up_weight   [expert_count][output_features][reduction_len] BF16
//   gate_out    [rows][slots][output_features]                F32
//   up_out      [rows][slots][output_features]                F32
//
// For every (row, slot), `selected[row][slot]` chooses one expert.  The kernel
// applies both that expert's gate matrix and its up matrix to input[row].  It
// is therefore a pair of routed matrix-vector products, fused so they share
// index calculation, input loads, and reduction barriers.
//
// Catena launches it as:
//
//   grid size:       output_len blocks (one block per output scalar)
//   block size:      256 threads
//   dynamic shared:  2 * reduction_len * sizeof(float) bytes
//
// The launch-side lowering checks the shape relationships above and requires
// 0 < reduction_len <= 4096.  The maximum dynamic shared allocation is thus
// 2 * 4096 * 4 = 32768 bytes.

#include <cuda_bf16.h>
#include <stdint.h>

// These are the relevant definitions copied from Catena's generated CUDA
// prelude.  They keep the kernel body below close to the code actually emitted.
typedef __nv_bfloat16 catena_bf16_t;

__device__ static inline float catena_bf16_to_f32(catena_bf16_t value) {
    return __bfloat162float(value);
}

__device__ static inline void catena_assert(uint8_t condition) {
    if (!condition) {
        __builtin_trap();
    }
}

__global__ void borrow_routed_bf16_gemv_pair_kernel(
    const float *input,
    const catena_bf16_t *gate_weight,
    const catena_bf16_t *up_weight,
    const uint64_t *selected,
    float *gate_out,
    float *up_out,
    uint64_t output_len,
    uint64_t reduction_len,
    uint64_t slots,
    uint64_t output_features,
    uint64_t expert_count
) {
    // A block owns one scalar in each output tensor.  The actual launch uses
    // exactly output_len blocks, but this guard makes an oversized grid safe.
    uint64_t output_index = (uint64_t)blockIdx.x;
    if (output_index >= output_len) {
        return;
    }

    // Flattening [row][slot][output_feature] means one row occupies this many
    // output scalars.  Reverse that flattening to find the work for this block.
    uint64_t active_width = slots * output_features;
    uint64_t row = output_index / active_width;
    uint64_t row_element = output_index % active_width;
    uint64_t slot = row_element / output_features;
    uint64_t output_feature = row_element % output_features;

    // Routing varies by (row, slot), not by output feature.  Every output-
    // feature block for the same pair reads the same selected expert ID.
    uint64_t expert = selected[row * slots + slot];
    catena_assert(expert < expert_count);

    // input_base addresses input[row][0].  weight_base addresses
    // weight[expert][output_feature][0] in either weight tensor.
    uint64_t input_base = row * reduction_len;
    uint64_t weight_base =
        (expert * output_features + output_feature) * reduction_len;

    // The launch provides two adjacent arrays of reduction_len floats.  The
    // arrays hold every product explicitly because Catena promises a precise,
    // deterministic adjacent-pair addition tree rather than an arbitrary dot-
    // product accumulation order.
    extern __shared__ float reduction_terms[];
    float *gate_terms = reduction_terms;
    float *up_terms = reduction_terms + reduction_len;

    // The 256 threads cooperatively form all terms in both dot products.  When
    // reduction_len is greater than 256, each thread handles multiple indices.
    // Threads in a block read contiguous input and weight elements in each pass.
    for (
        uint64_t reduction_index = (uint64_t)threadIdx.x;
        reduction_index < reduction_len;
        reduction_index += (uint64_t)blockDim.x
    ) {
        // Load input once and reuse it for the two independently rounded F32
        // products.  Each BF16 weight is widened to F32 before multiplication.
        float input_value = input[input_base + reduction_index];
        gate_terms[reduction_index] =
            input_value *
            catena_bf16_to_f32(gate_weight[weight_base + reduction_index]);
        up_terms[reduction_index] =
            input_value *
            catena_bf16_to_f32(up_weight[weight_base + reduction_index]);
    }

    // No reduction may start until every product in shared memory is ready.
    __syncthreads();

    // Reduce the two arrays in lockstep, but never mix their arithmetic.
    //
    // stride = 1:  [0] += [1], [2] += [3], [4] += [5], ...
    // stride = 2:  [0] += [2], [4] += [6], ...
    // stride = 4:  [0] += [4], ...
    //
    // Thus non-power-of-two lengths also retain the exact left-anchored,
    // adjacent-pair tree specified by `materializec.reduce-f32-pair`.
    for (uint64_t stride = 1; stride < reduction_len; stride <<= 1) {
        uint64_t step = stride << 1;

        // Mapping `left = threadIdx.x * step` assigns each pair at this level
        // to one thread.  The loop matters at early levels, where there can be
        // more pairs than the block's 256 threads.
        for (
            uint64_t left = (uint64_t)threadIdx.x * step;
            left + stride < reduction_len;
            left += (uint64_t)blockDim.x * step
        ) {
            gate_terms[left] = gate_terms[left] + gate_terms[left + stride];
            up_terms[left] = up_terms[left] + up_terms[left + stride];
        }

        // Up through stride 128, work can span several warps, so the whole
        // block must rendezvous.  From stride 256 onward, reduction_len <= 4096
        // guarantees that all remaining active threads are in warp 0; a warp
        // barrier is sufficient and avoids a full-block barrier.
        if (stride < 256) {
            __syncthreads();
        } else {
            __syncwarp();
        }
    }

    // Element zero is now the root of each tree.  Only one thread publishes
    // the pair.  (The validated primitive never has reduction_len == 0, though
    // the emitted kernel preserves the generic reduction's empty-sum rule.)
    if (threadIdx.x == 0) {
        gate_out[output_index] =
            reduction_len == 0 ? 0.0f : gate_terms[0];
        up_out[output_index] =
            reduction_len == 0 ? 0.0f : up_terms[0];
    }
}
