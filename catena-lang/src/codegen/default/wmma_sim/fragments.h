
// gfx11 WMMA requires a complete wave32 and duplicated A/B lanes 0..15/16..31.
struct catena_wmma_bf16x16_t { uint32_t pair[8]; };
struct catena_wmma_f32x8_t { float value[8]; };
__device__ inline catena_wmma_bf16x16_t catena_wmma_pack(catena_bf16x2_t a, catena_bf16x2_t b, catena_bf16x2_t c, catena_bf16x2_t d, catena_bf16x2_t e, catena_bf16x2_t f, catena_bf16x2_t g, catena_bf16x2_t h) { return {{a.bits,b.bits,c.bits,d.bits,e.bits,f.bits,g.bits,h.bits}}; }
__device__ inline catena_wmma_f32x8_t catena_wmma_zero() { return {}; }
__device__ inline float catena_wmma_get(catena_wmma_f32x8_t x, uint32_t i) { catena_assert(i < 8); return x.value[i]; }
__device__ inline catena_wmma_f32x8_t catena_wmma_native(catena_wmma_bf16x16_t a, catena_wmma_bf16x16_t b, catena_wmma_f32x8_t c) {
#if defined(__HIP_DEVICE_COMPILE__) && (defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__) || defined(__gfx1150__) || defined(__gfx1151__) || defined(__gfx1152__) || defined(__gfx1153__))
    typedef short bf16x16 __attribute__((ext_vector_type(16)));
    typedef float f32x8 __attribute__((ext_vector_type(8)));
    bf16x16 av, bv; f32x8 cv;
    __builtin_memcpy(&av, &a, sizeof(av)); __builtin_memcpy(&bv, &b, sizeof(bv)); __builtin_memcpy(&cv, &c, sizeof(cv));
    cv = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(av, bv, cv);
    __builtin_memcpy(&c, &cv, sizeof(cv)); return c;
#else
    catena_assert(false); return c;
#endif
}

// All participating lanes must execute every shuffle with the same mask.
// Keep fragment storage out of the native caller (its LDS must stay available
// for model tiles). Unrolling below lets the simulator use fixed registers.
__device__ __noinline__ catena_wmma_f32x8_t catena_wmma_simulate(
    catena_wmma_bf16x16_t a, catena_wmma_bf16x16_t b, catena_wmma_f32x8_t c) {
    const unsigned linear = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
    const unsigned parity = (linear & 31u) / 16;
    uint32_t accumulator[8];
    #pragma unroll
    for (unsigned i = 0; i < 8; ++i)
        accumulator[i] = catena_f32_bitcast_u32(c.value[i]);
    // Advance all eight independent outputs together. Only the eight k steps
    // within one output are ordered; the GPU can interleave different outputs.
    // Roll k to keep the instruction footprint small. A fixed register window
    // avoids dynamic array indices while advancing the input pairs in order.
    #pragma unroll 1
    for (unsigned k = 0; k < 8; ++k) {
        const uint32_t own_a = a.pair[0], bp = b.pair[0];
        #pragma unroll
        for (unsigned j = 0; j < 7; ++j) {
            a.pair[j] = a.pair[j+1];
            b.pair[j] = b.pair[j+1];
        }
        #pragma unroll
        for (unsigned i = 0; i < 8; ++i) {
#if defined(__CUDA_ARCH__)
            const uint32_t ap = __shfl_sync(0xffffffffu, own_a, parity + 2*i, 32);
#elif defined(__HIP_DEVICE_COMPILE__)
            const uint32_t ap = __shfl(own_a, parity + 2*i, 32);
#else
            const uint32_t ap = 0; // Host compiler pass never executes device code.
#endif
            accumulator[i] = catena_wmma_sim::simulate_dot2(
                uint16_t(ap), uint16_t(bp), uint16_t(ap >> 16), uint16_t(bp >> 16), accumulator[i]);
        }
    }
    #pragma unroll
    for (unsigned i = 0; i < 8; ++i)
        c.value[i] = catena_u32_bitcast_f32(accumulator[i]);
    return c;
}
// Host dispatch selects a kernel specialization. Native kernels must never
// retain a reachable simulator call: its ABI can force native MMA spills.
template<bool Emulate>
__device__ inline catena_wmma_f32x8_t catena_wmma_mma(
    catena_wmma_bf16x16_t a, catena_wmma_bf16x16_t b, catena_wmma_f32x8_t c) {
    if constexpr (Emulate) {
        return catena_wmma_simulate(a, b, c);
    } else {
        return catena_wmma_native(a, b, c);
    }
}
// Read on the host for each launch. No mutable module/device state is needed.
inline bool catena_wmma_emulation_enabled() {
    const char* value = getenv("CATENA_WMMA_EMULATE");
    return value && value[0] == '1' && value[1] == '\0';
}
