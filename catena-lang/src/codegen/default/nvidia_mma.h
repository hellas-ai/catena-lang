// NVIDIA m16n8k16 row.col: all 32 lanes must participate convergently.
// This instruction does not synchronize shared-memory staging.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
#error "NVIDIA BF16 m16n8k16 MMA requires sm_80 or later"
#endif
struct catena_mma_bf16_a_t { uint32_t pair[4]; };
struct catena_mma_bf16_b_t { uint32_t pair[2]; };
struct catena_mma_f32_acc_t { float elt[4]; };
static_assert(sizeof(catena_mma_bf16_a_t)==16, "MMA A requires four U32 registers");
static_assert(sizeof(catena_mma_bf16_b_t)==8, "MMA B requires two U32 registers");
static_assert(sizeof(catena_mma_f32_acc_t)==16, "MMA accumulator requires four F32 registers");
__device__ inline catena_mma_bf16_a_t catena_mma_a_pack(catena_bf16x2_t a, catena_bf16x2_t b, catena_bf16x2_t c, catena_bf16x2_t d) { return {{a.bits,b.bits,c.bits,d.bits}}; }
__device__ inline catena_mma_bf16_b_t catena_mma_b_pack(catena_bf16x2_t a, catena_bf16x2_t b) { return {{a.bits,b.bits}}; }
__device__ inline float catena_mma_acc_get(catena_mma_f32_acc_t a, uint32_t i) { catena_assert(i < 4); return a.elt[i]; }
__device__ inline catena_mma_f32_acc_t catena_mma_bf16_f32(catena_mma_bf16_a_t a, catena_mma_bf16_b_t b, catena_mma_f32_acc_t c) {
    catena_mma_f32_acc_t d{};
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    // Explicit PTX retains MMA arithmetic even with --fmad=false.
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
        : "=f"(d.elt[0]), "=f"(d.elt[1]), "=f"(d.elt[2]), "=f"(d.elt[3])
        : "r"(a.pair[0]), "r"(a.pair[1]), "r"(a.pair[2]), "r"(a.pair[3]),
          "r"(b.pair[0]), "r"(b.pair[1]),
          "f"(c.elt[0]), "f"(c.elt[1]), "f"(c.elt[2]), "f"(c.elt[3]));
#endif
    return d;
}
