//! BF16 SDK operations and backend-specific packed dot accumulation.
use crate::runtime::GpuDialect;

fn native_support(dialect: GpuDialect) -> String {
    let (header, native_type, from_f32) = match dialect {
        GpuDialect::Hip => (
            "hip/hip_bf16.h",
            "__hip_bfloat16",
            "__float2bfloat16(value)",
        ),
        GpuDialect::Cuda => ("cuda_bf16.h", "__nv_bfloat16", "__float2bfloat16_rn(value)"),
    };
    format!(
        r#"#include <{header}>
typedef {native_type} catena_bf16_t;
static_assert(sizeof(catena_bf16_t) == 2, "BF16 must occupy two bytes");
static_assert(alignof(catena_bf16_t) == 2, "BF16 must have two-byte alignment");

__host__ __device__ static inline catena_bf16_t catena_bf16_from_f32(float value) {{
    return {from_f32};
}}
__host__ __device__ static inline float catena_bf16_to_f32(catena_bf16_t value) {{
    return __bfloat162float(value);
}}
__host__ __device__ static inline catena_bf16_t catena_u16_bitcast_bf16(uint16_t bits) {{
    return __ushort_as_bfloat16(bits);
}}
__host__ __device__ static inline uint16_t catena_bf16_bitcast_u16(catena_bf16_t value) {{
    return __bfloat16_as_ushort(value);
}}
"#,
    )
}

pub(super) fn render(dialect: GpuDialect, dot: bool, wmma: bool) -> String {
    let mut source = native_support(dialect);
    source.push_str(r#"
__host__ __device__ inline catena_bf16_t catena_bf16_add(catena_bf16_t a,catena_bf16_t b) { return __hadd_rn(a,b); }
__host__ __device__ inline catena_bf16_t catena_bf16_sub(catena_bf16_t a,catena_bf16_t b) { return __hsub_rn(a,b); }
__host__ __device__ inline catena_bf16_t catena_bf16_mul(catena_bf16_t a,catena_bf16_t b) { return __hmul_rn(a,b); }
__host__ __device__ inline catena_bf16_t catena_bf16_div(catena_bf16_t a,catena_bf16_t b) { return __hdiv(a,b); }
__host__ __device__ inline catena_bf16_t catena_bf16_neg(catena_bf16_t a) { return __hneg(a); }
__device__ inline catena_bf16_t catena_bf16_exp(catena_bf16_t a) { return hexp(a); }
__device__ inline catena_bf16_t catena_bf16_sqrt(catena_bf16_t a) { return hsqrt(a); }
__device__ inline catena_bf16_t catena_bf16_rsqrt(catena_bf16_t a) { return hrsqrt(a); }
struct alignas(4) catena_bf16x2_t { uint32_t bits; };
static_assert(sizeof(catena_bf16x2_t)==4 && alignof(catena_bf16x2_t)==4,"packed BF16 must occupy four aligned bytes");
// Byte copies permit packed access to storage previously written as BF16 scalars.
__host__ __device__ inline catena_bf16x2_t catena_bf16_load_pair(const void* ptr) { catena_bf16x2_t out; __builtin_memcpy(&out,ptr,sizeof(out)); return out; }
__host__ __device__ inline void catena_bf16_store_pair(void* ptr,catena_bf16x2_t value) { __builtin_memcpy(ptr,&value,sizeof(value)); }
__host__ __device__ inline catena_bf16x2_t catena_bf16_pack(catena_bf16_t low,catena_bf16_t high) { return {uint32_t(catena_bf16_bitcast_u16(low)) | (uint32_t(catena_bf16_bitcast_u16(high))<<16)}; }
__host__ __device__ inline catena_bf16_t catena_bf16_low(catena_bf16x2_t p) { return catena_u16_bitcast_bf16(uint16_t(p.bits)); }
__host__ __device__ inline catena_bf16_t catena_bf16_high(catena_bf16x2_t p) { return catena_u16_bitcast_bf16(uint16_t(p.bits>>16)); }
"#);
    if dot {
        source.push_str(r#"
#if defined(__HIP_DEVICE_COMPILE__) && !(defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__) || defined(__gfx1150__) || defined(__gfx1151__) || defined(__gfx1152__) || defined(__gfx1153__))
#error "native packed BF16 dot requires a supported gfx11 target"
#endif
__host__ __device__ inline float catena_bf16_dot(catena_bf16x2_t a,catena_bf16x2_t b,float acc) {
#if defined(__HIP_DEVICE_COMPILE__)
    typedef __bf16 bf16x2 __attribute__((ext_vector_type(2)));
    union { uint32_t bits; bf16x2 values; } x,y;
    x.bits=a.bits; y.bits=b.bits;
    return __builtin_amdgcn_fdot2_f32_bf16(x.values,y.values,acc,false);
#elif defined(__CUDA_ARCH__)
    // CUDA has no per-thread equivalent of AMD fdot2. Keep BF16 storage and
    // evaluate the pair in double before a single F32 boundary conversion.
    // Explicit round-to-nearest intrinsics prevent contraction/reassociation.
    double lo = __dmul_rn(double(catena_bf16_to_f32(catena_bf16_low(a))), double(catena_bf16_to_f32(catena_bf16_low(b))));
    double hi = __dmul_rn(double(catena_bf16_to_f32(catena_bf16_high(a))), double(catena_bf16_to_f32(catena_bf16_high(b))));
    return __double2float_rn(__dadd_rn(__dadd_rn(lo, hi), double(acc)));
#elif defined(CATENA_BF16_CPU_EMULATION)
    // Test-only emulator: never a device fallback. One rounding after the pair sum.
    return float(double(catena_bf16_to_f32(catena_bf16_low(a)))*double(catena_bf16_to_f32(catena_bf16_low(b))) + double(catena_bf16_to_f32(catena_bf16_high(a)))*double(catena_bf16_to_f32(catena_bf16_high(b))) + double(acc));
#else
    catena_assert(false); return 0.0f;
#endif
}
"#);
    }
    if wmma {
        source.push_str(r#"
// gfx11 WMMA requires a complete wave32 and duplicated A/B lanes 0..15/16..31.
struct catena_wmma_bf16x16_t { uint32_t pair[8]; };
struct catena_wmma_f32x8_t { float value[8]; };
__device__ inline catena_wmma_bf16x16_t catena_wmma_pack(catena_bf16x2_t a, catena_bf16x2_t b, catena_bf16x2_t c, catena_bf16x2_t d, catena_bf16x2_t e, catena_bf16x2_t f, catena_bf16x2_t g, catena_bf16x2_t h) { return {{a.bits,b.bits,c.bits,d.bits,e.bits,f.bits,g.bits,h.bits}}; }
__device__ inline catena_wmma_f32x8_t catena_wmma_zero() { return {}; }
__device__ inline float catena_wmma_get(catena_wmma_f32x8_t x, uint32_t i) { catena_assert(i < 8); return x.value[i]; }
__device__ inline catena_wmma_f32x8_t catena_wmma_mma(catena_wmma_bf16x16_t a, catena_wmma_bf16x16_t b, catena_wmma_f32x8_t c) {
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
"#);
    }
    source
}
