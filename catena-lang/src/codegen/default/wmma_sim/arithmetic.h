// Portable GPU BF16 arithmetic adapted from the standalone RDNA model.
// Empirical gfx1151 model; keep all rounding and exceptional values bit-exact.
#if defined(__CUDACC__) || defined(__HIPCC__)
#define CATENA_WMMA_NOINLINE __noinline__
#else
#define CATENA_WMMA_NOINLINE __attribute__((noinline))
#endif
namespace catena_wmma_sim {
constexpr uint32_t OPERAND_NAN = 0xffc00a3du;
constexpr uint32_t INVALID_NAN = 0xffc00000u;
constexpr uint32_t INF = 0x7f800000u;
struct Decoded {
    uint32_t significand;
    int exponent; // value = significand * 2**(exponent - fraction_bits)
    bool sign, infinity, nan;
};
__host__ __device__ inline Decoded decode16(uint16_t bits) {
    constexpr int fraction = 7;
    constexpr int bias = 127;
    constexpr int exp_mask = 255;
    const int e = (bits >> fraction) & exp_mask;
    const uint32_t m = bits & ((1u << fraction) - 1);
    // BF16 subnormals flush to signed zero.
    const uint32_t sig = e ? ((1u << fraction) | m) : 0u;
    return {sig, e ? e - bias : 1 - bias, bool(bits & 0x8000),
            e == exp_mask && m == 0, e == exp_mask && m != 0};
}
__host__ __device__ inline Decoded decode32(uint32_t bits) {
    const int e = (bits >> 23) & 255;
    const uint32_t m = bits & 0x7fffff;
    return {e ? (0x800000u | m) : 0u, e ? e - 127 : -126,
            bool(bits >> 31), e == 255 && m == 0, e == 255 && m != 0};
}
__host__ __device__ inline uint32_t align_unsigned(uint32_t significand, int shift) {
    if (!significand) return 0;
    if (shift >= 0) return significand << shift; // shift/result bounded by 27 bits here
    return shift <= -32 ? 0 : significand >> -shift;
}
__host__ __device__ inline int32_t align_signed(uint32_t significand, int shift, bool sign) {
    const int32_t aligned = static_cast<int32_t>(align_unsigned(significand, shift));
    return sign ? -aligned : aligned; // truncation toward zero
}
// E >= ec-2 >= -128, so power=E-24 >= -152. Aligned products
// occupy at most 26 bits and C at most 27; their sum fits in signed 29 bits.
// Round the small integer with the GPU's RNE conversion, then adjust the FP32
// exponent as bits. This avoids general 64-bit alignment and software rounding.
__host__ __device__ inline uint32_t round_result(int32_t s, int power) {
    const uint32_t magnitude = s < 0 ? uint32_t(-s) : uint32_t(s);
    // Exact cutoff: (2^25 - 1) * 2^-151. For power > -151,
    // ceil(cutoff / 2^power) is a power of two. Clamp shift counts so both
    // candidates are safe to evaluate, allowing predicated GPU instructions.
    const int shift = -126 - power;
    const unsigned normal_shift = shift < 0 ? 0u : shift > 24 ? 24u : unsigned(shift);
    const uint32_t low_cutoff = ((1u << 25) - 1) << unsigned(power == -152);
    const uint32_t normal_cutoff = 1u << normal_shift;
    const bool underflow = power <= -151 ? magnitude < low_cutoff
                                        : power < -126 && magnitude < normal_cutoff;
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    const float rounded = __int2float_rn(s);
#else
    const float rounded = float(s); // CPU stress runs set FE_TONEAREST.
#endif
    uint32_t bits;
    __builtin_memcpy(&bits, &rounded, sizeof(bits));
    const int exponent = int((bits >> 23) & 255) + power;
    const uint32_t normal = (bits & 0x807fffffu) | (uint32_t(exponent) << 23);
    const uint32_t result = exponent >= 255 ? (bits & 0x80000000u) | INF : normal;
    return s && !underflow ? result : 0u;
}

// Nonfinite handling is cold in model kernels. Keep it out of the unrolled
// finite path, retaining the same NaN precedence and signed infinities.
__host__ __device__ CATENA_WMMA_NOINLINE uint32_t exceptional_dot2(uint16_t a0, uint16_t b0, uint16_t a1, uint16_t b1, uint32_t cbits) {
    const Decoded a[2] = {decode16(a0), decode16(a1)};
    const Decoded b[2] = {decode16(b0), decode16(b1)};
    const Decoded c = decode32(cbits);
    bool positive_inf = c.infinity && !c.sign;
    bool negative_inf = c.infinity && c.sign;
    if (a[0].nan || a[1].nan || b[0].nan || b[1].nan) return OPERAND_NAN;
    for (int i = 0; i < 2; ++i) {
        const bool sign = a[i].sign != b[i].sign;
        const bool invalid = (a[i].infinity && !b[i].significand && !b[i].infinity) ||
                             (b[i].infinity && !a[i].significand && !a[i].infinity);
        if (invalid) return INVALID_NAN;
        if (a[i].infinity || b[i].infinity) {
            positive_inf |= !sign; negative_inf |= sign;
        }
    }
    if (positive_inf && negative_inf) return INVALID_NAN;
    if (c.nan) return cbits | 0x00400000u;
    // Entry requires at least one nonfinite input, so a surviving case is infinity.
    return (negative_inf ? 0x80000000u : 0) | INF;
}

// Independent output chains use only 32-bit ALU operations and one RNE
// integer-to-F32 conversion per pair. No native dot2/MMA instruction is used.
__host__ __device__ inline uint32_t simulate_dot2(uint16_t a0, uint16_t b0, uint16_t a1, uint16_t b1, uint32_t cbits) {
    if ((a0 & 0x7f80u) == 0x7f80u || (b0 & 0x7f80u) == 0x7f80u ||
        (a1 & 0x7f80u) == 0x7f80u || (b1 & 0x7f80u) == 0x7f80u ||
        (cbits & INF) == INF)
        return exceptional_dot2(a0, b0, a1, b1, cbits);
    const Decoded x0 = decode16(a0), y0 = decode16(b0);
    const Decoded x1 = decode16(a1), y1 = decode16(b1), c = decode32(cbits);
    const uint32_t p0 = x0.significand * y0.significand;
    const uint32_t p1 = x1.significand * y1.significand;
    const int ep0 = x0.exponent + y0.exponent, ep1 = x1.exponent + y1.exponent;
    const bool sign0 = x0.sign != y0.sign, sign1 = x1.sign != y1.sign;
    int e = c.exponent - 2;
    if (p0 && ep0 > e) e = ep0;
    if (p1 && ep1 > e) e = ep1;
    int32_t sum = align_signed(c.significand, c.exponent + 1 - e, c.sign);
    sum += align_signed(p0, ep0 + 10 - e, sign0);
    sum += align_signed(p1, ep1 + 10 - e, sign1);
    sum += 2*int(c.sign) - int(sign0) - int(sign1);
    return round_result(sum, e - 24);
}

} // namespace catena_wmma_sim
#undef CATENA_WMMA_NOINLINE
