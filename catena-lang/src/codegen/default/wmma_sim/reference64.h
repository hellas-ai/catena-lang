// Test-only original 64-bit standalone arithmetic oracle.
// Empirical gfx1151 model; keep all rounding and exceptional values bit-exact.
namespace catena_wmma_reference64 {
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
__host__ __device__ inline uint64_t align_unsigned(uint64_t significand, int shift) {
    if (!significand) return 0;
    if (shift >= 0) return significand << shift; // bounded by 27 bits here
    return shift <= -64 ? 0 : significand >> -shift;
}
__host__ __device__ inline int64_t align_signed(uint64_t significand, int shift, bool sign) {
    const int64_t aligned = static_cast<int64_t>(align_unsigned(significand, shift));
    return sign ? -aligned : aligned; // truncation toward zero
}
__host__ __device__ inline int msb(uint64_t x) { // x != 0
#ifdef __CUDA_ARCH__
    return 63 - __clzll(static_cast<unsigned long long>(x));
#else
    return 63 - __builtin_clzll(x);
#endif
}

// Convert S * 2**power to raw FP32, with explicit RNE and measured underflow.
__host__ __device__ inline uint32_t round_result(int64_t s, int power) {
    if (!s) return 0; // arithmetic zeros always +0
    const bool negative = s < 0;
    const uint64_t magnitude = negative ? uint64_t(-s) : uint64_t(s);
    const uint32_t sign = negative ? 0x80000000u : 0;
    const int high = msb(magnitude);
    int exponent = high + power;
    if (exponent < -126) {
        if (exponent < -127) return 0;
        // Compare exactly with (2**25 - 1) * 2**-151.
        // E >= -128, hence power+151 >= -1. At exponent -127
        // both sides of this comparison fit within 27 bits.
        const int shift = power + 151;
        const uint64_t threshold = (uint64_t(1) << 25) - 1;
        const bool below = shift >= 0 ? (magnitude << shift) < threshold
                                      : magnitude < (threshold << -shift);
        return below ? 0 : sign | 0x00800000u;
    }
    const int shift = high - 23;
    uint64_t mantissa;
    if (shift > 0) {
        mantissa = magnitude >> shift;
        const uint64_t remainder = magnitude & ((uint64_t(1) << shift) - 1);
        const uint64_t half = uint64_t(1) << (shift - 1);
        if (remainder > half || (remainder == half && (mantissa & 1))) ++mantissa;
    } else {
        mantissa = magnitude << -shift;
    }
    if (mantissa == (uint64_t(1) << 24)) { mantissa >>= 1; ++exponent; }
    if (exponent > 127) return sign | INF;
    return sign | (uint32_t(exponent + 127) << 23) | (uint32_t(mantissa) & 0x7fffff);
}

__host__ __device__ inline uint32_t simulate_dot2(uint16_t a0, uint16_t b0, uint16_t a1, uint16_t b1, uint32_t cbits) {
    constexpr int fraction = 7;
    const Decoded a[2] = {decode16(a0), decode16(a1)};
    const Decoded b[2] = {decode16(b0), decode16(b1)};
    const Decoded c = decode32(cbits);
    bool positive_inf = c.infinity && !c.sign;
    bool negative_inf = c.infinity && c.sign;
    if (a[0].nan || a[1].nan || b[0].nan || b[1].nan) return OPERAND_NAN;
    uint64_t product[2];
    bool sign[2];
    int ep[2];
    int e = c.exponent - 2;
    for (int i = 0; i < 2; ++i) {
        sign[i] = a[i].sign != b[i].sign;
        const bool invalid = (a[i].infinity && !b[i].significand && !b[i].infinity) ||
                             (b[i].infinity && !a[i].significand && !a[i].infinity);
        if (invalid) return INVALID_NAN;
        if (a[i].infinity || b[i].infinity) {
            positive_inf |= !sign[i]; negative_inf |= sign[i];
        }
        product[i] = uint64_t(a[i].significand) * b[i].significand;
        ep[i] = a[i].exponent + b[i].exponent; // deliberately NOT normalized
        if (product[i] && ep[i] > e) e = ep[i];
    }
    if (positive_inf && negative_inf) return INVALID_NAN;
    if (c.nan) return cbits | 0x00400000u;
    if (positive_inf || negative_inf) return (negative_inf ? 0x80000000u : 0) | INF;
    int64_t sum = align_signed(c.significand, c.exponent + 1 - e, c.sign);
    for (int i = 0; i < 2; ++i)
        sum += align_signed(product[i], ep[i] - 2 * fraction + 24 - e, sign[i]);
    sum -= int(sign[0]) + int(sign[1]); // includes signed zero products
    sum += 2 * int(c.sign);           // includes a signed zero accumulator
    return round_result(sum, e - 24);
}

} // namespace catena_wmma_reference64
