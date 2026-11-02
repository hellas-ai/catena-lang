// Separate floating-point implementation, used only as a CPU cross-check.
// It shares the mathematical model, not the integer alignment/rounding code.
uint32_t as_bits(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
double as_double(uint16_t bits) {
    const auto x = decode16(bits);
    if (x.nan) return std::numeric_limits<double>::quiet_NaN();
    if (x.infinity) return x.sign ? -INFINITY : INFINITY;
    constexpr int fraction = 7;
    return std::copysign(std::ldexp(double(x.significand), x.exponent - fraction), x.sign ? -1. : 1.);
}
uint32_t reference_output(const uint16_t* a, const uint16_t* b, uint32_t cbits) {
    for (int k = 0; k < 16; k += 2) {
        const double av[2] = {as_double(a[k]), as_double(a[k + 1])};
        const double bv[2] = {as_double(b[k]), as_double(b[k + 1])};
        const double p[2] = {av[0] * bv[0], av[1] * bv[1]};
        const Decoded cd = decode32(cbits);
        const double c = cd.nan ? NAN : cd.infinity ? (cd.sign ? -INFINITY : INFINITY) :
            std::copysign(std::ldexp(double(cd.significand), cd.exponent - 23), cd.sign ? -1. : 1.);
        if (!std::isfinite(p[0]) || !std::isfinite(p[1]) || !std::isfinite(c)) {
            const bool input_nan = decode16(a[k]).nan || decode16(a[k+1]).nan ||
                                   decode16(b[k]).nan || decode16(b[k+1]).nan;
            if (input_nan) cbits = OPERAND_NAN;
            else if (std::isnan(p[0]) || std::isnan(p[1]) || std::isnan(p[0]+p[1])) cbits = INVALID_NAN;
            else if (cd.nan) cbits |= 0x00400000u;
            else if (std::isnan((p[0]+p[1])+c)) cbits = INVALID_NAN;
            else cbits = as_bits(float((p[0]+p[1])+c));
            continue;
        }
        int e = cd.exponent - 2;
        for (int j = 0; j < 2; ++j) {
            if (p[j] != 0) {
                const auto da = decode16(a[k + j]), db = decode16(b[k + j]);
                e = std::max(e, da.exponent + db.exponent);
            }
        }
        const double scale = std::ldexp(1., 24 - e);
        double s = std::trunc(p[0] * scale) + std::trunc(p[1] * scale) + std::trunc(c * scale);
        s -= int(std::signbit(p[0])) + int(std::signbit(p[1]));
        s += 2 * int(std::signbit(c));
        double result = s / scale;
        if (std::abs(result) < std::ldexp(1., -126) - std::ldexp(1., -151)) result = 0.;
        cbits = as_bits(float(result));
    }
    return cbits;
}

