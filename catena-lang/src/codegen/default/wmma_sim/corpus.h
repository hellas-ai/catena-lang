// BF16 corpus adapted from the standalone RDNA3.5 validator.
struct Tile {
    std::array<uint16_t, 256> a{}, bt{};
    std::array<uint32_t, 256> c{};
};
struct RNG {
    uint64_t state;
    uint64_t next() { // specified SplitMix64: reproducible raw bits across platforms
        uint64_t z = (state += 0x9e3779b97f4a7c15ull);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
        return z ^ (z >> 31);
    }
    int range(int low, int high) { return low + int(next() % uint64_t(high - low + 1)); }
};
uint16_t encode16(bool sign, int exponent, uint32_t mantissa = 0) {
    const int fraction = 7, bias = 127;
    const int max_exp = 127;
    exponent = std::max(1 - bias, std::min(max_exp, exponent));
    return uint16_t((sign ? 0x8000 : 0) | ((exponent + bias) << fraction) |
                    (mantissa & ((1 << fraction) - 1)));
}
uint32_t encode32(bool sign, int exponent, uint32_t mantissa = 0) {
    exponent = std::max(-126, std::min(127, exponent));
    return (sign ? 0x80000000u : 0) | (uint32_t(exponent + 127) << 23) | (mantissa & 0x7fffff);
}
Tile random_tile(const std::string& suite, RNG& rng) {
    Tile tile;
    const int targets[] = {-150, -127, -126, -125, -1, 0, 1, 126, 127, 128, 129};
    const int offsets[] = {-25, -24, -23, -1, 0, 1, 23};
    const int low = -126, high = 127;
    const int target = targets[rng.range(0, 10)], ea = rng.range(low, high);
    const int eb = std::max(low, std::min(high, target - ea));
    const int ec = target + offsets[rng.range(0, 6)];
    for (int i = 0; i < 256; ++i) {
        if (suite == "bits") {
            tile.a[i] = uint16_t(rng.next()); tile.bt[i] = uint16_t(rng.next()); tile.c[i] = uint32_t(rng.next());
        } else {
            const bool extreme = suite == "exponents";
            // Sequence RNG draws explicitly: C++ argument evaluation order
            // otherwise makes GCC and Clang generate different input corpora.
            const bool as = rng.next() & 1;
            const int ae = extreme ? ea : rng.range(-4, 3);
            const uint32_t am = uint32_t(rng.next());
            const bool bs = rng.next() & 1;
            const int be = extreme ? eb : rng.range(-4, 3);
            const uint32_t bm = uint32_t(rng.next());
            const bool cs = rng.next() & 1;
            const int ce = extreme ? ec : rng.range(-4, 3);
            const uint32_t cm = uint32_t(rng.next());
            tile.a[i] = encode16(as, ae, am);
            tile.bt[i] = encode16(bs, be, bm);
            tile.c[i] = encode32(cs, ce, cm);
        }
    }
    if (suite == "cancellation") {
        tile.c.fill(0);
        for (int i = 0; i < 256; i += 2) {
            tile.a[i + 1] = tile.a[i] ^ 0x8000;
            tile.bt[i + 1] = tile.bt[i];
        }
    }
    return tile;
}

