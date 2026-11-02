// Test-only BF16 differential validator. Production code is prepended by tests.rs.
#include <algorithm>
#include <array>
#include <cfenv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace catena_wmma_sim;
#include "reference64.h"
constexpr uint32_t QNAN = 0x7fc00000u;
#include "reference.h"
#include "corpus.h"

uint32_t simulate_output(const uint16_t* a, const uint16_t* b, uint32_t c) {
    for (int k = 0; k < 16; k += 2) c = simulate_dot2(a[k], b[k], a[k+1], b[k+1], c);
    return c;
}
struct Options {
    uint64_t trials = 1000, seed = 20261028, batch = 128;
    std::string report, replay;
};
uint64_t number(const std::string& s) {
    if (s.empty() || s.front() == '-') throw std::runtime_error("expected nonnegative integer: " + s);
    size_t n = 0; const auto value = std::stoull(s, &n, 0);
    if (n != s.size()) throw std::runtime_error("invalid integer: " + s);
    return value;
}
Options options(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (++i == argc) throw std::runtime_error("missing value for " + arg);
        const std::string value = argv[i];
        if (arg == "--trials") o.trials = number(value);
        else if (arg == "--seed") o.seed = number(value);
        else if (arg == "--batch") o.batch = number(value);
        else if (arg == "--report") o.report = value;
        else if (arg == "--replay") o.replay = value;
        else throw std::runtime_error("unknown argument: " + arg);
    }
    if (!o.batch || o.batch > 4096) throw std::runtime_error("batch must be 1..4096");
    return o;
}
std::string json_string(const std::string& s) {
    std::ostringstream out; out << '"';
    for (unsigned char c : s) {
        if (c == '"' || c == '\\') out << '\\' << char(c);
        else if (c < 32) out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << int(c) << std::dec;
        else out << char(c);
    }
    out << '"'; return out.str();
}
void save_failure(const Tile& tile, int repetitions, const std::string& path) {
    std::ofstream file(path);
    file << "CATENA_WMMA_BF16_V1 " << repetitions << '\n';
    for (auto x : tile.a) file << std::hex << x << ' ';
    file << '\n'; for (auto x : tile.bt) file << std::hex << x << ' ';
    file << '\n'; for (auto x : tile.c) file << std::hex << x << ' ';
    file << '\n';
    if (!file) throw std::runtime_error("cannot save failure tile: " + path);
}
std::pair<Tile, int> read_failure(const std::string& path) {
    std::ifstream file(path); std::string magic; int repetitions;
    if (!(file >> magic >> repetitions) || magic != "CATENA_WMMA_BF16_V1" || repetitions < 1 || repetitions > 4096)
        throw std::runtime_error("invalid replay header");
    Tile tile; uint64_t value;
    for (auto* array : {&tile.a, &tile.bt}) for (auto& x : *array) {
        if (!(file >> std::hex >> value) || value > 65535) throw std::runtime_error("invalid replay BF16");
        x = uint16_t(value);
    }
    for (auto& x : tile.c) {
        if (!(file >> std::hex >> value) || value > 0xffffffffu) throw std::runtime_error("invalid replay FP32");
        x = uint32_t(value);
    }
    return {tile, repetitions};
}

#ifndef WMMA_CPU_ONLY
void hip_check(hipError_t status, const char* operation) {
    if (status != hipSuccess) throw std::runtime_error(std::string(operation) + ": " + hipGetErrorString(status));
}
template<class T> struct DeviceBuffer {
    T* data = nullptr;
    explicit DeviceBuffer(size_t count) { hip_check(hipMalloc(reinterpret_cast<void**>(&data), count*sizeof(T)), "allocate"); }
    ~DeviceBuffer() { if (data) (void)hipFree(data); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    catena_mem_ref_t view(size_t count) { return {data, count*sizeof(T)}; }
};
#if WMMA_AMD_GPU
// Independent oracle: route global tiles straight into the hardware intrinsic.
// No production fragment adapter or arithmetic helper is used here.
__global__ void hardware_wmma(const uint16_t* a, const uint16_t* b, const uint32_t* c,
                              uint32_t* d, int repetitions) {
    using Short16 = short __attribute__((ext_vector_type(16)));
    using Float8 = float __attribute__((ext_vector_type(8)));
    const int base = blockIdx.x*256, lane = threadIdx.x;
    const int column = lane%16, parity = lane/16;
    Short16 av, bv; Float8 cv;
    for (int k = 0; k < 16; ++k) {
        av[k] = static_cast<short>(a[base + column*16 + k]);
        bv[k] = static_cast<short>(b[base + column*16 + k]);
    }
    for (int i = 0; i < 8; ++i) {
        const uint32_t bits = c[base + (parity+2*i)*16 + column];
        float value; __builtin_memcpy(&value, &bits, 4); cv[i] = value;
    }
    for (int r = 0; r < repetitions; ++r)
        cv = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(av, bv, cv);
    for (int i = 0; i < 8; ++i) {
        float value = cv[i]; uint32_t bits; __builtin_memcpy(&bits, &value, 4);
        d[base + (parity+2*i)*16 + column] = bits;
    }
}
#endif
#endif

struct Result {
    std::string suite;
    uint64_t seed = 0, tiles = 0, outputs = 0, nan_outputs = 0, diagonal_cases = 0;
    int repetitions = 1;
};
class Validator {
    Options opt;
    std::vector<Result> results;
    uint64_t golden_checks = 0, rounding_checks = 0;
    std::string device = "CPU only", arch = "none";
public:
    explicit Validator(Options o) : opt(std::move(o)) {
#ifndef WMMA_CPU_ONLY
        int count = 0; hip_check(hipGetDeviceCount(&count), "GPU count");
        if (!count) throw std::runtime_error("GPU stress requested but no GPU available");
        hip_check(hipSetDevice(0), "select GPU");
        hipDeviceProp_t prop{}; hip_check(hipGetDeviceProperties(&prop, 0), "GPU properties");
        device = prop.name;
#if WMMA_CUDA
        arch = "sm_" + std::to_string(prop.major*10 + prop.minor);
#else
        arch = prop.gcnArchName;
        if (arch.rfind("gfx110", 0) != 0 && arch.rfind("gfx115", 0) != 0)
            throw std::runtime_error("native oracle requires RDNA3/3.5, got " + arch);
#endif
#endif
        std::cout << "Device: " << device << " (" << arch << "); strict raw-bit comparisons\n";
    }
    void compare(Result& result, const std::vector<Tile>& tiles) {
        const size_t count = tiles.size(), elements = count*256;
        std::vector<uint16_t> a(elements), b(elements);
        std::vector<uint32_t> c(elements), cpu(elements), reference(elements), integer64(elements);
        for (size_t t = 0; t < count; ++t) {
            std::copy(tiles[t].a.begin(), tiles[t].a.end(), a.begin()+t*256);
            std::copy(tiles[t].bt.begin(), tiles[t].bt.end(), b.begin()+t*256);
            std::copy(tiles[t].c.begin(), tiles[t].c.end(), c.begin()+t*256);
            for (int cell = 0; cell < 256; ++cell) {
                const size_t index = t*256 + cell;
                const auto* av = a.data()+t*256+(cell/16)*16;
                const auto* bv = b.data()+t*256+(cell%16)*16;
                cpu[index] = reference[index] = integer64[index] = c[index];
                for (int r = 0; r < result.repetitions; ++r) {
                    cpu[index] = simulate_output(av, bv, cpu[index]);
                    reference[index] = reference_output(av, bv, reference[index]);
                    for (int k = 0; k < 16; k += 2)
                        integer64[index] = catena_wmma_reference64::simulate_dot2(
                            av[k], bv[k], av[k+1], bv[k+1], integer64[index]);
                }
            }
        }
        auto check = [&](const std::vector<uint32_t>& actual, const char* backend) {
            for (size_t i = 0; i < elements; ++i) if (cpu[i] != actual[i]) {
                save_failure(tiles[i/256], result.repetitions, "wmma-failure.txt");
                std::ostringstream error;
                error << "MISMATCH " << result.suite << " backend=" << backend << " tile=" << result.tiles+i/256
                      << " row=" << (i%256)/16 << " col=" << i%16 << std::hex
                      << " expected=0x" << cpu[i] << " actual=0x" << actual[i]
                      << "; replay saved to wmma-failure.txt";
                throw std::runtime_error(error.str());
            }
        };
        check(reference, "FP64 reference");
        check(integer64, "original integer64 reference");
#ifndef WMMA_CPU_ONLY
        DeviceBuffer<uint16_t> da(elements), db(elements);
        DeviceBuffer<uint32_t> dc(elements), dd(elements);
        hip_check(hipMemcpy(da.data, a.data(), elements*2, hipMemcpyHostToDevice), "upload A");
        hip_check(hipMemcpy(db.data, b.data(), elements*2, hipMemcpyHostToDevice), "upload B");
        hip_check(hipMemcpy(dc.data, c.data(), elements*4, hipMemcpyHostToDevice), "upload C");
        std::vector<uint32_t> actual(elements);
#if WMMA_AMD_GPU
        hardware_wmma<<<count,32>>>(da.data, db.data, dc.data, dd.data, result.repetitions);
        hip_check(hipGetLastError(), "launch independent native oracle");
        hip_check(hipDeviceSynchronize(), "finish native oracle");
        hip_check(hipMemcpy(actual.data(), dd.data, elements*4, hipMemcpyDeviceToHost), "download native oracle");
        check(actual, "native oracle");
#endif
        // The SAME compiled host launcher/kernel is reused for every mode and
        // geometry. Sentinel output catches missing writes. CUDA cannot run native.
        const char* modes[] = {
#if WMMA_AMD_GPU
            nullptr, "1", "0", "1", "invalid", "1",
#else
            "1",
#endif
        };
        for (const char* mode : modes) {
            if (mode ? setenv("CATENA_WMMA_EMULATE", mode, 1) : unsetenv("CATENA_WMMA_EMULATE"))
                throw std::runtime_error("cannot change WMMA environment in stress subprocess");
            for (unsigned shape = 0; shape < 3; ++shape) {
                std::fill(actual.begin(), actual.end(), 0xdeadbeefu);
                hip_check(hipMemcpy(dd.data, actual.data(), elements*4, hipMemcpyHostToDevice), "sentinel output");
                catena_stress_launch(da.view(elements), db.view(elements), dc.view(elements), dd.view(elements), count, result.repetitions, shape);
                hip_check(hipMemcpy(actual.data(), dd.data, elements*4, hipMemcpyDeviceToHost), "download generated launch");
                check(actual, mode && std::string(mode) == "1" ? "generated simulation" : "generated native");
            }
        }
#endif
        for (auto bits : cpu) result.nan_outputs += (bits & 0x7fffffffu) > INF;
        result.tiles += count; result.outputs += elements;
    }
    void finish(Result result) {
        results.push_back(result);
        std::cout << result.suite << ": " << result.outputs << " positions, " << result.nan_outputs
                  << " NaNs, 0 mismatches\n" << std::flush;
    }
#include "suites.h"
    void rounding_boundaries() {
        auto check = [&](int32_t value, int power) {
            ++rounding_checks;
            if (round_result(value, power) != catena_wmma_reference64::round_result(value, power))
                throw std::runtime_error("32-bit GPU rounding differs from original integer oracle");
        };
        // Cover halfway rounding, significand/exponent carries, the exact
        // underflow cutoff, and overflow throughout the attainable range.
        for (int power = -152; power <= 230; ++power) {
            check(0, power);
            for (int high = 0; high < 28; ++high) {
                const int32_t base = int32_t(1u << high);
                const int32_t step = high > 23 ? int32_t(1u << (high-23)) : 1;
                for (int offset = -4; offset <= 4; ++offset) {
                    const int32_t value = base + offset*step + step/2;
                    if (value > 0) { check(value, power); check(-value, power); }
                }
            }
        }
        RNG rng{opt.seed};
        for (unsigned i = 0; i < 100000; ++i) {
            const int32_t value = int32_t(rng.next() & 0x0fffffffu);
            const int power = rng.range(-152, 230);
            check(value, power); check(-value, power);
        }
        std::cout << "Rounding boundaries: " << rounding_checks << " passed\n";
    }
    void run() {
        rounding_boundaries();
        golden_cases();
        if (!opt.replay.empty()) {
            const auto replay = read_failure(opt.replay);
            Result r; r.suite = "replay"; r.repetitions = replay.second;
            compare(r, {replay.first}); finish(r); return;
        }
        mapping();
        const std::array<std::string,4> labels = {"bounded", "bits", "cancellation", "exponents"};
        for (size_t i = 0; i < labels.size(); ++i) {
            const uint64_t seed = opt.seed+i; RNG rng{seed};
            suite(labels[i], opt.trials, seed, [&](uint64_t) { return random_tile(labels[i], rng); });
        }
        special(); boundary(); all_encodings();
        // Three dependent instructions, nonzero accumulators, partial final block.
        for (const std::string family : {"bounded", "bits"}) {
            Result repeated; repeated.suite = "repeated_wmma_" + family;
            repeated.seed = opt.seed; repeated.repetitions = 3;
            RNG rng{opt.seed}; std::vector<Tile> tiles;
            for (int i = 0; i < 7; ++i) tiles.push_back(random_tile(family, rng));
            compare(repeated, tiles); finish(repeated);
        }
    }
    void report(double seconds, bool passed, const std::string& error) {
        uint64_t outputs = 0, nans = 0;
        for (const auto& r : results) { outputs += r.outputs; nans += r.nan_outputs; }
        std::ostringstream j;
        j << "{\n  \"passed\": " << (passed ? "true" : "false")
          << ",\n  \"source_sha256\": " << json_string(WMMA_SOURCE_SHA256)
          << ",\n  \"compiler\": " << json_string(__VERSION__)
          << ",\n  \"device\": " << json_string(device) << ",\n  \"architecture\": " << json_string(arch)
          << ",\n  \"hardware_verified\": "
#if defined(WMMA_AMD_GPU) && WMMA_AMD_GPU
          << (passed && outputs ? "true" : "false")
#else
          << "false"
#endif
          << ",\n  \"strict_nan_bits\": true,\n  \"fp64_reference_enabled\": true"
          << ",\n  \"seed\": " << opt.seed << ",\n  \"trials_per_random_family\": " << opt.trials
          << ",\n  \"batch\": " << opt.batch << ",\n  \"golden_checks\": " << golden_checks
          << ",\n  \"rounding_checks\": " << rounding_checks
          << ",\n  \"bitwise_checked_output_positions\": " << outputs
          << ",\n  \"nan_bitwise_outputs\": " << nans
          << ",\n  \"elapsed_seconds\": " << seconds << ",\n  \"error\": " << json_string(error)
          << ",\n  \"results\": [";
        for (size_t i = 0; i < results.size(); ++i) {
            const auto& r = results[i];
            j << (i ? ",\n    " : "\n    ") << "{\"suite\":" << json_string(r.suite)
              << ",\"seed\":" << r.seed << ",\"tiles\":" << r.tiles << ",\"outputs\":" << r.outputs
              << ",\"repetitions\":" << r.repetitions << ",\"nan_outputs\":" << r.nan_outputs
              << ",\"diagonal_cases\":" << r.diagonal_cases << ",\"mismatches\":0}";
        }
        j << "\n  ]\n}\n";
        if (!opt.report.empty()) {
            std::ofstream file(opt.report); file << j.str();
            if (!file) throw std::runtime_error("cannot write report: " + opt.report);
        }
        std::cout << (passed ? "PASS: " : "FAIL: ") << outputs << " positions, " << nans << " NaN bits; " << seconds << "s\n";
#if defined(WMMA_AMD_GPU) && WMMA_AMD_GPU
        std::cout << "Native RDNA WMMA oracle and runtime switching checked.\n";
#else
        std::cout << "Native RDNA hardware equivalence was NOT checked.\n";
#endif
    }
};
int main(int argc, char** argv) {
    try {
        if (std::fesetround(FE_TONEAREST)) throw std::runtime_error("cannot select nearest-even rounding");
        Validator validator(options(argc, argv));
        const auto start = std::chrono::steady_clock::now();
        try { validator.run(); }
        catch (const std::exception& e) {
            validator.report(std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count(), false, e.what());
            std::cerr << e.what() << '\n'; return 1;
        }
        validator.report(std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count(), true, "");
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 2; }
}
