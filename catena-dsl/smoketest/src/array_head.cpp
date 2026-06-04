#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "../../report/gpu/program.cpp"

#ifndef __HIP_DEVICE_COMPILE__
static inline void array_head_hip_check(hipError_t err) {
  if (err != hipSuccess) {
    std::cerr << "HIP error: " << hipGetErrorString(err) << std::endl;
    std::abort();
  }
}

int main() {
  std::vector<uint64_t> values = {
      0x123456789abcdef0ULL,
      7,
      11,
  };

  std::cout << "making array with elems:";
  for (uint64_t value : values) {
    std::cout << " 0x" << std::hex << value;
  }
  std::cout << std::dec << std::endl;

  uint64_t *device_values = nullptr;
  array_head_hip_check(hipMallocManaged(
      reinterpret_cast<void **>(&device_values),
      values.size() * sizeof(uint64_t)));
  for (size_t i = 0; i < values.size(); ++i) {
    device_values[i] = values[i];
  }

  catena_mem_t mem = {
      device_values,
      values.size() * sizeof(uint64_t),
  };

  uint64_t out = 0;
  program_array_head_u64(mem, &out);

  std::cout << "array_head_u64: 0x" << std::hex << out << std::dec << std::endl;

  if (out != values.front()) {
    std::cerr << "verification: FAILED" << std::endl;
    array_head_hip_check(hipFree(device_values));
    return 1;
  }

  array_head_hip_check(hipFree(device_values));
  std::cout << "verification: PASSED" << std::endl;
  return 0;
}
#endif
