#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>

typedef uint8_t catena_unit_t;
typedef uint8_t catena_gpu_state_t;

typedef struct {
    uint32_t x;
    uint32_t y;
    uint32_t z;
} catena_dim3_t;

typedef struct {
    uint64_t thread_id;
} catena_gpu_env_t;

typedef struct {
    catena_dim3_t grid_dim;
    catena_dim3_t block_dim;
} catena_launch_params_t;

typedef struct {
    void *data;
    uint64_t len;
} catena_mem_t;

typedef struct {
    void *data;
    uint64_t len;
} catena_gpu_buf_t;

__host__ __device__ static inline void catena_assert(uint8_t condition) {
    if (!condition) {
#ifndef __HIP_DEVICE_COMPILE__
        fprintf(stderr, "catena assertion failed\n");
        fflush(stderr);
#endif
        __builtin_trap();
    }
}

#ifndef __HIP_DEVICE_COMPILE__
static inline void catena_hip_check(hipError_t err) {
    if (err != hipSuccess) {
        fprintf(stderr, "catena HIP error: %s\n", hipGetErrorString(err));
        fflush(stderr);
        __builtin_trap();
    }
}

#endif

__host__ __device__ static inline uint64_t catena_launch_len(catena_launch_params_t params) {
    return (uint64_t)params.grid_dim.x * params.grid_dim.y * params.grid_dim.z
        * params.block_dim.x * params.block_dim.y * params.block_dim.z;
}

__host__ __device__ static inline void bool_not(uint8_t arg0, uint8_t *out1) {
    *out1 = !arg0;
}

__host__ __device__ static inline void bool_or(uint8_t arg0, uint8_t arg1, uint8_t *out2) {
    *out2 = arg0 || arg1;
}

__host__ __device__ static inline void bool_and(uint8_t arg0, uint8_t arg1, uint8_t *out2) {
    *out2 = arg0 && arg1;
}

__host__ __device__ static inline void bool_id(uint8_t arg0, uint8_t *out1) {
    *out1 = arg0;
}

__host__ __device__ static inline void bool_copy(uint8_t arg0, uint8_t *out1, uint8_t *out2) {
    *out1 = arg0;
    *out2 = arg0;
}

__host__ __device__ static inline void bool_li(uint8_t arg0, uint8_t *out1) {
    *out1 = arg0;
}

extern "C" __host__ __device__ void program_assert_then(uint8_t x0);
extern "C" __host__ __device__ void program_bool_li(uint8_t x0, uint8_t *out_x0);
extern "C" __host__ __device__ void program_plain_ifc(uint8_t x0, uint8_t x1, uint8_t *out_x6);
extern "C" __host__ __device__ void program_u64_assert_nz(uint64_t x0);

extern "C" __host__ __device__ void program_assert_then(uint8_t x0) {
    catena_assert(x0);
    return;
}

extern "C" __host__ __device__ void program_bool_li(uint8_t x0, uint8_t *out_x0) {
    *out_x0 = x0;
    return;
}

extern "C" __host__ __device__ void program_plain_ifc(uint8_t x0, uint8_t x1, uint8_t *out_x6) {
    uint8_t x2;
    x2 = 1;
    uint8_t x4;
    x4 = 0;
    uint8_t x6;
    if (x0) { plain_then(x2, x1, &x6); } else { plain_else(x4, x1, &x6); }
    *out_x6 = x6;
    return;
}

extern "C" __host__ __device__ void program_u64_assert_nz(uint64_t x0) {
    uint64_t x1;
    x1 = 0;
    uint8_t x2;
    x2 = x0 > x1;
    program_assert_then(x2);
    return;
}

